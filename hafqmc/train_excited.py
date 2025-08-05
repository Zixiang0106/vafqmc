import logging
import jax 
import optax
import os
from jax import numpy as jnp
from optax._src import alias as optax_alias
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter
from typing import NamedTuple

from .molecule import build_mf
from .hamiltonian import Hamiltonian, HamiltonianPW, Hamiltonian_sym
from .ansatz import Ansatz, BraKet
from .estimator import make_eval_total
from .ovlp import make_ovlp_total
from .sampler import make_sampler, make_multistep, make_batched, SamplerUnion
from .utils import ensure_mapping, save_pickle, load_pickle, Printer, cfg_to_yaml
from .utils import make_moving_avg, PyTree, tree_map
from flax import jax_utils
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
def log_once(logger, process_index, message, level='info'):
    """Only log a message once per process."""
    if process_index == 0:
        getattr(logger, level)(message)


def lower_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(target - s, 0) ** power

def upper_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(s - target, 0) ** power


def make_optimizer(name, lr_schedule, grad_clip=None, **kwargs):
    opt_fn = getattr(optax_alias, name)
    opt = opt_fn(lr_schedule, **kwargs)
    if grad_clip is not None:
        opt = optax.chain(optax.clip(grad_clip), opt)
    return opt


def make_lr_schedule(start=1e-4, decay=1., delay=1e4):
    if decay is None:
        return start
    return lambda t: start * jnp.power((1.0 / (1.0 + (t/delay))), decay)



def make_loss(expect_fn, ovlp_fn, num_states, weights,
              sign_factor=0., sign_target=1., sign_power=2.,
              std_factor=0., std_target=1., std_power=2,):
    def loss(params, data, params_lower, data_lower, *extra, **kwargs):
        e_tot, aux = expect_fn(params, data, *extra, **kwargs)
        loss = e_tot
        ovlps = []
        s_ovlps = []
        energies = []
        if sign_factor > 0:
            exp_s = aux["exp_s"]
            loss += lower_penalty(exp_s, sign_factor, sign_target, sign_power)
        if std_factor > 0:
            std_es = aux["std_es"]
            loss += upper_penalty(std_es, std_factor, std_target, std_power)
        for i in range(num_states):
            ovlp, aux_ovlp = ovlp_fn(params_lower[i], params, data_lower[i], data)
            s_ovlp = aux_ovlp["S_ab"] * aux_ovlp["S_ba"]
            e, _ = expect_fn(params_lower[i], data_lower[i], *extra, **kwargs)
            loss += weights[i] * ovlp
            if sign_factor > 0:
                loss += lower_penalty(s_ovlp, sign_factor, sign_target, sign_power)
            s_ovlps.append(s_ovlp)
            ovlps.append(ovlp)
            energies.append(e)
        for i in range(num_states):
            aux[f'e{i}'] = energies[i]
            aux[f"ovlp{i}"] = ovlps[i]
            aux[f's{i}'] = s_ovlps[i]
        return loss, aux
    return loss



class TrainingState(NamedTuple):
    step: int
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree
    est_state: PyTree
    params_lower: PyTree
    mc_state_lower: PyTree

class TrainingStateSave(NamedTuple):
    step: int
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree
    est_state: PyTree = None


def make_training_step(loss_and_grad, mc_sampler, optimizer, accumulator=None, num_states=1):
    is_union = isinstance(mc_sampler, SamplerUnion)

    def step(key, train_state, sample_flag=None):
        keys = jax.random.split(key, num_states)
        ii, params, mc_state, opt_state, ebar, params_lower, mc_state_lower = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state = sampler.refresh(mc_state, params)
        mc_state, data = sampler.sample(key, params, mc_state)
        data_lower = []
        mc_state_new = []
        for i in range(num_states):
            mc_statei = sampler.refresh(mc_state_lower[i], params_lower[i])
            mc_statei, datai = sampler.sample(keys[i], params_lower[i], mc_statei)
            data_lower.append(datai)
            mc_state_new.append(mc_statei)
        (loss, aux), grads = loss_and_grad(params, data, params_lower, data_lower, ebar)
        grads = tree_map(jnp.conj, grads) # for complex parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if accumulator is not None: ebar = accumulator(ebar, aux["e_tot"], ii)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar, params_lower, mc_state_new)
        return new_state, (loss, aux)

    return step

def make_training_step_pmap(loss_and_grad, mc_sampler, optimizer, accumulator=None, num_states=1):
    #multi-gpu training step
    is_union = isinstance(mc_sampler, SamplerUnion)

    def step(key, train_state, sample_flag=None):
        keys = jax.random.split(key, num_states)
        ii, params, mc_state, opt_state, ebar, params_lower, mc_state_lower = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state = sampler.refresh(mc_state, params)
        mc_state, data = sampler.sample(key, params, mc_state)
        data_lower = []
        mc_state_new = []
        for i in range(num_states):
            mc_statei = sampler.refresh(mc_state_lower[i], params_lower[i])
            mc_statei, datai = sampler.sample(keys[i], params_lower[i], mc_statei)
            data_lower.append(datai)
            mc_state_new.append(mc_statei)
        (loss, aux), grads = loss_and_grad(params, data, params_lower, data_lower, ebar)
        grads = tree_map(jnp.conj, grads) # for complex parameters
        
        # sync grads in multi-gpu environment
        grads = jax.lax.pmean(grads, axis_name="device")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if accumulator is not None: ebar = accumulator(ebar, aux["e_tot"], ii)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar, params_lower, mc_state_new)
        return new_state, (loss, aux)

    return step


def make_evaluation_step(expect_fn, ovlp_fn, mc_sampler, num_states):
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        keys = jax.random.split(key, num_states)
        ii, params, mc_state, opt_state, ebar, params_lower, mc_state_lower = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        ovlps = []
        s_ovlps = []
        energies = []
        mc_state_new = []
        for i in range(num_states):
            mc_statei = sampler.refresh(mc_state_lower[i], params_lower[i])
            mc_statei, datai = sampler.sample(keys[i], params[i], mc_statei)
            ovlp, aux_ovlp = ovlp_fn(params_lower[i], datai, params, data)
            s_ovlp = aux_ovlp["S_ab"] * aux_ovlp["S_ba"]
            e, _ = expect_fn(params_lower[i], datai)
            s_ovlps.append(s_ovlp)
            ovlps.append(ovlp)
            energies.append(e)
            mc_state_new.append(mc_statei)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar, params_lower, mc_state_new)
        for i in range(num_states):
            aux[f"e{i}"] = energies[i]
            aux[f"ovlp{i}"] = ovlps[i]
            aux[f"s{i}"] = s_ovlps[i]
        return new_state, (e_tot, aux)
    
    return step
        

def make_evaluation_step_pmap(expect_fn, ovlp_fn, mc_sampler, num_states):
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        keys = jax.random.split(key, num_states)
        ii, params, mc_state, opt_state, ebar, params_lower, mc_state_lower = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        e_tot = jax.lax.pmean(e_tot, axis_name="device")
        aux = jax.lax.pmean(aux, axis_name="device")
        ovlps = []
        energies = []
        mc_state_new = []
        for i in range(num_states):
            mc_statei = sampler.refresh(mc_state_lower[i], params_lower[i])
            mc_statei, datai = sampler.sample(keys[i], params_lower[i], mc_statei)
            ovlp, _ = ovlp_fn(params_lower[i], datai, params, data)
            e, _ = expect_fn(params_lower[i], datai)
            ovlp = jax.lax.pmean(ovlp, axis_name="device")
            ovlps.append(ovlp)
            energies.append(e)
            mc_state_new.append(mc_statei)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar, params_lower, mc_state_new)
        for i in range(num_states):
            aux[f"e{i}"] = energies[i]
            aux[f"ovlp{i}"] = ovlps[i]
        return new_state, (e_tot, aux)
    
    return step
def train(cfg: ConfigDict):
    # handle logging
    logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger("train")
    log_level = getattr(logging, cfg.log.level.upper())
    logger.setLevel(log_level)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    n_visible_gpus = len(cuda_visible.split(","))
    jax.distributed.initialize(local_device_ids=list(range(n_visible_gpus)))
    n_devices = jax.device_count()
    n_local_devices = jax.local_device_count()
    process_index = jax.process_index()
    n_processes = jax.process_count()
    is_multi_gpu = n_devices > 1
    lower_params_paths = cfg.restart.lower_state_params
    weights = cfg.restart.weights
    num_states = len(lower_params_paths)
    assert len(weights) == num_states, \
    f"weight list length must be num_state = {num_states}, got {len(weights)}"
    
    if is_multi_gpu:
        log_once(logger, process_index,f"JAX detected {n_devices} total devices across {n_processes} processes")
        log_once(logger, process_index,f"Current process {process_index} has {n_local_devices} local devices")
        log_once(logger, process_index,f"Running on {n_devices} GPUs with pmap for data parallelism")
    else:
        logger.info("Running on single device")
    if process_index == 0:
        writer = SummaryWriter(cfg.log.stat_path)
    else:
        writer = None
    print_fields = {"step": "", "loss": ".4f", "e_tot": ".4f", 
                    "exp_es": ".4f", "exp_s": ".4f"}
    if cfg.loss.std_factor >= 0:
        print_fields.update({"std_es": ".4f", "std_s": ".4f"})
    for i in range(num_states):
        print_fields[f"e{i}"] = ".4f"
        print_fields[f"ovlp{i}"] = ".4f"
        print_fields[f"s{i}"] = ".4f"
    print_fields["lr"] = ".1e"
    printer = Printer(print_fields, time_format=".4f")
    if cfg.log.hpar_path and process_index == 0:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)

    # get the constants
    total_iter = cfg.optim.iteration
    sample_size = cfg.sample.size
    sample_batch = cfg.sample.batch
    
    if is_multi_gpu:
        local_batch = sample_batch // n_devices
        if sample_batch % n_devices != 0:
            logger.warning(f"Batch size {sample_batch} not divisible by {n_devices} devices, "
                         f"adjusting to {local_batch * n_devices}")
            sample_batch = local_batch * n_devices
        local_sample_size = sample_size // n_devices
        sample_step = -(-local_sample_size // local_batch)
    else:
        local_batch = sample_batch
        sample_step = -(-sample_size // sample_batch)
    
    if is_multi_gpu:
        sample_size = local_batch * sample_step * n_devices
    else:
        if sample_size % sample_batch != 0:
            logger.warning("Sample size not divisible by batch size, rounding up")
        sample_size = sample_batch * sample_step
    
    sample_prop = cfg.sample.prop_steps
    eval_batch = cfg.optim.batch if cfg.optim.batch is not None else sample_batch
    
    if sample_size % eval_batch != 0:
        logger.warning("Eval batch size not dividing sample size, using sample batch size")
        eval_batch = sample_batch
    
    if is_multi_gpu:
        local_eval_batch = eval_batch // n_devices
        if eval_batch % n_devices != 0:
            eval_batch = local_eval_batch * n_devices
            logger.warning(f"Eval batch size {eval_batch} not divisible by {n_devices} devices, "
                         f"adjusting to {local_eval_batch * n_devices}")
    else:
        local_eval_batch = eval_batch
    log_once(logger, process_index,f"Sample configuration:")
    log_once(logger, process_index,f"  Total sample_size: {sample_size}")
    log_once(logger, process_index,f"  Sample batch: {sample_batch}")
    log_once(logger, process_index,f"  Local batch per device: {local_batch}")
    log_once(logger, process_index,f"  Sample steps: {sample_step}")
    if is_multi_gpu:
        log_once(logger, process_index,f"  Local sample size per device: {local_sample_size}")
        log_once(logger, process_index,f"  Expected samples per device: {local_batch * sample_step}")

    # set up the hamiltonian
    if cfg.restart.hamiltonian is None:
        if "ueg" not in cfg:
            log_once(logger, process_index,"Building molecule and doing HF calculation to get Hamiltonian")
            mf = build_mf(**cfg.molecule)
            if process_index == 0:
                print(f"# HF energy from pyscf calculation: {mf.e_tot}")
            if not mf.converged:
                logger.warning("HF calculation does not converge!")
            hamiltonian = Hamiltonian.from_pyscf(mf, **cfg.hamiltonian)
        else:
            log_once(logger, process_index,"Using uniform electron gas Hamiltonian")
            hamiltonian = HamiltonianPW.from_ueg(**cfg.ueg)
            if process_index == 0:
                print(f"# HF energy for UEG hamiltonian: {hamiltonian.local_energy()}")
        save_pickle(cfg.log.hamil_path, hamiltonian.to_tuple())
    else:
        log_once(logger, process_index,"Loading Hamiltonian from saved file")
        hamil_data = load_pickle(cfg.restart.hamiltonian)
        HamCls = Hamiltonian_sym
        hamiltonian = HamCls(*hamil_data)
        if process_index == 0:
            print(f"# HF energy from loaded: {hamiltonian.local_energy()}")

    # set up all other classes and functions
    log_once(logger, process_index,"Setting up the training loop")
    ansatz = Ansatz.create(hamiltonian, **cfg.ansatz)
    trial = (None if cfg.trial is None 
             else ansatz if isinstance(cfg.trial, str) and 
                    cfg.trial.lower() in ("same", "share", "ansatz")
             else Ansatz.create(hamiltonian, **cfg.trial))
    braket = BraKet(ansatz, trial)
    
    if (sample_prop is None or isinstance(sample_prop, int)
      or (isinstance(sample_prop, (tuple, list)) and len(sample_prop) == 2)):
        sampler_1s_1c = make_sampler(braket, max_prop=sample_prop,
            **ensure_mapping(cfg.sample.sampler, default_key="name"))
    else:
        sampler_1s_1c = SamplerUnion({
            mp: make_sampler(braket, max_prop=mp,
                **ensure_mapping(cfg.sample.sampler, default_key="name"))
            for mp in sample_prop})
    
    sampler_1s_nc = make_batched(sampler_1s_1c, local_batch if is_multi_gpu else sample_batch, concat=False)
    mc_sampler = make_multistep(sampler_1s_nc, sample_step, concat=True)
    lr_schedule = make_lr_schedule(**cfg.optim.lr)
    optimizer = make_optimizer(lr_schedule=lr_schedule, grad_clip=cfg.optim.grad_clip,
        **ensure_mapping(cfg.optim.optimizer, default_key="name"))
    
    expect_fn = make_eval_total(hamiltonian, braket,
        default_batch=local_eval_batch, calc_stds=True)
    ovlp_fn = make_ovlp_total(hamiltonian, braket,
        default_batch=local_eval_batch, calc_stds=True)
    
    loss_fn = make_loss(expect_fn, ovlp_fn, num_states, weights, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    moving_avg_fn = (make_moving_avg(**cfg.optim.baseline)
        if cfg.optim.baseline is not None else None)

    # the core training iteration
    if cfg.optim.lr.start > 0:
        if is_multi_gpu:
            train_step = make_training_step_pmap(loss_and_grad, mc_sampler, optimizer, moving_avg_fn, num_states=num_states)
        else:
            train_step = make_training_step(loss_and_grad, mc_sampler, optimizer, moving_avg_fn, num_states=num_states)
    else:
        log_once(logger, process_index,"Running inference mode")
        if is_multi_gpu:
            train_step = make_evaluation_step_pmap(expect_fn, mc_sampler)
        else:
            train_step = make_evaluation_step(expect_fn, mc_sampler)
    
    if is_multi_gpu:
        train_step = jax.pmap(train_step, axis_name="device", static_broadcasted_argnums=(2,))
        #train_step = jax.pmap(train_step, axis_name="device", static_broadcasted_argnums=(2,))
    else:
        train_step = jax.jit(train_step, static_argnums=(2,))
    
    # set up all states
    log_once(logger, process_index, "Loading lower states parameters from saved file")
    key = jax.random.PRNGKey(cfg.seed)
    keys = jax.random.split(key, num_states)
    params_lower = []
    mc_state_lower = []

    for i,p in enumerate(lower_params_paths):
        paramsi = load_pickle(p)
        if isinstance(paramsi, tuple): paramsi = paramsi[1]
        if isinstance(paramsi, tuple): paramsi = paramsi[1]
        mc_statei = mc_sampler.init(keys[i], paramsi)
        params_lower.append(paramsi)
        mc_state_lower.append(mc_statei)

    if cfg.restart.states is None:
        log_once(logger, process_index,"Initializing parameters and states")
        key = jax.random.PRNGKey(cfg.seed)
        key, pakey, mckey = jax.random.split(key, 3)
        fshape = braket.fields_shape()
        
        if cfg.restart.params is None:
            params = jax.jit(braket.init)(pakey, tree_map(jnp.zeros, fshape))
        else:
            log_once(logger, process_index, "Loading parameters from saved file")
            params = load_pickle(cfg.restart.params)
            if isinstance(params, tuple): params = params[1]
            if isinstance(params, tuple): params = params[1]
        mc_state = mc_sampler.init(mckey, params)
        opt_state = optimizer.init(params)
        
        if cfg.sample.burn_in > 0:
            log_once(logger, process_index, f"Burning in the {num_states} samplers for {cfg.sample.burn_in} steps")
            key, subkey = jax.random.split(key)
            mc_state = sampler_1s_nc.burn_in(subkey, params, mc_state, cfg.sample.burn_in)
            subkeys = jax.random.split(subkey, num_states)
            for i in range(num_states):
                mc_state_lower[i] = sampler_1s_nc.burn_in(subkeys[i], params_lower[i], mc_state_lower[i], cfg.sample.burn_in)
        
        ebar = hamiltonian.local_energy() if cfg.optim.baseline is not None else None
        train_state = TrainingState(0, params, mc_state, opt_state, ebar, params_lower, mc_state_lower)
        
        if is_multi_gpu:
            train_state = jax_utils.replicate(train_state)
    else:
        log_once(logger, process_index, "Loading parameters and states from saved file")
        key, *rest = load_pickle(cfg.restart.states)
        rest = rest[0] if len(rest) == 1 else (0, *rest)
        if len(rest) < 5 and cfg.optim.baseline is not None:
            rest = (*rest, hamiltonian.local_energy())
        train_state = TrainingState(*rest)
        
        if is_multi_gpu:
            train_state = jax_utils.replicate(train_state)

    if is_multi_gpu:
        base_key = jax.random.fold_in(key, process_index)
        keys = jax.random.split(base_key, n_local_devices)
        device_indices = jnp.arange(n_local_devices) + process_index * n_local_devices
        keys = jax.vmap(lambda k, idx: jax.random.fold_in(k, idx))(keys, device_indices)
        #keys = jax.random.split(key, n_devices)
    
    # the actual training iteration
    log_once(logger, process_index, "Start training")
    if process_index == 0:
        printer.print_header(prefix="# ")
    if process_index == 0 and not os.path.exists(cfg.log.ckpt_path):
        os.makedirs(cfg.log.ckpt_path, exist_ok=True)
    
    for ii in range(total_iter + 1):
        printer.reset_timer()
        
        # choose sampler
        sflag = None
        if not (sample_prop is None or isinstance(sample_prop, int)):
            if is_multi_gpu:
                keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
                keys, flagkeys = keys[:, 0], keys[:, 1]
                flagkey = flagkeys[0] 
            else:
                key, flagkey = jax.random.split(key)
            sflag = sample_prop[jax.random.choice(flagkey, len(sample_prop))]
        
        # core training step
        if is_multi_gpu:
            keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
            subkeys, keys = keys[:, 0], keys[:, 1]
            train_state, (loss, aux) = train_step(subkeys, train_state, sflag)
            loss = float(loss[0])
            aux = jax.tree_map(lambda x: float(x[0]) if jnp.isscalar(x[0]) else x[0], aux)
        else:
            key, subkey = jax.random.split(key)
            train_state, (loss, aux) = train_step(subkey, train_state, sample_flag=sflag)
        
        # logging and checkpointing
        if ii % cfg.log.stat_freq == 0:
            if sflag is not None: aux["nprop"] = sflag
            if callable(lr_schedule):
                if is_multi_gpu:
                    opt_state_first = jax.tree_map(lambda x: x[0], train_state.opt_state)
                    step_count = opt_state_first[1][1].count
                else:
                    step_count = train_state.opt_state[-1][0].count
                _lr = lr_schedule(step_count)
            else:
                _lr = lr_schedule
            if process_index == 0:
                printer.print_fields({"step": ii, "loss": loss, **aux, "lr": _lr})
            if writer is not None:
                writer.add_scalars("stat", {"loss": loss, **aux, "lr": _lr}, global_step=ii)
            
        if ii % cfg.log.ckpt_freq == 0:
            if process_index == 0:
                checkpoint_filename = f"./{cfg.log.ckpt_path}/ckpt_{ii}.pkl"
                ii, params, mc_state, opt_state, ebar, params_lower, mc_state_lower = train_state
                train_state_save = TrainingStateSave(ii, params, mc_state, opt_state, ebar)
                if is_multi_gpu:
                    unreplicated_state = jax.tree_map(lambda x: x[0], train_state_save)
                    save_pickle(checkpoint_filename, (keys[0], tuple(unreplicated_state)))
                else:
                    save_pickle(checkpoint_filename, (key, tuple(train_state_save)))
    if process_index == 0: 
        writer.close()
    return train_state