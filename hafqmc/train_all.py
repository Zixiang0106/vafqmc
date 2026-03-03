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


def make_loss(expect_fn, ovlp_fn, 
              sign_factor=0., sign_target=1., sign_power=2.,
              std_factor=0., std_target=1., std_power=2,
              num_states=1, weights=[]):
    assert len(weights) == num_states * (num_states - 1) // 2, \
    f"weight list length must be num_states*(num_states-1)//2 = {num_states*(num_states-1)//2}, got {len(weights)}"
    def single_loss(params, data, *extra, **kwargs):
        e_tot, aux = expect_fn(params, data, *extra, **kwargs)
        loss = e_tot
        if sign_factor > 0:
            exp_s = aux["exp_s"]
            loss += lower_penalty(exp_s, sign_factor, sign_target, sign_power)
        if std_factor > 0:
            std_es = aux["std_es"]
            loss += upper_penalty(std_es, std_factor, std_target, std_power)
        return loss, e_tot, aux
    def loss(params, data_list, *extra, **kwargs):
        total_loss = 0.0
        aux_list = []
        ovlp_list = []
        energy_list = []
        for i in range(num_states):
            li, ei, auxi = single_loss(params[i], data_list[i], *extra, **kwargs)
            total_loss+=li
            aux_list.append(auxi)
            energy_list.append(ei)
        k = 0
        for i in range(num_states):
            for j in range(i):
                ovlp, _ = ovlp_fn(params[j], params[i], data_list[j], data_list[i])
                ovlp_list.append(ovlp)
                total_loss+= weights[k]*ovlp
                k+=1
        aux = {"aux_list": aux_list,
               "ovlp": ovlp_list,
               "energy": energy_list}
        return total_loss, aux        
    return loss


class TrainingState(NamedTuple):
    step: int
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree
    est_state: PyTree = None



def make_training_step_multi_state(loss_and_grad, mc_sampler, optimizer, accumulator=None, num_states=1):
    is_union = isinstance(mc_sampler, SamplerUnion)
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, opt_state, ebar = train_state
        mc_states = []
        data_list = []
        keys = jax.random.split(key, num_states)
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        for i in range(num_states):
            mc_state_i = sampler.refresh(mc_state[i], params[i])
            mc_state_i, data_i = sampler.sample(keys[i], params[i], mc_state_i)
            mc_states.append(mc_state_i)
            data_list.append(data_i)
        mc_states = tuple(mc_states)
        (loss, aux), grads = loss_and_grad(params, data_list, ebar)
        grads = tree_map(jnp.conj, grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_state = TrainingState(ii+1, params, mc_states, opt_state, ebar)
        return new_state, (loss, aux)
    return step

def make_training_step_pmap_multi_state(loss_and_grad, mc_sampler, optimizer, accumulator=None, num_states=1):
    is_union = isinstance(mc_sampler, SamplerUnion)
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, opt_state, ebar = train_state
        mc_states = []
        data_list = []
        keys = jax.random.split(key, num_states)
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        for i in range(num_states):
            mc_state_i = sampler.refresh(mc_state[i], params[i])
            mc_state_i, data_i = sampler.sample(keys[i], params[i], mc_state_i)
            mc_states.append(mc_state_i)
            data_list.append(data_i)
        mc_states = tuple(mc_states)
        (loss, aux), grads = loss_and_grad(params, data_list, ebar)
        grads = tree_map(jnp.conj, grads)
        grads = jax.lax.pmean(grads, axis_name="device")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_state = TrainingState(ii+1, params, mc_states, opt_state, ebar)
        return new_state, (loss, aux)
    return step

def make_evaluation_step(expect_fn, mc_sampler):
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        new_state = TrainingState(ii+1, params, mc_state, *other)
        return new_state, (e_tot, aux)
    
    return step

def make_evaluation_step_multi_state(expect_fn, ovlp_fn, mc_sampler, num_states):
    is_union = isinstance(mc_sampler, SamplerUnion)
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        keys = jax.random.split(key, num_states)
        mc_states = []
        energies = []
        aux_list = []
        datas = []
        ovlp_list = []
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        for i in range(num_states):
            mc_state_i, data_i = sampler.sample(keys[i], params[i], mc_state[i])
            e_i, aux_i = expect_fn(params[i], data_i)
            datas.append(data_i)
            mc_states.append(mc_state_i)
            energies.append(e_i)
            aux_list.append(aux_i)
        for i in range(num_states):
            for j in range(i):
                ovlp_ji, _ = ovlp_fn(params[j], params[i], datas[j], datas[i])
                ovlp_list.append(ovlp_ji)
        mc_states = tuple(mc_states)
        new_state = TrainingState(ii+1, params, mc_states, *other)
        return new_state, (energies, ovlp_list, aux_list)
    return step
        
def make_evaluation_step_pmap(expect_fn, mc_sampler):
    """Multi-GPU evaluation step with pmap support"""
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        
        # Sync results across devices - average the energy and auxiliary values
        e_tot = jax.lax.pmean(e_tot, axis_name="device")
        aux = jax.lax.pmean(aux, axis_name="device")
        
        new_state = TrainingState(ii+1, params, mc_state, *other)
        return new_state, (e_tot, aux)
    
    return step
def make_evaluation_step_pmap_multi_state(expect_fn, ovlp_fn, mc_sampler, num_states):
    is_union = isinstance(mc_sampler, SamplerUnion)
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        keys = jax.random.split(key, num_states)
        mc_states = []
        energies = []
        aux_list = []
        datas = []
        ovlp_list = []
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        for i in range(num_states):
            mc_state_i, data_i = sampler.sample(keys[i], params[i], mc_state[i])
            e_i, aux_i = expect_fn(params[i], data_i)
            e_i = jax.lax.pmean(e_i, axis_name="device")
            aux_i = jax.lax.pmean(aux_i, axis_name="device")
            datas.append(data_i)
            mc_states.append(mc_state_i)
            energies.append(e_i)
            aux_list.append(aux_i)
        for i in range(num_states):
            for j in range(i):
                ovlp_ji, _ = ovlp_fn(params[j], params[i], datas[j], datas[i])
                ovlp_ji = jax.lax.pmean(ovlp_ji, axis_name="device")
                ovlp_list.append(ovlp_ji)
        mc_states = tuple(mc_states)
        new_state = TrainingState(ii+1, params, mc_states, *other)
        return new_state, (energies, ovlp_list, aux_list)
    return step
def train(cfg: ConfigDict):
    # handle logging
    logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger("train")
    log_level = getattr(logging, cfg.log.level.upper())
    logger.setLevel(log_level)

    def _env_int(name, default=1):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    # Initialize distributed only when environment explicitly requests multi-process.
    has_coord = any(os.environ.get(k) for k in ("JAX_COORDINATOR_ADDRESS", "COORDINATOR_ADDRESS"))
    env_nproc = max(_env_int("JAX_PROCESS_COUNT", 1),
                    _env_int("WORLD_SIZE", 1),
                    _env_int("SLURM_NTASKS", 1),
                    _env_int("PMI_SIZE", 1),
                    _env_int("OMPI_COMM_WORLD_SIZE", 1))
    requested_distributed = has_coord or env_nproc > 1
    if requested_distributed:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible = [x for x in cuda_visible.split(",") if x.strip() != ""]
        dist_kwargs = {}
        if visible:
            dist_kwargs["local_device_ids"] = list(range(len(visible)))
        try:
            jax.distributed.initialize(**dist_kwargs)
            logger.info("Initialized jax.distributed")
        except ValueError as err:
            emsg = str(err)
            if "coordinator_address should be defined" in emsg:
                raise ValueError(
                    "Distributed mode requested by environment, but coordinator "
                    "address is missing. Set JAX_COORDINATOR_ADDRESS "
                    "(and JAX_PROCESS_COUNT/JAX_PROCESS_INDEX when needed).") from err
            raise
        except RuntimeError as err:
            emsg = str(err)
            if "must be called before any JAX computations are executed" in emsg:
                raise RuntimeError(
                    "jax.distributed.initialize() was called too late. "
                    "Ensure no JAX call runs before hafqmc.train_all.train().") from err
            if "already initialized" in emsg.lower():
                logger.info("jax.distributed already initialized")
            else:
                raise

    n_devices = jax.device_count()
    n_local_devices = jax.local_device_count()
    process_index = jax.process_index()
    n_processes = jax.process_count()
    is_multi_gpu = n_devices > 1
    num_states = cfg.loss.num_states
    weights = cfg.loss.weights
    assert len(weights) == num_states * (num_states - 1) // 2, \
    f"weight list length must be num_states*(num_states-1)//2 = {num_states*(num_states-1)//2}, got {len(weights)}"

    #n_devices = jax.local_device_count()
    #is_multi_gpu = n_devices > 1
    
    if is_multi_gpu:
        log_once(logger, process_index,f"JAX detected {n_devices} total devices across {n_processes} processes")
        log_once(logger, process_index,f"Current process {process_index} has {n_local_devices} local devices")
        log_once(logger, process_index,f"Running on {n_devices} GPUs with pmap for data parallelism")
#        n_gpu_devices = jax.local_device_count("gpu")
#        logger.info(f"JAX detected {n_devices} local devices")
#        logger.info(f"Running on {n_gpu_devices} GPUs with pmap for data parallelism")
    else:
        logger.info("Running on single device")
    if process_index == 0:
        writer = SummaryWriter(cfg.log.stat_path)
    else:
        writer = None
    print_fields = {"step": "", "loss": ".4f"}
    for i in range(num_states):
        print_fields[f"e{i}"] = ".4f"
    for i in range(num_states):
        for j in range(i):
            print_fields[f"o{j}{i}"] = ".4f"
    #if cfg.loss.std_factor >= 0:
    #    print_fields.update({"std_es": ".4f", "std_s": ".4f"})
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
    
    #loss_fn = make_loss(expect_fn, **cfg.loss)
    loss_fn = make_loss(expect_fn, ovlp_fn, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    moving_avg_fn = (make_moving_avg(**cfg.optim.baseline)
        if cfg.optim.baseline is not None else None)

    # the core training iteration
    if cfg.optim.lr.start > 0:
        if is_multi_gpu:
            train_step = make_training_step_pmap_multi_state(loss_and_grad, mc_sampler, optimizer, moving_avg_fn, num_states=num_states)
        else:
            train_step = make_training_step_multi_state(loss_and_grad, mc_sampler, optimizer, moving_avg_fn, num_states=num_states)
    else:
        log_once(logger, process_index,"Running inference mode")
        if is_multi_gpu:
            train_step = make_evaluation_step_pmap_multi_state(expect_fn, ovlp_fn, mc_sampler, num_states=num_states)
        else:
            train_step = make_evaluation_step_multi_state(expect_fn, ovlp_fn, mc_sampler, num_states=num_states)
    
    if is_multi_gpu:
        train_step = jax.pmap(train_step, axis_name="device", static_broadcasted_argnums=(2,))
    else:
        train_step = jax.jit(train_step, static_argnums=(2,))
    
    # set up all states
    if cfg.restart.states is None:
        log_once(logger, process_index, "Initializing parameters and states")
        key = jax.random.PRNGKey(cfg.seed)
        key, pakey, mckey = jax.random.split(key, 3)
        fshape = braket.fields_shape()
        
        if cfg.restart.params is None:
            pakeys = jax.random.split(pakey, num_states)
            init_fn = jax.jit(braket.init)
            params = tuple(init_fn(k, tree_map(jnp.zeros, fshape)) for k in pakeys)
            #params = jax.jit(braket.init)(pakey, tree_map(jnp.zeros, fshape))
        else:
            # TODO: change the way params are loaded in inference running.
            log_once(logger, process_index, "Loading parameters from saved file")
            params = load_pickle(cfg.restart.params)
            if isinstance(params, tuple): params = params[1]
            if isinstance(params, tuple): params = params[1]
        #mc_state = mc_sampler.init(mckey, params)
        mckeys = jax.random.split(mckey, num_states)
        mc_states = tuple(mc_sampler.init(mckeys[i], params[i]) for i in range(num_states))
        opt_state = optimizer.init(params)
        
        if cfg.sample.burn_in > 0:
            log_once(logger, process_index, f"Burning in the {num_states} samplers for {cfg.sample.burn_in} steps")
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, num_states)
            mc_state_list = []
            for i in range(num_states):
                mc_state_i = sampler_1s_nc.burn_in(keys[i], params[i], mc_states[i], cfg.sample.burn_in)
                mc_state_list.append(mc_state_i)
            mc_states = tuple(mc_state_list)
            #key, subkey = jax.random.split(key)
            #mc_state = sampler_1s_nc.burn_in(subkey, params, mc_state, cfg.sample.burn_in)
        
        ebar = hamiltonian.local_energy() if cfg.optim.baseline is not None else None
        train_state = TrainingState(0, params, mc_states, opt_state, ebar)
        
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
            aux_list = aux["aux_list"]
            aux_list = [
                jax.tree_map(lambda x: float(x[0]) if jnp.isscalar(x[0]) else x[0], aux_i)
                for aux_i in aux_list
            ]
            ovlp_list = [float(o[0]) for o in aux["ovlp"]]
            energy_list = [float(e[0]) for e in aux["energy"]]
            aux = {"aux_list": aux_list,
                   "ovlp": ovlp_list,
                   "energy": energy_list}
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
                flat = {"step": ii, "loss": loss, "lr": _lr}
                for i, e in enumerate(aux["energy"]):
                    flat[f"e{i}"] = e
                k = 0
                for i in range(num_states):
                    for j in range(i):
                        flat[f"o{j}{i}"] = aux["ovlp"][k]
                        k += 1
                printer.print_fields(flat)
            if writer is not None:
                flat = {"loss": loss, "lr": _lr}
                for i, e in enumerate(aux["energy"]):
                    flat[f"e{i}"] = e
                k = 0
                for i in range(num_states):
                    for j in range(i):
                        flat[f"o{j}{i}"] = aux["ovlp"][k]
                        k += 1
                writer.add_scalars("stat", flat, global_step=ii)
            
        if ii % cfg.log.ckpt_freq == 0:
            if process_index == 0:
                checkpoint_filename = f"./{cfg.log.ckpt_path}/ckpt_{ii}.pkl"
                if is_multi_gpu:
                    unreplicated_state = jax.tree_map(lambda x: x[0], train_state)
                    save_pickle(checkpoint_filename, (keys[0], tuple(unreplicated_state)))
                else:
                    save_pickle(checkpoint_filename, (key, tuple(train_state)))
    if process_index == 0: 
        writer.close()
    return train_state
