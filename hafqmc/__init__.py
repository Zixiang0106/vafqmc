import scipy.signal as _signal
del _signal # just to avoid import failure of jax

import jax
jax.config.update("jax_enable_x64", True)