import jax
import jax.numpy as jnp
from jax.interpreters import xla
from jax.lib import xla_client

def get_optimized_hlo(fn, input_val):
    # 1. JIT the function which is needed to get optimized HLO
    jitted_fn = jax.jit(fn)
    
    comp = jitted_fn.lower(input_val).compile()
    
    if hasattr(comp, 'hlo_modules'):
        hlo_module = comp.hlo_modules()[0]
    else:
        hlo_module = comp._executable.xla_executable.hlo_modules()[0]
    
    return hlo_module.to_string()

def model(x):
    return jnp.sin(x) + jnp.cos(x)

# JITs model and gradient function
jitted_model = jax.jit(model)
grad_fn = jax.jit(jax.grad(model))

# Gets optimized HLO for forward pass and gradient pass
forward_hlo = get_optimized_hlo(jitted_model, 1.0)
grad_hlo = get_optimized_hlo(grad_fn, 1.0)

# Save both forard and gradient HLO IR
with open("forward_optimized.hlo", "w") as f:
    f.write(forward_hlo)

with open("grad_optimized.hlo", "w") as f:
    f.write(grad_hlo)
