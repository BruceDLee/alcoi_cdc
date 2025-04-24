import jax.numpy as jnp
import jax
from functools import partial
from jax import jit, jacfwd, grad
from jax.scipy import optimize
import numpy as onp

@partial(jit, static_argnames=['dyn', 'controller'])
def simulate_sys(noise, dyn, controller, phi, x0):
  dx = len(x0)
  def step(x, noise):
    w = noise[:dx]; v = noise[dx:]
    u = controller(x, v)
    xprev = x
    x = dyn(x, u, w, phi)
    return x, (xprev, u)

  init = x0
  x, (xs, us) = jax.lax.scan(step, init, noise)
  return jnp.vstack([xs, x]), us

@partial(jit, static_argnames=['du', 'dyn', 'controller', 'T'])
def collect_traj(key, phi, x0s, du, dyn, controller, T, budget):
  dx = x0s.shape[1]
  noise = jax.random.normal(key, (len(x0s), T, dx + du))
  noise = noise.at[:,:,dx:].set(jnp.sqrt(budget*T)*noise[:,:,dx:]/jnp.sqrt(jnp.sum(noise[:,:,dx:]**2, keepdims=True, axis=(1,2))))

  return jax.vmap(simulate_sys, in_axes=(0, None, None, None, 0), out_axes=(0,0))(noise, dyn, controller, phi, x0s)

def collect_trajectories(key, phi, x0s, du, dyn, controller, split, T, budget):
  all_xs = []; all_us = []
  dx = x0s.shape[1]
  for i in range(int(onp.ceil(len(x0s)/split))):
    key, subkey = jax.random.split(key)
    x0s_padded = jnp.concatenate([x0s[i*split:(i+1)*split], jnp.zeros((split- len(x0s[i*split:(i+1)*split]), dx))])
    xs, us = collect_traj(subkey, phi, x0s_padded, du, dyn, controller, T, budget)
    all_xs.append(xs)
    all_us.append(us)

  return jnp.concatenate(all_xs)[:len(x0s)], jnp.concatenate(all_us)[:len(x0s)]

def est_phi(key, data, noiseless_dyn, phi_star, n_inits):
  noiseless_dyn_vec = jax.vmap(jax.vmap(noiseless_dyn, in_axes=(0,0,None)), in_axes=(0,0,None))
  def l(phi, data):
    xs, us = data
    error = jnp.mean((xs[:,1:] - noiseless_dyn_vec(xs[:,:-1], us,phi))**2)
    return error

  loss = partial(l, data=data)

  solution = optimize.minimize(loss, phi_star, method='BFGS').x
  minval = loss(solution)

  for i in range(n_inits):
    key, subkey = jax.random.split(key)
    init = jax.random.normal(subkey, (len(phi_star),))
    candidate = optimize.minimize(loss, init, method='BFGS').x
    val = loss(candidate)
    minval = jnp.minimum(val, minval)
    solution = jax.lax.cond(val < minval, lambda x: candidate, lambda x: solution, None)
  return solution

est_phi = jit(partial(est_phi, n_inits=5), static_argnames=['noiseless_dyn'])

@partial(jit, static_argnames=['noiseless_dyn'])
def exp_cost(rollout, Hhat, Lamb, phi_hat, noiseless_dyn):
  Df = jacfwd(noiseless_dyn, 2)
  for x,u in zip(rollout[0], rollout[1]):
    Lamb += Df(x,u, phi_hat).T@Df(x,u, phi_hat)
  return jnp.trace(Hhat@jnp.linalg.inv(Lamb))

@partial(jit, static_argnames=['noiseless_dyn'])
def evaluate_input_candidates(outer_carry, u, noiseless_dyn):
  mincost, best_u, phi, Hhat, Lamb, x = outer_carry

  @jit
  def zero_dyn(carry, u):
    x, phi = carry
    return (noiseless_dyn(x,u,phi), phi), (x,u) 

  _, rollout = jax.lax.scan(zero_dyn, (x,phi), u)
  rollout_cost = exp_cost(rollout, Hhat, Lamb, phi, noiseless_dyn)

  best_u = jax.lax.cond(rollout_cost < mincost, lambda _: u, lambda _: best_u, None)
  mincost = jnp.minimum(rollout_cost, mincost)

  return (mincost, best_u, phi, Hhat, Lamb, x), None

@partial(jit, static_argnames=['noiseless_dyn'])
def empirical_covariance(data, phi_hat, noiseless_dyn):
  xs, us = data
  Df = jacfwd(noiseless_dyn, 2)
  emp = jnp.zeros((len(phi_hat), len(phi_hat)))
  for x, u in zip(xs,us):
    for i in range(len(u)):
      emp += Df(x[i], u[i], phi_hat).T@Df(x[i], u[i], phi_hat)
  return emp

def model_predictive_exploration(key, x, us_past, t, Hhat, Lamb, exp_cost, phi, budget,  noiseless_dyn, N_sample, T):
  remaining_budget = T*budget - jnp.sum(us_past[:t]**2)
  du = us_past.shape[1]
  key, subkey = jax.random.split(key)
  us = jax.random.normal(subkey, (N_sample, T-t, du))
  us = 1/jnp.sqrt(jnp.expand_dims(jnp.sum(us**2, axis = (1,2)),axis=(1,2)))*us*jnp.sqrt(remaining_budget)

  best_u = us_past[t:]

  @jit
  def zero_dyn(carry, u):
    x, phi = carry
    return (noiseless_dyn(x,u,phi), phi), (x,u)

  _, rollout = jax.lax.scan(zero_dyn, (x,phi), best_u)
  mincost = exp_cost(rollout, Hhat, Lamb, phi, noiseless_dyn)
  carry, _ = jax.lax.scan(partial(evaluate_input_candidates, noiseless_dyn=noiseless_dyn), (mincost, best_u, phi, Hhat, Lamb, x), us[1:])
  best_u = carry[1]

  return best_u

policy = jit(model_predictive_exploration, static_argnames=['exp_cost', 't', 'noiseless_dyn', 'T', 'N_sample'])

@partial(jit, static_argnames=['t', 'dyn', 'policy', 'noiseless_dyn', 'T', 'N_sample'])
def step(carry, input, policy, t, dyn, phi_star, noiseless_dyn, T, N_sample): 
  key, x, us, Hhat, Lamb, phi_hat, budget  = carry
  key, subkey = jax.random.split(key)
  us_mpc = policy(subkey, x, us, t, Hhat, Lamb, exp_cost, phi_hat, budget, noiseless_dyn, N_sample, T)
  us = us.at[t:].set(us_mpc)
  xprev = x
  
  Df = jacfwd(noiseless_dyn, 2)
  Lamb = Lamb + Df(x, us[t], phi_hat).T@Df(x, us[t], phi_hat)

  x = dyn(x,us[t],input,phi_star)
  return (key, x, us, Hhat, Lamb, phi_hat, budget), (xprev, jnp.array(us[t]))

def targeted_exploration(key, Hhat, Lamb, K, phi_hat, T, budget, noiseless_dyn, dyn, phi_star, du, dx, N_sample, x0):

  all_data = []
  for _ in range(K):
    #collect a dataset by playing model predictive control with the objective trace(Hhat(Lambda + stuff)^{-1})
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, dx))
    key, subkey = jax.random.split(key)
    carry = (subkey, x0, jnp.zeros((T, du)), Hhat, Lamb, phi_hat, budget)
    dat = []
    for t, noise_val in enumerate(noise):
      carry, data = step(carry, noise_val, policy, t, dyn, phi_star, noiseless_dyn, T, N_sample)
      dat.append(data)

    data = (jnp.stack([data[0] for data in dat]), jnp.stack([data[1] for data in dat]))
    data = (jnp.expand_dims(jnp.vstack([data[0], carry[1]]), 0), jnp.expand_dims(data[1], 0))
    Lamb = Lamb + empirical_covariance(data, phi_hat, noiseless_dyn)
    all_data.append(data)

  all_data = (jnp.concatenate([data[0] for data in all_data]),jnp.concatenate([data[1] for data in all_data]))
  
  return all_data
 
def alcoi(key, N, T, pi0, phi_star, budget, dx, du, dyn, get_H, noiseless_dyn, Aopt = False, debug=False, x0 = None, Hstar=None):
  N_2 = round(N/2)
  key, subkey = jax.random.split(key)

  #collect initial dataset
  x0 = x0 if x0 is not None else jnp.zeros(dx)
  x0s = jnp.tile(x0, (N_2, 1))
  data1 = collect_trajectories(subkey, phi_star, x0s,  du, dyn, pi0, split = 50, T=T, budget=budget)
  data1 = (jnp.stack(data1[0]), jnp.stack(data1[1]))

  #obtain coarse dynamics estimate
  key, subkey = jax.random.split(key)
  phi_hat_minus = est_phi(subkey, data1, noiseless_dyn, phi_star)

  Lamb = empirical_covariance(data1, phi_hat_minus, noiseless_dyn)

  #find model-task hessian
  key, subkey = jax.random.split(key)
  Hhat = jnp.eye(phi_star.shape[0])
  
  if not Aopt:
    Hhat = get_H(subkey, phi_hat_minus)
    Hhat = Hhat*jnp.sign((jnp.max(jnp.real(jnp.linalg.eigvalsh(Hhat))) - jnp.abs(jnp.min(jnp.real(jnp.linalg.eigvalsh(Hhat))))))
    Hhat = Hhat - (jnp.min(jnp.real(jnp.linalg.eigvalsh(Hhat)))-0.05)*jnp.eye(Hhat.shape[0])
  
  #run targeted exploration
  key, subkey = jax.random.split(key)
  data2 = targeted_exploration(subkey, Hhat, Lamb, N_2, phi_hat_minus, T, budget, noiseless_dyn, dyn, phi_star, du, dx, 200, x0=x0)

  #concatenate all data
  all_data = (jnp.concatenate([data1[0], data2[0]]), jnp.concatenate([data1[1], data2[1]]))
  key, subkey = jax.random.split(key)
  phi_hat_plus = est_phi(subkey, all_data, noiseless_dyn, phi_star)
  return phi_hat_plus, all_data, Hhat