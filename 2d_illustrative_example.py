import scipy.linalg as la
from jax.scipy import optimize
import matplotlib.pyplot as plt
import numpy as onp

import jax.numpy as jnp
import jax
from jax import jacfwd, grad, hessian
from jax import jit
from functools import partial
from tqdm import tqdm

from utilsv2 import collect_trajectories, est_phi, alcoi

gauss = lambda x, c: jnp.exp(-jnp.sum((x-c)**2))*(x-c)/jnp.sqrt(jnp.sum((x-c)**2))

dx = 2
du = 2

phi_star = jnp.array([[-5.0,0.0], [5.0, 0.0], [0.0, -5.0], [0.0, 5.0]]).reshape(8)
desired_location=jnp.array([-5.5, 0])
dphi = 2

def noiseless_dyn(x,u,phi):
    phi = phi.reshape(4,2)
    return x + u + 5*(gauss(x, phi[0]) + gauss(x, phi[1]) + gauss(x, phi[2]) + gauss(x, phi[3]))

def dyn(x,u,w,phi):
  return noiseless_dyn(x,u,phi) +  w

def pi0(x, v):
  return v

T = 10
N = 16
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

data = collect_trajectories(subkey, phi_star, jnp.zeros((N, dx)),  du, dyn, pi0, split = 50, T=T, budget=1.0)

key, subkey = jax.random.split(key)
phi_hat = est_phi(subkey, data, noiseless_dyn, phi_star)


def traj_cost(xs,us):
  return jnp.sum((xs-desired_location)**2) + 1/100*jnp.sum(us**2)

Q = jnp.eye(dx)
R = 1/100*jnp.eye(du)
P = la.solve_discrete_are(jnp.eye(dx), jnp.eye(du),Q,R)
K = -la.solve(P+R, P)

def cost(key, phi_synth, phi_eval,T=T, N = 100):
  def CE_controller(phi):
    phi = phi.reshape(4,2)
    def controller(x,v):
      return K@(x-desired_location) - (5*gauss(x, phi[0]) + 5*gauss(x, phi[1]) + 5*gauss(x, phi[2]) + 5*gauss(x, phi[3]))
    return controller
  xs, us = collect_trajectories(key, phi_eval, jnp.zeros((N, dx)), du, dyn, CE_controller(phi_synth), split=10, T = T, budget=0.0)
  return traj_cost(jnp.stack(xs), jnp.stack(us))/N

Neval = 100
eval_cost = jit(partial(cost, T=T, N=Neval))
hess_cost = hessian(cost,1)
def get_H(key, phi):
  return hess_cost(key, phi,phi)

chosen_budget = 1.0

phi_hat_plus, data, _ = alcoi(subkey, 16, 10, pi0, phi_star, chosen_budget, dx, du, dyn, get_H, noiseless_dyn, Aopt = False, debug=False)
print('phi hat plus from alcoi: ', phi_hat_plus)


def eval_excess_cost(eval_subkey, phi_hat, phi_star):
  return eval_cost(eval_subkey, phi_hat, phi_star) - eval_cost(eval_subkey, phi_star, phi_star)
vec_eval_excess_cost = jit(jax.vmap(eval_excess_cost, in_axes=(0, None, None)))

all_ecs = {'random_exploration': [], 'aopt': [], 'alcoi': []}

key, subkey = jax.random.split(key)
eval_subkeys = jax.random.split(subkey, 100)

def estimate_phi(method, key, pi0, phi_star, N, T, budget=chosen_budget):
  if method == 'random_exploration':
    data = collect_trajectories(key, phi_star,jnp.zeros((N, dx)), du, dyn, pi0, split=50, T=T, budget=budget)
    phi_hat = est_phi(subkey, data, noiseless_dyn, phi_star)
  elif method == 'aopt':
    phi_hat, _, _ = alcoi(key, N, T, pi0, phi_star, budget, dx, du, dyn, get_H, noiseless_dyn, Aopt=True, debug=False)
  elif method == 'alcoi':
    phi_hat, _, _ = alcoi(key, N, T, pi0, phi_star, budget, dx, du, dyn, get_H, noiseless_dyn, Aopt=False, debug=False)
  else:
    raise ValueError('method not recognized')
  return phi_hat

def eval_method(method, fit_key, eval_subkeys, pi0, phi_star, N, T, budget):
    phi_hat = estimate_phi(method, fit_key, pi0, phi_star, N, T, budget=budget)
    return jnp.mean(vec_eval_excess_cost(eval_subkeys, phi_hat, phi_star)), phi_hat
vectorized_eval_method = jax.vmap(eval_method, in_axes=(None, 0, None, None, None, None, None, None), out_axes=(0,0))
key, subkey = jax.random.split(key)
fit_subkeys = jax.random.split(subkey, 5)

T = 10

for method in all_ecs.keys():
  for N in [16, 54, 128, 250]:
    print('Running experiment with Number of Episodes = ', N)
    ecs, phi_hats = vectorized_eval_method(method, fit_subkeys, eval_subkeys, pi0, phi_star, N, T,chosen_budget)
    all_ecs[method].append(jnp.nan_to_num(ecs, nan=1000))


data_amt = [16, 54, 128, 250]

# Using median as the central tendency measure
all_ecs['random_exploration'] = jnp.stack(all_ecs['random_exploration'])
all_ecs['alcoi'] = jnp.stack(all_ecs['alcoi'])
all_ecs['aopt'] = jnp.stack(all_ecs['aopt'])

import pickle

mean_random_exploration = jnp.mean(all_ecs['random_exploration'], axis=1)
mean_alcoi = jnp.mean(all_ecs['alcoi'], axis=1)
mean_aopt = jnp.mean(all_ecs['aopt'], axis=1)

std_error_random_exploration = jnp.std(all_ecs['random_exploration'], axis=1)/jnp.sqrt(len(fit_subkeys))
std_error_alcoi = jnp.std(all_ecs['alcoi'], axis=1)/jnp.sqrt(len(fit_subkeys))
std_error_aopt = jnp.std(all_ecs['aopt'], axis=1)/jnp.sqrt(len(fit_subkeys))

# Plotting medians with increased line width
plt.figure(figsize=(10, 6))
plt.plot(data_amt, mean_random_exploration, label='Random Exploration', linewidth=2, marker='o', markersize=8)  # Circle marker
plt.plot(data_amt, mean_alcoi, label='ALCOI', linewidth=2, marker='s', markersize=8)  # Square marker
plt.plot(data_amt, mean_aopt, label='AOPT', linewidth=2, marker='^', markersize=8)

# Shading 25-75 quantiles
plt.fill_between(data_amt, mean_random_exploration - std_error_random_exploration, mean_random_exploration + std_error_random_exploration, alpha=0.1)
plt.fill_between(data_amt, mean_alcoi - std_error_alcoi, mean_alcoi + std_error_alcoi, alpha=0.1)
plt.fill_between(data_amt,  mean_aopt - std_error_aopt, mean_aopt + std_error_aopt, alpha=0.1)

# Adding axis labels and a grid for better readability
plt.xlabel('Number of Episodes', fontsize=14)
plt.ylabel('Excess Control Cost', fontsize=14)
plt.legend(['Random', 'ALCOI', 'A-Optimal'], fontsize=12)

# Setting tick parameters
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.savefig('illustrative_example_active_learning_nonlinear_exp_design.pdf',bbox_inches='tight')

