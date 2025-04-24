import scipy.linalg as la
from jax.scipy import optimize
from jax.scipy import linalg as jla
import matplotlib.pyplot as plt
import numpy as onp

import jax.numpy as jnp
import jax
# from jax import config
# config.update("jax_enable_x64", True)
from jax import jacfwd, grad, hessian
from jax import jit

from functools import partial
from tqdm import tqdm

import matplotlib.pyplot as plt

import pickle as pkl
import os

from utilsv2 import collect_trajectories, est_phi, alcoi
from utilsv2 import collect_traj

#Define the filename for the results
i = 0
filename = 'cartpole_results_'+str(i)+'.pkl'
debug = False
if not debug:
  while os.path.exists(os.path.join('cartpole_data', filename)):
    i += 1
    filename = 'cartpole_results_'+str(i)+'.pkl'

print('Dataset number: ', i)
with open(os.path.join('cartpole_data', filename), 'wb') as f:
  pkl.dump({'filename':filename}, f)

#Define cartpole dynamics

dx = 4
du = 1
dt = 0.2
phi_star = jnp.array([1.0, 0.1, 1.0, 1.0, 1.0, 9.81]) #cart mass, pole mass, pole length, friction

@jit
def noiseless_dyn(in_state,u,phi):
    """
    Compute the continuous-time dynamics of the cart-pole system.

    Parameters:
    state (array): The state vector [x, x_dot, theta, theta_dot].
    control (array): The control vector [F], where F is the force applied to the cart.

    Returns:
    array: The derivative of the state vector [x_dot, x_ddot, theta_dot, theta_ddot].
    """
    # Unpack the state vector
  
    x, x_dot, theta, theta_dot = in_state

    # Unpack the control vector
    F = u[0]
    m_c, m_p, l, b_x, b_theta, g = phi

    # Intermediate calculations
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    a1 = jnp.array([[m_c + m_p, m_p*l*cos_theta], [m_p*cos_theta, m_p*l]])
    b1 = jnp.array([[m_p*l*theta_dot**2*sin_theta+F], [m_p*g*sin_theta]])
    c1 = jnp.linalg.inv(a1)@b1
    x_ddot = c1[0,0]
    theta_ddot = c1[1,0]

    new_state = in_state + dt*jnp.array([x_dot, x_ddot - b_x*x_dot, theta_dot, theta_ddot - b_theta*theta_dot])
    new_state = new_state.at[2].set(jnp.mod(new_state[2] + jnp.pi, 2*jnp.pi) - jnp.pi)
    return new_state

@jit
def dyn(x,u,w,phi):
    return noiseless_dyn(x,u,phi) +  0.01*w

@jit
def pi0(x, v):
  return v

T = 30
N = 1
key = jax.random.PRNGKey(i)
key, subkey = jax.random.split(key)

#simulate the system, and verify that the dynamics appear reasonable.
x0s =  jnp.zeros((N, dx))
x0s = x0s.at[:,2].set(jnp.pi)
data = collect_traj(subkey, phi_star, x0s,  du, dyn, pi0, T=T, budget=0.1)

#plot the trajectories
plt.figure()
for i in range(dx):
  plt.plot(data[0][0,:,i])

plt.savefig('uncontrolled_cartpole.png')

@jit
def traj_cost(xs,us):
  return jnp.mean(jnp.sum(xs[:,20:,:2]**2 , axis=(1,2)) + jnp.sum(xs[:,20:,3]**2 , axis=1) + 5*jnp.sum(jnp.sin(xs[:,20:,2])**2 , axis=1) + jnp.sum(us[:,20:]**2,axis=(1,2)))

@jit
def lqr(A,B,Q,R, iterations=100):
  lqr_iter = lambda i, P: Q + A.T@P@A - A.T@P@B@jla.inv(R + B.T@P@B)@B.T@P@A
  P = jax.lax.fori_loop(0, iterations, lqr_iter, Q)
  return -jla.inv(R + B.T@P@B)@B.T@P@A

state_jac = jacfwd(noiseless_dyn, 0)
input_jac = jacfwd(noiseless_dyn, 1)

def CE_controller(phi):
  A = state_jac(jnp.zeros(4), jnp.zeros(1), phi)
  B = input_jac(jnp.zeros(4), jnp.zeros(1), phi)
  Q = jnp.eye(4)
  R = jnp.eye(1)
  K = lqr(A,B,Q,R)

  m_c, m_p, l, b_x, b_theta, g = phi
  des_in_gain = jnp.hstack([1/2*m_p*l**2, m_p*g*l]).reshape(1,2)
  swing_up_gain = jnp.hstack([m_c+m_p, -m_p, m_p*g, -m_p*l])
  def controller(x,v):
    lqr_control = K@x
    pos, pos_dot, theta, theta_dot = x
    desired_input= 5*theta_dot*jnp.cos(theta)*des_in_gain@jnp.hstack([theta_dot**2, (jnp.cos(theta) - 1)]) - 0.01*pos - 0.01*pos_dot
    swing_up_control = (swing_up_gain[:2]@jnp.hstack([1, jnp.cos(theta)]))*desired_input + swing_up_gain[2:]@jnp.hstack([jnp.cos(theta)*jnp.sin(theta), theta_dot**2*jnp.sin(theta)])

    return jax.lax.cond(jnp.abs(x[2]) < jnp.pi/4, lambda _: lqr_control, lambda _: swing_up_control, None)
  return controller

x0 = jnp.array([0.0, 0.0, jnp.pi, 0.0])
x0s = jnp.tile(x0, (N,1))
xs, us = collect_traj(key, phi_star, x0s, du, dyn, CE_controller(phi_star),  T = T, budget=0.0)
plt.figure()

for i in range(dx):
  plt.plot(xs[0,:,i])

plt.savefig('controlled_cartpole.png')

def cost(key, phi_synth, phi_eval,T=T, N = 100):
  def CE_controller(phi):
    A = state_jac(jnp.zeros(4), jnp.zeros(1), phi)
    B = input_jac(jnp.zeros(4), jnp.zeros(1), phi)
    Q = jnp.eye(4)
    R = jnp.eye(1)
    K = lqr(A,B,Q,R)

    m_c, m_p, l, b_x, b_theta, g = phi
    des_in_gain = jnp.hstack([1/2*m_p*l**2, m_p*g*l]).reshape(1,2)
    swing_up_gain = jnp.hstack([m_c+m_p, -m_p, m_p*g, -m_p*l])
    def controller(x,v):
      lqr_control = K@x
      pos, pos_dot, theta, theta_dot = x
      desired_input= 5*theta_dot*jnp.cos(theta)*des_in_gain@jnp.hstack([theta_dot**2, (jnp.cos(theta) - 1)]) - 0.01*pos - 0.01*pos_dot
      swing_up_control = (swing_up_gain[:2]@jnp.hstack([1, jnp.cos(theta)]))*desired_input + swing_up_gain[2:]@jnp.hstack([jnp.cos(theta)*jnp.sin(theta), theta_dot**2*jnp.sin(theta)])

      return jax.lax.cond(jnp.abs(x[2]) < jnp.pi/4, lambda _: lqr_control, lambda _: swing_up_control, None)
    return controller
  x0 = jnp.array([0.0, 0.0, jnp.pi, 0.0])
  x0s = jnp.tile(x0, (N,1))
  xs, us = collect_traj(key, phi_eval, x0s, du, dyn, CE_controller(phi_synth), T = T, budget=0.0)
  return traj_cost(jnp.stack(xs), jnp.stack(us))

cost(subkey, phi_star,phi_star)

Neval = 100
eval_cost = jit(partial(cost, T=T, N=Neval)) 
hess_cost = hessian(cost,1)
def get_H(key, phi):
  return hess_cost(key, phi,phi)

for _ in range(1):
  key, subkey = jax.random.split(key)
  Hstar = get_H(subkey, phi_star)
  if jnp.max(jnp.real(jnp.linalg.eigvalsh(Hstar))) > jnp.abs(jnp.min(jnp.real(jnp.linalg.eigvalsh(Hstar)))):
    break

chosen_budget = 0.1

@jit
def eval_excess_cost(eval_subkey, phi_hat, phi_star):
  return eval_cost(eval_subkey, phi_hat, phi_star) - eval_cost(eval_subkey, phi_star, phi_star)
vec_eval_excess_cost = jit(jax.vmap(eval_excess_cost, in_axes=(0, None, None)))

all_ecs = {'random_exploration': [], 'aopt': [], 'alcoi': []}

key, subkey = jax.random.split(key)
eval_subkeys = jax.random.split(subkey, 100)

def estimate_phi(method, key, pi0, phi_star, N, T, budget=chosen_budget):
  x0 = jnp.array([0,0,jnp.pi,0])
  if method == 'random_exploration':
    data = collect_traj(key, phi_star,jnp.tile(x0, (N,1)), du, dyn, pi0, T=T, budget=budget)
    phi_hat = est_phi(subkey, data, noiseless_dyn, phi_star)
    Hhat  =None
  elif method == 'aopt':
    phi_hat, _, Hhat = alcoi(key, N, T, pi0, phi_star, budget, dx, du, dyn, get_H, noiseless_dyn, Aopt=True, debug=False, x0=x0)
  elif method == 'alcoi':
    phi_hat, _, Hhat = alcoi(key, N, T, pi0, phi_star, budget, dx, du, dyn, get_H, noiseless_dyn, Aopt=False, debug=False, x0=x0)
  else:
    raise ValueError('method not recognized')
  return phi_hat, Hhat 

def eval_method(method, fit_key, eval_subkeys, pi0, phi_star, N, T, budget):
    phi_hat, Hhat  = estimate_phi(method, fit_key, pi0, phi_star, N, T, budget=budget)
    return jnp.mean(vec_eval_excess_cost(eval_subkeys, phi_hat, phi_star)), jnp.sum((phi_hat-phi_star)**2), Hhat

vectorized_eval_method = jax.vmap(eval_method, in_axes=(None, 0, None, None, None, None, None, None), out_axes=(0,0,0))
key, subkey = jax.random.split(key)
fit_subkeys = jax.random.split(subkey, 100)

T = 30
for method in all_ecs.keys():
  for N in [16, 36, 54, 72]:
    print('N: ', N, 'method: ', method)
    ecs, est_error, Hhats = vectorized_eval_method(method, fit_subkeys, eval_subkeys, pi0, phi_star, N, T,chosen_budget)
    all_ecs[method].append(jnp.nan_to_num(ecs, nan=100000))
    print('mean excess cost: ', jnp.mean(ecs))
    print('std excess cost: ', jnp.std(ecs)/jnp.sqrt(len(fit_subkeys)))

with open(os.path.join('cartpole_data', filename), 'wb') as f:
  pkl.dump(all_ecs, f)
