import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pickle 
import os

i = 0
j = 0
filename = 'cartpole_results_'+str(i)+'.pkl'

all_data = []
while os.path.exists(os.path.join('cartpole_data', filename)):
    with open(os.path.join('cartpole_data', filename), 'rb') as f:
        all_data.append(pickle.load(f))
    i += 1
    filename = 'cartpole_results_'+str(i)+'.pkl'

all_ecs = {'random_exploration': [], 'alcoi': [], 'aopt': []}
data_amt = [16, 36, 54, 72]
for method in all_ecs.keys():
    for i in range(len(data_amt)):
        temp = []
        for data in all_data:
            temp.extend(data[method][i])
        all_ecs[method].append(jnp.hstack(temp))

# Using median as the central tendency measure

all_ecs['random_exploration'] = jnp.clip(jnp.stack(all_ecs['random_exploration']), -30, 30)
all_ecs['alcoi'] = jnp.stack(all_ecs['alcoi'])
all_ecs['aopt'] = jnp.stack(all_ecs['aopt'])

mean_random_exploration = jnp.mean(all_ecs['random_exploration'], axis=1)
mean_alcoi = jnp.mean(all_ecs['alcoi'], axis=1)
mean_aopt = jnp.mean(all_ecs['aopt'], axis=1)


print('len(all_ecs[\'random_exploration\'])', len(all_ecs['random_exploration']))
std_error_random_exploration = jnp.std(all_ecs['random_exploration'], axis=1)/jnp.sqrt(len(all_ecs['random_exploration'][0]))
std_error_alcoi = jnp.std(all_ecs['alcoi'], axis=1)/jnp.sqrt(len(all_ecs['alcoi'][0]))
std_error_aopt = jnp.std(all_ecs['aopt'], axis=1)/jnp.sqrt(len(all_ecs['aopt'][0]))

plt.figure(figsize=(10, 6))
# Plotting medians with increased line width
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
plt.savefig('cartpole_active_learning_nonlinear_exp_design.pdf',bbox_inches='tight')
