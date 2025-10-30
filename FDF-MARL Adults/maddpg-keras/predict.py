import csv 
import time
import warnings
import numpy as np
import tensorflow as tf

from sklearn.exceptions import UndefinedMetricWarning
from tensorflow.keras.models import load_model
from pathlib import Path

from FDF_env import ENVIRONMENT
from env.env_predict import *
from maddpg.buffer import *
from maddpg.model import *
from maddpg.noise import *

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

#Dimension of State Space for single agent
dim_agent_state = 2

num_agents = 10

#Dimension of State Space
dim_state = dim_agent_state*num_agents

#Number of Episodes
num_episodes = 50

#Number of Steps
num_steps = 200

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

ac_models = []
cr_models = []
target_ac = []
target_cr = []

path = './saved_models_adults_agg_400/'

for i in range(num_agents):
  ac_models.append(load_model(path + 'actor'+str(i)+'.h5')) 
  cr_models.append(load_model(path + 'critic'+str(i)+'.h5'))

  target_ac.append(load_model(path + 'target_actor'+str(i)+'.h5'))
  target_cr.append(load_model(path + 'target_critic'+str(i)+'.h5'))



def policy(state, noise_object, model):
    
    sampled_actions = tf.squeeze(model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + 0

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, 0, 1.0)

    return [np.squeeze(legal_action)]


agentsFile = f"AR-adults-keras_testing_DI_agg.csv"
episodesFile = f"TR-adults-keras_testing_DI_agg.csv"

ep_reward_list = []
avg_reward_list = []

with open(agentsFile, mode='w', newline='') as file1, open(episodesFile, mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)
    
    for ep in range(num_episodes):
        writer1.writerow([f"episode{ep}"])
        writer2.writerow([f"episode{ep}"])
        env = ENVIRONMENT()
        env.reset()
        prev_state = env.initial_obs()
    
        episodic_reward = 0
        
        print("Testing has started")
        start_time = time.time()

        for i in range(num_steps):
            
            # Expanding dimension of state from 1-d array to 2-d array
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            
            # Action Value for each agents will be stored in this list
            actions = []
            actions_mapped= {}

            agent_map = {0: 'Server', **{i: f'Client {i}' for i in range(1, 10)}}

            for j, model in enumerate(ac_models):
              action = policy(tf_prev_state[:,dim_agent_state*j:dim_agent_state*(j+1)], ou_noise, model)
              actions.append(float(round(action[0].item())))
              actions_mapped[agent_map[j]]=float(round(action[0].item()))
              
            new_state, rewards, infos = env.step(actions_mapped)
            rewards  = list(rewards.values())

            print(f"f1: {new_state[1]}")
            print(f"fairness: {new_state[0]}")
            writer1.writerow(rewards)
            
            episodic_reward += sum(rewards)
            prev_state = new_state

        # Mean of last 40 episodes
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)
        
        Path(f"FL Models Agg/ep{ep}/").mkdir(parents=True, exist_ok=True)

        for i  in env.agents:
           env.models[i].save(f"FL Models Agg/ep{ep}/{i}.keras")

        writer2.writerow([episodic_reward, new_state[1], new_state[0], "-" , avg_reward, new_state, infos])