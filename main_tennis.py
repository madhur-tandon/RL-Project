from actor import Actor
from critic import Critic

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from unityagents import UnityEnvironment
from collections import deque
from itertools import count
import datetime

from noise import OUNoise
from replay_buffer import ReplayBuffer

from agent import Agent

env = UnityEnvironment(file_name="./Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

BUFFER_SIZE = int(1e6)
BUFFER_FILL = int(1e4)
CACHE_SIZE = int(1e3)
NUM_UPDATES_CACHE = 2
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0
UPDATE_EVERY = 20
NUM_UPDATES = 15
EPSILON = 1.0
EPSILON_DECAY = 1e-6
NOISE_SIGMA = 0.05

fc1_units=96
fc2_units=96

random_seed=23
RECREATE_EVERY=1

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]

avg_over = 100
print_every = 10

def store(buffers, states, actions, rewards, next_states, dones, timestep):
    memory, cache = buffers
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        memory.add(state, action, reward, next_state, done)
        cache.add(state, action, reward, next_state, done)

def learn(agent, buffers, timestep, i_episode):
    memory, cache = buffers
    if len(memory) > BUFFER_FILL and timestep % UPDATE_EVERY == 0: 
        for _ in range(NUM_UPDATES):
            experiences = memory.sample()
            agent.learn(experiences, GAMMA)
        for _ in range(NUM_UPDATES_CACHE):
            experiences = cache.sample()
            agent.learn(experiences, GAMMA)
    elif timestep == 0 and i_episode % RECREATE_EVERY == 0:        
        agent.reset()

def ddpg(agent, buffers, n_episodes=200, stopOnSolved=True):
    scores_deque = deque(maxlen=avg_over)
    scores_global = []
    average_global = []
    min_global = []    
    best_avg = -np.inf

    tic = time.time()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        
        score_average = 0
        timestep = time.time()
        for t in count():
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done       
            store(buffers, states, actions, rewards, next_states, dones, t)
            learn(agent, buffers, t, i_episode)
            states = next_states
            scores += rewards         
            if np.any(dones):
                break
        
        score = np.max(scores)        
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores_global.append(score)
        average_global.append(score_average)  
        min_global.append(np.min(scores))  
        
        if i_episode % print_every == 0:
            agent.save('./')
            print('\r {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'\
              .format(str(i_episode).zfill(3), score, score_average, 
                      np.min(scores), time.time() - timestep), len(buffers[0]), end="\n")
        if  stopOnSolved and score_average > 0.5:            
            print('\nSolved in {:d} episodes!\tAvg Score: {:.2f}'.format(i_episode, score_average))
            agent.save('./'+str(i_episode)+'_')
            break
     
    print('End: ',datetime.datetime.now())
    return scores_global, average_global, min_global

buffers = [ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, modification=False, seed=random_seed),
           ReplayBuffer(action_size, CACHE_SIZE, BATCH_SIZE, modification=False, seed=random_seed)]
agent = Agent(state_size=state_size, action_size=action_size, modification=False, random_seed=23, fc1_units=96, fc2_units=96, environment='tennis')
scores, averages, minima = ddpg(agent, buffers, n_episodes=2000)

plt.plot(np.arange(1, len(scores)+1), scores, label='DDPG')
plt.plot(np.arange(1, len(averages)+1), averages, c='r', label='moving avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left')
plt.show()

def play(agent, episodes=3):
    for i_episode in range(episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = np.random.randn(num_agents, action_size)
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break
            time.sleep(.050)
        print('Ep No: {} Total score (averaged over agents): {}'.format(i_episode, np.max(scores)))

agent = Agent(state_size=state_size, action_size=action_size, modification=False, random_seed=23, 
             fc1_units=96, fc2_units=96, environment='tennis')
agent.load('./tennis_actor.pth', './tennis_critic.pth')

play(agent, episodes=10)