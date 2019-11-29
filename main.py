import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent

environment = 'pendulum'
basePath = '/content/gdrive/My Drive/RL/'
basePath = '.'

modification = True

n_experiments = 2
n_episodes = 1000
max_t = 300
env = gym.make('Pendulum-v0')
env.seed(2)
agent = Agent(state_size=3, action_size=1, modification=modification, random_seed=2, fc1_units=400, fc2_units=300)

def plot(mean_scores, std_scores, mean_avg_reward, std_avg_reward, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, n_episodes+1)
    
    plt.figure()
    plt.plot(x, mean_scores, label='average return')
    plt.fill_between(x, mean_scores-std_scores, mean_scores+std_scores, alpha=0.5)
    plt.legend(loc='best')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if modification:
        plt.savefig('{0}/plots/{1}/avg_return_plot_{2}'.format(basePath, 'avg_returns_modified', i+1))
    else:
        plt.savefig('{0}/plots/{1}/avg_return_plot_{2}'.format(basePath, 'avg_returns', i+1))
    
    plt.figure()
    plt.plot(x, np.cumsum(mean_avg_reward), label='average reward')
    plt.fill_between(x, np.cumsum(mean_avg_reward)-np.cumsum(std_avg_reward), np.cumsum(mean_avg_reward)+np.cumsum(std_avg_reward), alpha=0.5)
    plt.legend(loc='best')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if modification:
        plt.savefig('{0}/plots/{1}/avg_cumulative_reward_plot_{2}.png'.format(basePath, 'avg_cum_rewards_modified', i+1))
    else:
        plt.savefig('{0}/plots/{1}/avg_cumulative_reward_plot_{2}.png'.format(basePath, 'avg_cum_rewards', i+1))

    plt.figure()
    plt.plot(x, mean_scores, label='average return')
    plt.fill_between(x, mean_scores-std_scores, mean_scores+std_scores, alpha=0.5)
    plt.plot(x, np.cumsum(mean_avg_reward), label='average reward')
    plt.fill_between(x, np.cumsum(mean_avg_reward)-np.cumsum(std_avg_reward), np.cumsum(mean_avg_reward)+np.cumsum(std_avg_reward), alpha=0.5)
    plt.legend(loc='best')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if modification:
        plt.savefig('{0}/plots/{1}/avg_cumulative_reward_and_return_plot_{2}.png'.format(basePath, 'avg_cum_rewards_and_returns_modified', i+1))
    else:
        plt.savefig('{0}/plots/{1}/avg_cumulative_reward_and_return_plot_{2}.png'.format(basePath, 'avg_cum_rewards_and_returns', i+1))


def ddpg(n_episodes, max_t, save_freq):
    scores = np.zeros((n_experiments, n_episodes))
    avg_reward = np.zeros((n_experiments, n_episodes))
    max_experiment_index = np.full(n_episodes,-np.inf)
    for i in range(n_experiments):
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            agent.reset()
            score = 0
            count = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                count += 1
                if done:
                    break 
            scores[i, i_episode-1]+= score
            avg_reward[i, i_episode-1] += (score / min(count, max_t))
            print('\rExperiment {}\tEpisode {}\tScore: {:.2f}'.format(i, i_episode, score))
            if i_episode % save_freq == 0:
                if scores[i, i_episode-1] >= max_experiment_index[i_episode-1]:
                    max_experiment_index[i_episode-1] = scores[i, i_episode-1]
                    if modification:
                        torch.save(agent.actor_local.state_dict(), '{0}/{1}/actors/checkpoint_actor_{2}_modified.pth'.format(basePath, environment+'_modified', i_episode))
                        torch.save(agent.critic_local.state_dict(), '{0}/{1}/critics/checkpoint_critic_{2}_modified.pth'.format(basePath, environment+'_modified', i_episode))
                    else:
                        torch.save(agent.actor_local.state_dict(), '{0}/{1}/actors/checkpoint_actor_{2}.pth'.format(basePath, environment, i_episode))
                        torch.save(agent.critic_local.state_dict(), '{0}/{1}/critics/checkpoint_critic_{2}.pth'.format(basePath, environment, i_episode))
        mean_scores = np.mean(scores[:i+1,:], axis=0)
        std_scores = np.std(scores[:i+1,:], axis=0)

        mean_avg_reward = np.mean(avg_reward[:i+1,:], axis=0)
        std_avg_reward = np.std(avg_reward[:i+1,:], axis=0)

        plot(mean_scores, std_scores, mean_avg_reward, std_avg_reward, i)

    return mean_scores, std_scores, mean_avg_reward, std_avg_reward

# mean_scores, std_scores, mean_avg_reward, std_avg_reward = ddpg(n_episodes, max_t, 1)

if modification:
    agent.actor_local.load_state_dict(torch.load('{0}/{1}/actors/checkpoint_actor_{2}_modified.pth'.format(basePath, environment+'_modified', n_episodes), map_location={'cuda:0': 'cpu'}))
    agent.critic_local.load_state_dict(torch.load('{0}/{1}/critics/checkpoint_critic_{2}_modified.pth'.format(basePath, environment+'_modified', n_episodes), map_location={'cuda:0': 'cpu'}))
else:
    agent.actor_local.load_state_dict(torch.load('{0}/{1}/actors/checkpoint_actor_{2}.pth'.format(basePath, environment, n_episodes), map_location={'cuda:0': 'cpu'}))
    agent.critic_local.load_state_dict(torch.load('{0}/{1}/critics/checkpoint_critic_{2}.pth'.format(basePath, environment, n_episodes), map_location={'cuda:0': 'cpu'}))

state = env.reset()
done = False
while not done:
    action = agent.act(state, add_noise=False)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        print('done')
        break 

env.close()
