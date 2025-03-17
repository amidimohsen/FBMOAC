import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import seaborn as sns
# import gym
# import roboschool
# from Environment_FBMDP_OMPMC import NetEnv
from EnvGithub import NetEnv

import matplotlib.pyplot as plt

from Agent_PPo_cuda import PPOAgent, ActorCritic

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

################################### Training ###################################
def train():
    print("============================================================================================")
    ####### initialize environment hyperparameters ######
    max_ep_len = 256                    # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timesteps > max_training_timesteps

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80                         # update policy for K epochs in one PPO update
    eps_clip = 0.2                        # clip parameter for PPO
    gamma = 0.92                          # discount factor
    lr_actor_critic = 0.00001

    #  environment parameters.
    env             = NetEnv()
    CacheCapacity   = env.CacheCapacity
    state_dim       = env.StateDim_FW
    action_dim      = env.ActionDim

    ################# training procedure ################
    # initialize a PPO actor-critic network and move it to device
    actor_critic = ActorCritic(state_dim, action_dim, CacheCapacity).to(device)
    
    # initialize a PPO agent (inside its __init__, set train_device accordingly)
    ppo_agent = PPOAgent(actor_critic, gamma, eps_clip, lr_actor_critic, K_epochs, CacheCapacity, device)
    
    # track total training time
    time_step = 0
    i_episode = 0

    reward_history_FW    = [] 
    reward_history_BW    = []
    reward_QoS_history   = [] 
    reward_BanW_history  = []
        
    # training loop
    while time_step <= max_training_timesteps:
        ''' Forward Evaluation '''
        state = env.ForwardReset()  # assume this returns a numpy array
        StateFW_Acc   = np.zeros([0, state_dim])
        Lambda_UE_Acc = np.zeros([0, state_dim])
        Action_Acc    = np.zeros([0, 2 * state_dim])
        
        current_ep_reward = 0
        AccumRewardFW, AccumReward_QoS, AccumReward_BanW, AccumReward_BakH = 0, 0, 0, 0
        
        for t in range(1, max_ep_len + 1):
            # select action with policy (which now runs on CUDA)
            action_cache, action_BW = ppo_agent.select_action(state)
            action = np.concatenate((action_cache, action_BW), axis=0)

            time_step = ppo_agent.timestep  # update tracking variable
            # Next_state, RewardFW, done, Reward_QoS, Reward_BandW, _, Lambda_UE = env.ForwardStep(action, t)
            Next_state, RewardFW, CoupledState = env.ForwardStep(action, t)
            Lambda_UE = CoupledState
            
            # Accumulate rewards
            AccumRewardFW += RewardFW[0] + 1/3*RewardFW[1]
            AccumReward_QoS += RewardFW[0]
            AccumReward_BanW += RewardFW[1]

            # Accumulate action, state 
            Lambda_UE_Acc = np.concatenate((Lambda_UE_Acc, np.expand_dims(Lambda_UE, 0)), axis=0)
            Action_Acc    = np.concatenate((Action_Acc, np.expand_dims(action, 0)), axis=0)
            StateFW_Acc   = np.concatenate((StateFW_Acc, np.expand_dims(state, 0)), axis=0)
            
            done = (t == max_ep_len)
            ppo_agent.buffer.rewards.append(RewardFW)
            current_ep_reward += RewardFW

            # update PPO agent if enough timesteps are collected
            if ppo_agent.timestep >= update_timestep:
                ppo_agent.update()
                # Optionally, break out of the episode loop if update() resets state.
                # break

            if done:
                break
            
            state = Next_state
        
        StateFW_Acc = np.concatenate((StateFW_Acc, np.expand_dims(np.zeros_like(state), 0)), axis=0)
            
        ''' Step 2: Backward Evaluation '''
        StateBW = env.BackwardReset()
        AccumRewardBW = 0
        done = True
        BackwardTimestep = max_ep_len
        
        while done:
            BackwardTimestep -= 1
            if BackwardTimestep < 1:
                done = False
                
            Action = Action_Acc[BackwardTimestep-1, :]
            NewStateBW, RewardBW = env.BackwardStep(Action, Lambda_UE_Acc[BackwardTimestep-1, :])
            AccumRewardBW += RewardBW
            # StateBW = NewStateBW  (if needed)
        
        reward_history_FW.append(AccumRewardFW)      
        reward_history_BW.append(AccumRewardBW)
        reward_QoS_history.append(AccumReward_QoS)      
        reward_BanW_history.append(AccumReward_BanW)  
        
        i_episode += 1
        
        ''' plotting '''
        sns.set()
        plt.figure(0)
        plt.clf()
        plt.plot(reward_QoS_history, 'g', alpha=0.8)
        plt.plot(reward_BanW_history, 'r', alpha=0.8)
        plt.plot(reward_history_BW, 'b', alpha=0.8)
        plt.xlabel("Episode Number")
        plt.ylabel("Cumulative Reward")
        plt.legend(["$r_{QoS}$", "$r_{BW}$", "$r_{Lat}$"])
        allResults = torch.cat((torch.tensor(reward_QoS_history)[:, None],
                                 torch.tensor(reward_BanW_history)[:, None],
                                 torch.tensor(reward_history_BW)[:, None]), dim=1)
        np.save('AllRewards_PPO_MPMC_N200_Lat3__3.npy', allResults.numpy())
        plt.pause(0.003)
        plt.savefig('TrainPerformance_PPO__3.png')
        plt.show()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Finished training at (GMT):", end_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
