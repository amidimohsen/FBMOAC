# In the Name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:29:48 2023
"""

import numpy as np
import torch
import seaborn as sns
from FBMOAC import Agent, ForwardCriticBase, BackwardCriticBase, Actor
from environments.EdgeCaching import NetEnv                               # Uncomment it for the edge caching experiment
# from environments.ComputationOffloading import NetEnv                   # Uncomment it for the computation offloading experiment
import matplotlib.pyplot as plt


# %% 
def train():
    print("=============================================================")
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Global Parameters '''
    Print_freq       = 25                                    # The frequency based on which the results are printed. (after how many episodes)
    Save_model_freq  = 1000                                  # The frequency based on which the parameters of model are saved.
    AverageFrequency = 64                                    # The frequency  based on which the cumulative rewards are averagd for the printing and logging purposes.


    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Initializing algorithm hyperparameters '''
    N_MCS           = 4                                      # Number of Monte-Carlo Samples for the episodic MCS-average  add-on
    EpisodeNumber   = 40000                                  # Number of training episodes
    TimeSlots       = 256                                    # Number of time-steps in each episode
    LearningRate    = 3e-4                                   # Learning-Rate of the FB-MOAC algorithm (for the multi-objective actor, forward critic and backward critic networks)
    SmoothingFactor = 0.95                                   # The smoothing factor of the episodic MCS-average  add-on
    DiscountFactor  = 0.92                                   # Discount-factor related to the cumulative rewards.
    PreferenceCoeff = torch.tensor([1,  1,  1/3])            # Preference parameter to extract a Pareto-front.
    
    print("-------------------------------------------------------------") 
    print("Number of training episodes = {}\nNumber of time-steps in each episode = {}\nLearning Rate = {}\nDiscount Factor = {}\nNumber of Monte-Carlo Samples = {}\nsmoothing factor = {}".\
          format(EpisodeNumber, TimeSlots, LearningRate, DiscountFactor, N_MCS, SmoothingFactor))
       
          
    
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Environment Parameters '''
    env = NetEnv()                                           # Importing the environment
    env_name = env.env_name                                  # Name of the environment
    
    ActionDim            = env.ActionDim                     # Action-space dimension
    MaxDirichletAction   = env.MaxDirichletAction        
    NormalActionDim      = env.NormalActionDim               # Dimension of unrestricted action drawn from Guassian distribution
    DirichletActionDims  = env.DirichletActionDims           # List of dimensions of (dirichlet) actions which lie within [0, 1] with their sum equal to list of MaxDirichletAction
    PositiveActionDim    = env.PositiveActionDim             # Dimension of actions which are positive
    CategoricalRange     = env.CategoricalRange              # Discrete range of the categorical action
    CategoricalActionDim = env.CategoricalActionDim          # Dimension of categorical action  which belongs to the set {0, CategoricalRange}
    BetaActionDim        = env.BetaActionDim                 # List of dimensions of actions which lie within [0, 1]  
    
    StateDim_FW          = env.StateDim_FW                   # Space dimension of forward state
    StateDim_BW          = env.StateDim_BW                   # Space dimension of backward state
    N_forwadRewards      = env.N_forwadRewards               # Number of forwad rewards
    N_backwadRewards     = env.N_backwadRewards              # Number of backward rewards
    CoupledStateDim      = env.CoupledStateDim               # Space dimension of variables coupled between forward and backward dynamics
    
    Optimizer = env.Optimizer                                # Type of the optimizer, Adam or SGD.
    Resampling_flag = env.resampling_flag                    # The sampling method
    
    Legends_of_Rewards = env.RewardLegend                    # The legends of the forward-backward rewards, needed for plotting purposes.
    
    print("-------------------------------------------------------------") 
    print( "Environemnt = {}\nSpace dimension of forward state = {}\nSpace dimension of backward state = {}\nAction-space dimension = {}". format(env_name,StateDim_FW,StateDim_BW,ActionDim))
    
    
    
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Making a directory to log the results and parameters of the trained model'''
    directory = "Results"
    directory = directory + '/' + env_name 
    if not os.path.exists(directory):
          os.makedirs(directory)
    print("-------------------------------------------------------------") 
    print( "Directory to save results and model = ",directory )          
          
    
    
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Instantiating the forward-backward critics and single-policy actor of FB-MOAC '''
    actor = Actor(StateDim_FW, DirichletActionDims, PositiveActionDim, \
                  CategoricalActionDim, CategoricalRange, NormalActionDim, BetaActionDim)      # Instantiate actor
    
    criticFWBase = ForwardCriticBase( StateDim_FW, N_MCS, N_forwadRewards, LearningRate)                     # Instantiate forward-critic
    FWCriticOptimizers = criticFWBase.optimizers                                                             # set the forward-critic optimizer
    
    criticBWBase = BackwardCriticBase( StateDim_FW + StateDim_BW, N_MCS, N_backwadRewards, LearningRate)     # Instantiate backward-critic
    BWCriticOptimizers = criticBWBase.optimizers                                                             # set the backward-critic optimizer
    

    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Instantiating the agent of FB-MOAC '''
    # Instantiate agent  
    agent = Agent(actor, criticFWBase, criticBWBase, FWCriticOptimizers, BWCriticOptimizers, MaxDirichletAction,  N_MCS, DiscountFactor, SmoothingFactor, LearningRate)
    

    
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\    
    ''' Arrays to keep track of forward-backward rewards and their averages'''
    reward_history_FW, reward_history_BW = [], []
    
    
    print("=============================================================")
    
    
    
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ''' Start of traingin the model '''
    for n_episode in range(EpisodeNumber):
        
        ''' Step 1: Forward Evaluation Step '''
        # Reset forward state
        StateFW = env.ForwardReset()
        
        # Initialize coupled States between forward and backward
        CoupledState_Acc = np.zeros([0,CoupledStateDim])
        
        # Initialize accumulated action
        Action_Acc = np.zeros([0, ActionDim])
        
        # Initialize acumulated forward state
        StateFW_Acc = np.zeros([0, StateDim_FW ])
        
        AccumRewardFW = 0
        done = True
        ForwardTimestep=0
        
        while done:
            ForwardTimestep += 1
            
            if ForwardTimestep > TimeSlots-1:
                done = False
                
                
            # Get Action from the actor neural network
            Action, Log_policyFW, EntropyFW = agent.Get_Action(StateFW, Resampling_flag)
            
            # Get forward stateValue funcction from the forward-critic neural network
            StateValue_FW = agent.Get_ForwardStateValue(StateFW, N_MCS)
            
            # Take action, go to the next action and receive immediate forward rewards
            NewStateFW, RewardFW, CoupledState = env.ForwardStep(Action.detach().numpy(), ForwardTimestep)
            
            # Terminal State-values functions
            NewStateValue_FW = [torch.zeros(size=[N_forwadRewards]).float() for _ in range(N_MCS)]
            if done == True:
                NewStateValue_FW_ = criticFWBase.forward( torch.from_numpy(NewStateFW).float() )
                for i in range(N_MCS):
                    NewStateValue_FW[i]  =  NewStateValue_FW_[i]
                
                
            # Accumulate action, forward state and coupled state
            CoupledState_Acc = np.concatenate((CoupledState_Acc, np.expand_dims(CoupledState,0)), axis=0)
            Action_Acc = np.concatenate( (Action_Acc, np.expand_dims(Action.detach().numpy(),0)), axis=0)
            StateFW_Acc = np.concatenate( (StateFW_Acc, np.expand_dims(StateFW, 0)), axis=0)
            
            # Accumulate forward rewards, policy distribution, state-value functions, distribution entropy etc.
            agent.AccumulateFW(RewardFW, Log_policyFW, StateValue_FW, NewStateValue_FW, EntropyFW, N_MCS)
            
            # Accumulate forward rewards
            AccumRewardFW+= RewardFW
            
            StateFW = NewStateFW
                        
        
        StateFW_Acc = np.concatenate( (StateFW_Acc, np.expand_dims(np.zeros_like(StateFW),0)), axis=0)


        #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
        ''' Step 2: Backward Evaluation Step '''
        # Reset Backward MDP
        StateBW = env.BackwardReset()
        
        AccumRewardBW = 0
        done = True
        BackwardTimestep = TimeSlots
        
        while done:
            BackwardTimestep -= 1
            
            if BackwardTimestep < 1:
                done = False
                
                
            # Get current forward state
            StateFW = StateFW_Acc[BackwardTimestep+1,:]
            
            # Augmenting the backward state with the forward state
            StateBW_augmented = np.concatenate((StateBW, StateFW), axis=0)
            
            
            # Get the action from the forwad evaluation step
            Action = Action_Acc[BackwardTimestep,:]
            
            
            # Get backward StateValue function from the actor neural network
            StateValueBW = agent.Get_BackwardStateValue(StateBW_augmented, N_MCS)
            
            
            # Take action, go to the next action and receive reward
            NewStateBW, RewardBW = env.BackwardStep(StateBW, Action, CoupledState_Acc[BackwardTimestep,:])
            
            # Terminal State-values functions
            NewStateValueBW = [torch.tensor(0).float().unsqueeze(0) for _ in range(N_MCS)]
            if done == True:
                NewStateFW = StateFW_Acc[BackwardTimestep-1,:]
                NewStateBW_Augmented =  np.concatenate((NewStateBW, NewStateFW ), axis=0)
                NewStateValueBW = criticBWBase.forward(torch.from_numpy(NewStateBW_Augmented).float())
                
            # Accumulate backward rewards and state-value functions.
            agent.AccumulateBW(RewardBW, StateValueBW, NewStateValueBW, N_MCS)
                
            
            # Accumulate backward rewards
            AccumRewardBW+= RewardBW

            
            StateBW = NewStateBW

        
        #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
        ''' Step 3: Forward-Backward Optimization (updating forward-critic, backward-critic and actor agents) '''
        agent.ForwardBackwardActorCriticUpdate(PreferenceCoeff, N_MCS, N_forwadRewards, N_backwadRewards, Optimizer) 
            
        
        
        #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
        ''' Printing, logging, and saving the model parameters'''
        # Appending for plotting '''
        reward_history_FW.append(AccumRewardFW)      
        reward_history_BW.append(AccumRewardBW)
        
        
        # Averaging of forward and backward rewards
        if n_episode > AverageFrequency:
            avg_FW = np.mean(np.stack(reward_history_FW)[-AverageFrequency:,:], axis=0)
            avg_BW = np.mean(np.stack(reward_history_BW)[-AverageFrequency:,:], axis=0)
        else:
            avg_FW = np.mean(np.stack(reward_history_FW), axis=0)
            avg_BW = np.mean(np.stack(reward_history_BW), axis=0)
        
        
        sns.set()
        plt.figure(0)
        plt.clf()
        plt.plot( np.concatenate(reward_history_FW, axis=1).T, alpha=0.8)
        plt.plot( np.concatenate(reward_history_BW, axis=1).T, alpha=0.8)
        plt.xlabel("Episode Number")
        plt.ylabel("Cumulative Reward")
        plt.legend(Legends_of_Rewards)
        plt.ylim(-4000,100)
        plt.show()

        if n_episode%Print_freq == 0:
            print("Episode {} Finished, Forward rewards={}, Backward Rewards={}.\n\n".\
                  format(n_episode, avg_FW, avg_BW))
    
        if n_episode%Save_model_freq ==0:
            allResults = torch.cat( (torch.tensor(reward_history_FW),\
                                     torch.tensor(reward_history_BW) ), dim = 1 )
            file_path = os.path.join(directory, "Rresults.npy")
            np.save(file_path, allResults)
            file_path = os.path.join(directory, "TrainPerformance.png")
            plt.savefig(file_path)
            # plt.show()
            file_path = os.path.join(directory, "FBMOAC_ActorAgent.mdl")
            torch.save(agent.Actor.state_dict(), file_path)
         
    
   
    
# %%  Main function  
if __name__ == "__main__":
    train()
            
