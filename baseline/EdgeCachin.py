# In the Name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:44:56 2022
"""

import numpy as  np
import torch as torch
from scipy.special import erfc
np.seterr(over='ignore')


class NetEnv:
    # %%
    def __init__(self):
        self.env_name = 'EdgeCaching'
        
        self.Nfile            = 200                      # Number of contents to be multicasted
        self.Skewness         = 0.6                      # Skewness of content popularity
        self.StateDim_FW      = self.Nfile               # Dimension of forward state
        self.StateDim_BW      = self.Nfile               # Dimension of backward state
        self.CacheCapacity    = 10                       # Cache capacity of serving nodes
        self.ActionDim        = 2*self.Nfile+1           # Dimension of action
        self.CoupledStateDim  = self.Nfile               # Dimension of variables coupled between forward and backward states
        self.N_forwadRewards  = 2                        # Number of forward reward functions
        self.N_backwadRewards = 1                        # Number of backward reward functions


        self.DirichletActionDims  = [self.Nfile]            # List of dimensions of (dirichlet) actions which lie within [0, 1] with their sum equal to list of MaxDirichletAction
        self.MaxDirichletAction   = [self.CacheCapacity]    # List of summations of dirichlet distributions
        self.NormalActionDim      = 0                       # Dimension of unrestricted actions with normal distribution
        self.PositiveActionDim    = self.Nfile              # Dimension of positive actions with lognormal distribution
        self.CategoricalActionDim = 1                       # Dimension of categorical action
        self.CategoricalRange     = 10                      # Range of categorical action
        self.BetaActionDim        = 0                       # Dimensions of actions which lie within [0, 1] 
        
        
        self.Lambda_UE_fixed     = 1e5                      # Spatial intensity of users
        self.Lambda_HN           = 100                      # Spatial intensity of serving nodes
        self.Rate                = 1e3                      # Desired information rate of contents
        self.P_N0_               = 2e7                      # Ratio between transmitting power of serving nodes to the spectral noise
        self.FileDuration        = 600.0                    # Slot duration in seconds
        self.NormalizationFactor = 1000
        self.gammaR = self.P_N0_/self.Rate/self.NormalizationFactor             


        self.CacheWeight_old = [[np.zeros(self.Nfile)]]     # Cache probability of serving nodes
        self.Lambda_UE_n = np.zeros([self.Nfile])           # Spatial intensity of "requesting" users
        self.Total_Outage = np.zeros([self.Nfile])          # Instantaneous total outage of content transmissions

        # Harmonic numbers of the harmonic broadcasting to effeciently decrease the latency
        self.invLatencyRelatedToHarmonic = np.array([1.0, 4.0, 11.0, 30.0, 83.0, 227.0, 620.0, 1680.0, 4550.0, 12400.0])
        self.Latency = self.FileDuration/self.invLatencyRelatedToHarmonic  # Instantaneous latency for different tasks            

        self.data_arr         = []                                         # Content popularity of different contents
        self.ExperiencedDelay = []                         
        self.I_t              = []
        
        self.Optimizer = "SGD"
        self.resampling_flag = False
        
        self.RewardLegend = ["$r_{QoS}$", "$r_{BW}$", "$r_{Lat}$"]         # The legends of the forward-backward rewards, needed for plotting purposes.
                                                                           # r_QoS: Quality-of-Serveice, r_BW: bandwidth consumption, r_Lat:  overal expected latency
        
    # %%
    def ForwardStep(self, Action, TimeStep):


        ''' File Popularity Evolution '''
        Popularity = self.data_arr[4*TimeStep,:]


        ''' Action variables '''
        CacheWeight  = Action[0              : self.Nfile]
        BW_Allocate  = Action[self.Nfile     : 2*self.Nfile]
        Harmionic_BW = Action[2*self.Nfile   : 1+2*self.Nfile]
        Delay = self.Latency[int(Harmionic_BW-1)]

        ''' '''
        threshold = 1e-12
        g_alpha = np.where(BW_Allocate < threshold, 0, (2**(1/BW_Allocate)-1)*np.sum(BW_Allocate))


        ''' Network Operation '''

        ''' Multicast Transmission and Outage '''
        Outage_MC_t = erfc(self.Lambda_HN * np.pi**2/4 * CacheWeight * np.sqrt(self.gammaR/g_alpha/Harmionic_BW))


        '''  Updating UEs Intensity that are interested in files'''
        if TimeStep == 1:
            Lambda_UE_n =  Popularity * self.Lambda_UE_fixed
        else:
            Lambda_UE_n  =  Popularity * np.sum( self.Lambda_UE_n*(1.0-self.Total_Outage) ) +\
                            self.Lambda_UE_n * self.Total_Outage
        self.Lambda_UE_n = Lambda_UE_n


        ''' Request from Unicast '''
        ReqIntensity_UC_n = Lambda_UE_n * Outage_MC_t
        ReqIntensity_UC = ReqIntensity_UC_n.sum()


        ''' Intensity of unsatisified UEs '''
        self.Inten_UnsatisUEs_n = ReqIntensity_UC_n 
        Forward_State = ReqIntensity_UC_n
        
        ''' Reward Computatoin '''
        self.CacheWeight_old = CacheWeight


        # ''' Rewards (total outage, total resource consumption)'''
        # Outage_total_bar = ( Func_delay(Delay) + (1 - Func_delay(Delay)) * Outage_MC_t ) * Outage_UC_t
        Outage_total_bar = Outage_MC_t
        Reward_QoS_t =  -(  Lambda_UE_n * Outage_total_bar  ).sum()  /4.5e3
        Reward_BW_t = - (Harmionic_BW * self.Rate * self.NormalizationFactor * np.sum(BW_Allocate) )/3e8    #1e8

        self.Total_Outage =  Outage_total_bar

        ''' Forward Reward '''
        Forward_Reward = np.concatenate( (Reward_QoS_t[None],  Reward_BW_t), axis=0)[:,None]

        CoupledState = Lambda_UE_n

        return Forward_State, Forward_Reward, CoupledState


  # %%
    def BackwardStep(self, Backward_State,  Action, CoupledState):

        
        ''' Action variables '''
        CacheWeight  = Action[0              : self.Nfile]
        BW_Allocate  = Action[self.Nfile     : 2*self.Nfile]
        Harmionic_BW = Action[2*self.Nfile   : 1+2*self.Nfile]
        Delay = self.Latency[int(Harmionic_BW-1)]


        ''' '''
        threshold = 1e-12
        g_alpha = np.where(BW_Allocate < threshold, 0, (2**(1/BW_Allocate)-1)*np.sum(BW_Allocate))


        ''' Network Operation '''

        ''' Multicast Transmission and Outage '''
        Outage_MC_t = erfc(self.Lambda_HN * np.pi**2/4 * CacheWeight * np.sqrt(self.gammaR/g_alpha/Harmionic_BW))


        ''' File-specific Experienced Delay '''
        ExperiencedDelay = Outage_MC_t * (Delay + Backward_State) + Delay/2 * (1 - Outage_MC_t)
        Backward_State_new = ExperiencedDelay


        ''' Reward Computatoin '''
        ''' Total Experiencd Delay '''
        Lambda_UE_n = CoupledState
        Reward_delay = -np.sum(Lambda_UE_n * ExperiencedDelay)*0.5e-6
        Backward_Reward = Reward_delay[None][:,None]

        return Backward_State_new, Backward_Reward


# %%
    def BackwardReset(self):
        self.ExperiencedDelay = np.zeros([self.Nfile])
        return self.ExperiencedDelay


    # %%

    def ForwardReset(self):
        np.random.seed(10)

        ''' File Popularity Based on diffusion Model '''
        TimeSteps = 2000
        tau  = self.Skewness
        N_file = self.Nfile
        Zipf = np.linspace(1,N_file,N_file)**(-tau)
        Zipf = Zipf/Zipf.sum()
        Zipf = np.random.permutation(Zipf)
        P_vec = Zipf
        t0_vec = np.random.randint(1, TimeSteps, size=[1, N_file])
        HalfWidth_vec = np.random.randint(100, 300, size=[1, N_file])
        Pop = np.zeros([N_file,0])
        for i in range(TimeSteps):
            fn = 2 * P_vec/np.cosh((i-t0_vec[0])/HalfWidth_vec[0])
            fn = np.expand_dims(fn, 1)
            Pop= np.append(Pop, fn, 1)
        Pop = Pop/np.sum(Pop, 0)


        self.data_arr = Pop.T


        Popularity = self.data_arr[0,:]

        ''' Intensity of interested UEs towards files '''
        self.I_t = Popularity * self.Lambda_UE_fixed

        return self.I_t

