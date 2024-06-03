# In the Name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:44:56 2022

"""

import numpy as  np
import torch as torch
# torch.manual_seed(10)
# np.random.seed(10)


class NetEnv:
    # %%    
    def __init__(self):
        self.env_name = 'ComputationOffloading'
        
        self.Nfile    = 100                                          # number of tasks
        self.N_UE     = 20                                           # number of users offloading their computations
        self.FileSize = 10.0 * np.ones([self.Nfile])            
        self.FileSize = np.linspace(1, 100, self.Nfile) + 10.0       # tasks size in [KBits]
        self.N_UE_times_Filesize = self.N_UE * self.FileSize
        self.Skewness = 0.6                                          # skewness of tasks popularity
        
        self.StateDim_FW     = 2*self.Nfile                          # dimension of forward state
        self.StateDim_BW     = 1*self.Nfile                          # dimension of backward state
        self.ActionDim       = 3*self.Nfile                          # dimension of action state
        self.CoupledStateDim = 2*self.Nfile                          # dimension of variables coupled between forward and backward states
        self.N_forwadRewards  = 1                                    # number of forward reward functions
        self.N_backwadRewards = 1                                    # number of backward reward functions

        self.ResourceComputationUE = 10                              # computation resource of each user in [Kbits/slot]
        self.BufferCapacity        = 100                             # buffer capacity of the cloud [Kbits]
        self.ComputationCapacity   = 100                             # computation resource of the cloud in [Kbits/slot], should be greater than that of users
        
        self.DirichletActionDims  = [self.Nfile, self.Nfile]                         # List of dimensions of (dirichlet) actions which lie within [0, 1] with their sum equal to list of MaxDirichletAction
        self.MaxDirichletAction   = [self.ComputationCapacity,self.BufferCapacity]   # List of summations of dirichlet distributions
        self.NormalActionDim      = 0                                                # dimension of unrestricted actions with normal distribution
        self.PositiveActionDim    = 0                                                # dimension of positive actions with lognormal distribution
        self.CategoricalActionDim = 0                                                # dimension of categorical action
        self.CategoricalRange     = 0                                                # range of categorical action
        self.BetaActionDim        = self.Nfile                                       # List of dimensions actions which lie within [0, 1] 
        
        self.SlotDuration     = 60.0                                   # Slot duration in [Seconds]
        self.QueueLength      = np.zeros([self.Nfile])                 # Queue length of different buffers in [KBits]
        self.Outage           = np.zeros([self.Nfile])                 # Overflow probability of different buffer [0,1]
        self.OffloadProb      = np.zeros([self.Nfile])                 # Offloading probability of different tasks
        self.ExperiencedDelay = []                                     # Instantaneous experienced delay for different tasks
        self.I_t = []
        
        self.Optimizer = "Adam"
        self.resampling_flag = True
        
        
    # %%
    def ForwardStep(self, Action, TimeStep):
        
        ''' File Popularity Evolution '''
        Popularity = self.data_arr[4*TimeStep,:]
            
            
        
        ''' Action variables '''
        ComputationResource  = Action[0            : self.Nfile]      # [Kbits/slot]
        BufferCapacities     = Action[self.Nfile   : 2*self.Nfile]    # [Kbits]
        OffloadAction        = Action[2*self.Nfile : 3*self.Nfile]    # Pr( Offload | Prefer ) , [0,1]
        
        
        
        ''' Preference and Offloading Probability '''
        PreferenceProb = Popularity * ( 1 - np.sum( self.Outage * self.OffloadProb) ) + self.Outage * self.OffloadProb
        OffloadProb = OffloadAction * PreferenceProb
        self.OffloadProb = OffloadProb
        
        
        ''' Overflow probability '''
        Alpha = (BufferCapacities - self.QueueLength) / OffloadProb / self.N_UE_times_Filesize
        Alpha = (Alpha >= 1.0) * 1.0 + (Alpha < 1.0) * Alpha
        Outage  = 1 - Alpha
        self.Outage = Outage
        
        
        ''' Computing the Queues Length '''
        QueueLength = self.QueueLength + self.N_UE_times_Filesize * Alpha * OffloadProb - ComputationResource
        QueueLength = (QueueLength <=0 )*0 + (QueueLength > 0) * QueueLength
        self.QueueLength=  QueueLength
        
        ''' Computing the Queues Length '''
        # QueueLength = self.QueueLength  - ComputationResource
        # QueueLength = (QueueLength <=0 )*0 + (QueueLength > 0) * QueueLength
        # QueueLength = QueueLength + self.N_UE_times_Filesize * Alpha * OffloadProb
        # self.QueueLength=  QueueLength
        
        
        ''' Computing the energy consumption '''
        Energy_edge  = 1e-5 * ComputationResource**2 
        Energy_prime = ( 1 - Outage) * Energy_edge
        Energy_UE    = 1e-5 * self.ResourceComputationUE**2 
        Energy       = Energy_prime * OffloadAction + (1 - OffloadAction) * Energy_UE
        
        
        ''' Forward State '''
        ForwardState = np.concatenate((PreferenceProb, QueueLength), axis=0)
        
        
        ''' Forward Reward: expected energy consumption '''
        ''' '''
        Expected_Energy =  - np.sum(Energy * PreferenceProb)  
        Expected_Outage =  - np.sum(Outage * PreferenceProb)
        ForwardReward   = Expected_Outage[None][:,None]
        
        CoupledVariables = np.concatenate((PreferenceProb, QueueLength), axis=0)
        
        return ForwardState, ForwardReward, CoupledVariables
        
    
        
    # %%
    def BackwardStep(self, BackwardState, Action, CoupledState):
        
        
        PreferenceProb = CoupledState[0:int(self.CoupledStateDim/2)]
        QueueLength    = CoupledState[int(self.CoupledStateDim/2):]
        
        ''' Action variables '''
        ComputationResource  = Action[0            : self.Nfile]      # [Kbits/slot]
        BufferCapacities     = Action[self.Nfile   : 2*self.Nfile]    # [Kbits]
        OffloadAction        = Action[2*self.Nfile : 3*self.Nfile]    # Pr( Offload | Prefer ) , [0,1]
        
        
        
        ''' Preference and Offloading Probability '''
        OffloadProb = OffloadAction * PreferenceProb
        
        ''' Outage probability '''
        Alpha = (BufferCapacities - QueueLength) / OffloadProb / self.N_UE_times_Filesize
        Alpha = (Alpha >= 1.0) * 1.0 + (Alpha < 1.0) * Alpha
        Outage  = 1 - Alpha
        
        
        ''' Computing the computation delay '''
        Delay_prime = Outage * (self.SlotDuration + BackwardState) + (1 - Outage) * ( QueueLength + 0.5 * Alpha * OffloadProb * self.N_UE_times_Filesize + self.FileSize )/(ComputationResource+1e-5) * self.SlotDuration
        ExperiencedDelay       = Delay_prime * OffloadAction + (1 - OffloadAction) * self.FileSize / self.ResourceComputationUE * self.SlotDuration
        
        ''' Forward State '''
        BackwardState_new =  ExperiencedDelay
        
        
        ''' Backward Reward: expected computation delay '''
        ''' '''
        Expected_delay =  - np.sum(ExperiencedDelay * PreferenceProb) /1e3
        BackwardReward = Expected_delay[None][:,None]
        
        return  BackwardState_new, BackwardReward
    

# %%
    def BackwardReset(self):
        self.ExperiencedDelay = np.zeros([self.Nfile])
        return self.ExperiencedDelay

    
    # %% 

    def ForwardReset(self):
        # np.random.seed(15)
        
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
        # HalfWidth_vec = np.random.randint(40, 200, size=[1, N_file])
        Pop = np.zeros([N_file,0])
        for i in range(TimeSteps):
            fn = 2 * P_vec/np.cosh((i-t0_vec[0])/HalfWidth_vec[0])
            fn = np.expand_dims(fn, 1)
            Pop= np.append(Pop, fn, 1)
        Pop = Pop/np.sum(Pop, 0)
        
        
        self.data_arr = Pop.T

        
        # self.Popularity = np.linspace(1, self.Nfile, self.Nfile)**(-self.Skewness)
        # self.Popularity = self.Popularity/np.sum(self.Popularity)
        # self.Popularity = self.Popularity[np.random.permutation(self.Nfile)]
        Popularity = self.data_arr[0,:]
        
        QueueLength = np.zeros([self.Nfile])
        
        State = np.concatenate((Popularity, QueueLength), axis=0)
        return State
        
    
    
