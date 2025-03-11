# In the Name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:09:55 2022

"""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
from utils import Buffer, Optimize_Alpha
# torch.manual_seed(10)
# np.random.seed(10)

class Actor(nn.Module):
    def __init__(self, StateDim, DirichletActionDims, PositiveActionDim, CategoricalActionDim, CategoricalRange, NormalActionDim, BetaActionDim):
        super().__init__()
        self.StateDim =  StateDim
        self.HiddenDim = 100
        
        self.CategoricalRange = CategoricalRange
        self.PositiveActionDim = PositiveActionDim
        self.CategoricalActionDim = CategoricalActionDim
        self.NormalActionDim = NormalActionDim
        self.BetaActionDim = BetaActionDim
        
        
        self.Lay1 = nn.Linear(StateDim, self.HiddenDim)
        self.Lay2_DirichletActions = nn.ModuleList([ nn.Linear(self.HiddenDim, dim) for dim in DirichletActionDims ])
        
        self.Lay2_logNormalAction_mu = nn.Linear(self.HiddenDim, PositiveActionDim)
        self.Lay2_logNormalAction_sigma = nn.Linear(self.HiddenDim, PositiveActionDim)
        
        self.Lay2_NormalAction_mu = nn.Linear(self.HiddenDim, NormalActionDim)
        self.Lay2_NormalAction_sigma = nn.Linear(self.HiddenDim, NormalActionDim)
        
        self.Lay2_BetaAction_concent1 = nn.Linear(self.HiddenDim, BetaActionDim)
        self.Lay2_BetaAction_concent2 = nn.Linear(self.HiddenDim, BetaActionDim)
        
        self.Lay2_CategoricalAction = nn.Linear(self.HiddenDim, self.CategoricalRange)

            
    def init_weights_actor(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)



    def Forward(self, State):
        OutputHidden = torch.tanh(self.Lay1(State))
        
        # actions with Dirichlet distributions
        Policy_dists_dirichlet = []
        Mus_dirichlet = []
        for layer in self.Lay2_DirichletActions:
            Concentrate = F.softplus(layer(OutputHidden))
            policy_dist = D.Dirichlet(Concentrate)
            Mu = Concentrate / Concentrate.sum()
            Policy_dists_dirichlet.append(policy_dist)
            Mus_dirichlet.append(Mu)

        # Positive action with Lognormal distributions
        PolicyDist_logNormal = []
        Mu_logNormal = []
        if self.PositiveActionDim > 0:
            Mu_lognormal = F.softplus(self.Lay2_logNormalAction_mu(OutputHidden))
            Scale_lognormal = F.softplus(self.Lay2_logNormalAction_sigma(OutputHidden))
            PolicyDist_logNormal = D.LogNormal(Mu_lognormal, Mu_lognormal)
            Mu_logNormal = torch.exp(Mu_lognormal + Mu_lognormal ** 2 / 2)

        # action with Categorical  distribution
        PolicyDist_Categorical = []
        Mu_Categorical = []
        if self.CategoricalActionDim > 0:
            CategorParam = F.softplus(self.Lay2_CategoricalAction(OutputHidden))
            PolicyDist_Categorical = D.Categorical(CategorParam)
            Mu_Categorical = torch.argmax(CategorParam)[None]
        
        # action with Guassian distribution
        PolicyDist_Normal = []
        Mu_Normal = []
        if self.NormalActionDim > 0:
            Mu_normal = self.Lay2_NormalAction_mu(OutputHidden)
            Sigma_normal = F.softplus(self.Lay2_NormalAction_sigma(OutputHidden))
            PolicyDist_Normal = D.Normal(Mu_normal, Sigma_normal)
            Mu_Normal = Mu_normal
            
        # action with Beta distribution
        PolicyDist_Beta = []
        Mu_Beta = []
        if self.BetaActionDim > 0:
            Concent1_beta = F.softplus(self.Lay2_BetaAction_concent1(OutputHidden))
            Concent2_beta = F.softplus(self.Lay2_BetaAction_concent2(OutputHidden))
            PolicyDist_Beta = D.Beta(Concent1_beta, Concent2_beta)
            Mu_Beta = Concent1_beta/(Concent1_beta+Concent2_beta)

        return Policy_dists_dirichlet, PolicyDist_logNormal, PolicyDist_Categorical, PolicyDist_Normal, PolicyDist_Beta, \
               Mus_dirichlet,          Mu_logNormal,         Mu_Categorical,         Mu_Normal,         Mu_Beta

# %%
class BackwardCriticBase(torch.nn.Module):
    def __init__(self, StateDim, N_MCS, N_backward_rewards, LearningRate):
        super(BackwardCriticBase, self).__init__()
        self.StateDim = StateDim
        self.HiddenDim = 100
        self.N_MCS = N_MCS
        self.backwardcritics = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(StateDim, self.HiddenDim),
                torch.nn.Tanh(),
                torch.nn.Linear(self.HiddenDim, N_backward_rewards)  )
                for _ in range(N_MCS) 
                ])
        
        self.optimizers = [torch.optim.Adam(critic.parameters(), lr=LearningRate) for critic in self.backwardcritics]

    def init_weights_critic(self):
        for critic in self.backwardcritics:
            for m in critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
                    

    def forward(self, State):
        critic_outputs = [critic(State) for critic in self.backwardcritics]
        return critic_outputs
    
    
# %%
class ForwardCriticBase(torch.nn.Module):
    def __init__(self, StateDim, N_MCS, N_forward_rewards, LearningRate):
        super(ForwardCriticBase, self).__init__()
        self.StateDim = StateDim
        self.HiddenDim = 100
        self.N_MCS = N_MCS
        self.forwardcritics = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(StateDim, self.HiddenDim),
                torch.nn.Tanh(),
                torch.nn.Linear(self.HiddenDim, N_forward_rewards) )
                for _ in range(N_MCS)    
                ])
        
        self.optimizers = [torch.optim.Adam(critic.parameters(), lr=LearningRate) for critic in self.forwardcritics]

    def init_weights_critic(self):
        for critic in self.forwardcritics:
            for m in critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

    def forward(self, State):
        critic_outputs = [critic(State) for critic in self.forwardcritics]
        return critic_outputs
    
    
    
  # %%
class Agent(object):
    def __init__(self, Actor, ForwardCriticBase, BackwardCriticBase, ForwardCriticOptimizers, BackwardCriticOptimizers, MaxDirichletAction, N_MCS, DiscountFactor, SmoothingFactor, LearningRate):
        self.train_device = 'cpu'
        
        self.LearningRate = LearningRate
        
        self.Entropys_FW = []
        self.Log_Policies_FW = []
        self.ForwardRewards = []   
        
        self.RewardsQoS_FW = []        
        self.RewardsBandW_FW = []
        self.Rewards_BW = []
        
        self.StateValues_FW = [[] for _ in range(N_MCS)]
        self.NewStateValues_FW = [[] for _ in range(N_MCS)]
        self.StateValues_BW = [[] for _ in range(N_MCS)]
        self.NewStateValues_BW = [[] for _ in range(N_MCS)]


        self.gamma = DiscountFactor
        self.SmoothingFactor = SmoothingFactor
        self.betta = -.0

        self.Actor = Actor
        self.ActorOptimizer = torch.optim.Adam(Actor.parameters(), lr=LearningRate) #LearningRate

        # Initialize critics and optimizers in a loop
        self.ForwardCriticBase = ForwardCriticBase
        self.BackwardCriticOptimizers = BackwardCriticOptimizers
        self.ForwardCriticOptimizers = ForwardCriticOptimizers
        self.BackwardCriticBase = BackwardCriticBase
        
        self.MaxDirichletAction = MaxDirichletAction

        self.count = 0
        self.Gradient_Actor_mean = 0
        self.GradientBandW_Actor_mean = 0
        self.GradientQoS_Actor_mean = 0
        self.GradientBW_Actor_mean = 0



    def ForwardBackwardActorCriticUpdate(self, PreferenceCoeff, N_MCS, N_forward_rewards, N_backward_rewards, Optimizer):
        
        ''' Initialization '''
        PreferenceCoeff_FW = PreferenceCoeff[0:N_forward_rewards]   # Preference parameter of the forward rewards
        PreferenceCoeff_BW = PreferenceCoeff[-N_backward_rewards:]  # Preference parameter of the backward rewards
        
        # Initialization of logarithm of policy distribution and  the entropy of policy distribution
        Log_Policies_FW = torch.stack(self.Log_Policies_FW, dim=0).to(self.train_device).squeeze(-1)   
        Entropys_FW = torch.stack(self.Entropys_FW, dim=0).to(self.train_device).squeeze(-1)           
        self.Log_Policies_FW, self.Entropys_FW, = [], []


        # Initialization of forward rewards and forward statevalues
        ForwardRewards = torch.stack(self.ForwardRewards, dim=0).to(self.train_device).squeeze(-1)
        self.ForwardRewards = []
        
        StateValues_FW = [[] for _ in range(N_MCS)]
        NewStateValues_FW = [[] for _ in range(N_MCS)]
        StateValues_BW = [[] for _ in range(N_MCS)]
        NewStateValues_BW = [[] for _ in range(N_MCS)]

        
        for i in range(N_MCS):
            StateValues_FW[i] = torch.stack(self.StateValues_FW[i], dim=0).to(self.train_device)
            NewStateValues_FW[i] = torch.stack(self.NewStateValues_FW[i], dim=0).to(self.train_device)
            self.StateValues_FW[i] = []
            self.NewStateValues_FW[i] = []
        
                    
        # Initialization of backward rewards and backward statevalues
        BackwardRewards = torch.stack(self.Rewards_BW, dim=0).to(self.train_device).squeeze(-1)
        self.Rewards_BW = []
        
        for i in range(N_MCS):
            StateValues_BW[i] = torch.stack(self.StateValues_BW[i], dim=0).to(self.train_device)
            NewStateValues_BW[i] = torch.stack( self.NewStateValues_BW[i], dim=0).to(self.train_device)
            self.StateValues_BW[i] = []
            self.NewStateValues_BW[i] = []




        
        ''' To update forward-critics '''
        # To compute the forward advantage '''
        Advantages_FW, Loss_Critic_FW = [[] for _ in range(N_MCS)], [[] for _ in range(N_MCS)]
        
        for i in range(N_MCS):
            Advantages_FW[i] = (PreferenceCoeff_FW*ForwardRewards + self.gamma * NewStateValues_FW[i]).detach() -\
                StateValues_FW[i]
            Loss_Critic_FW[i] = torch.sum( Advantages_FW[i]**2, dim=0 )
        

        # Constituting the multi-objective loss of forward-critics'''
        for i in range(N_MCS):
            Gradients_FW_Critic = [[] for _ in range(N_forward_rewards)]
            for k in range(N_forward_rewards):
                self.ForwardCriticOptimizers[i].zero_grad()
                Loss_Critic_FW[i][k].backward(retain_graph=True)
                Gradient_FW_Critic = []
                for params in self.ForwardCriticBase.forwardcritics[i].parameters():
                    if (params.grad != None):
                        Gradient_FW_Critic = np.append(Gradient_FW_Critic, torch.flatten(params.grad))
        
                Gradients_FW_Critic[k] = Gradient_FW_Critic
         
            if N_forward_rewards ==1:
                OptimumAlphaMOO = 1.0
                
            elif N_forward_rewards==2:
                OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO2(np.stack(Gradients_FW_Critic))
                
            elif N_forward_rewards==3:
                OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO3(np.stack(Gradients_FW_Critic))
                
            else:
                raise NotImplementedError()
    
            Loss_Critic_FW_MOO = (OptimumAlphaMOO * Loss_Critic_FW[i] ).sum()
    
    
            # backpropagating the forward-critics loss '''
            self.ForwardCriticOptimizers[i].zero_grad()
            Loss_Critic_FW_MOO.backward()
            if Optimizer == "Adam":
                self.ForwardCriticOptimizers[i].step()
            elif Optimizer == "SGD":
                for (_,params) in enumerate(self.ForwardCriticBase.forwardcritics[i].parameters()):
                    if (params.grad != None):
                        params.data.copy_( params -self.LearningRate * params.grad )
                        
            # for param_group in self.ForwardCriticOptimizers[i].param_groups:
                # param_group['lr'] = 0.997*param_group['lr']



        
        ''' To update backward-critics '''
        # To compute the backward advantage '''
        Advantages_BW, Loss_Critic_BW = [[] for _ in range(N_MCS)], [[] for _ in range(N_MCS)]
        
        for i in range(N_MCS):
            Advantages_BW[i] = (PreferenceCoeff_BW*BackwardRewards + self.gamma * NewStateValues_BW[i]).detach() -\
                StateValues_BW[i]
            Loss_Critic_BW[i] = torch.sum( Advantages_BW[i]**2, dim=0 )
        

        # Constituting the multi-objective loss of backward-critics'''
        for i in range(N_MCS):
            Gradients_BW_Critic = [[] for _ in range(N_backward_rewards)]
            for k in range(N_backward_rewards):
                self.BackwardCriticOptimizers[i].zero_grad()
                Loss_Critic_BW[i][k].backward(retain_graph=True)
                Gradient_BW_Critic = []
                for params in self.BackwardCriticBase.backwardcritics[i].parameters():
                    if (params.grad != None):
                        Gradient_BW_Critic = np.append(Gradient_BW_Critic, torch.flatten(params.grad))
        
                Gradients_BW_Critic[k] = Gradient_BW_Critic
         
            if N_backward_rewards ==1:
                OptimumAlphaMOO = 1.0
                
            elif N_backward_rewards==2:
                OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO2(np.stack(Gradients_BW_Critic))
                
            elif N_backward_rewards==3:
                OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO3(np.stack(Gradients_BW_Critic))
                
            else:
                raise NotImplementedError()
    
            Loss_Critic_BW_MOO = (OptimumAlphaMOO * Loss_Critic_BW[i] ).sum()
    
    
            # backpropagating the backward-critics loss '''
            self.BackwardCriticOptimizers[i].zero_grad()
            Loss_Critic_BW_MOO.backward()
            if Optimizer == "Adam":
                self.BackwardCriticOptimizers[i].step()
            elif Optimizer == "SGD":
                for (_,params) in enumerate(self.BackwardCriticBase.backwardcritics[i].parameters()):
                    if (params.grad != None):
                            params.data.copy_( params -self.LearningRate * params.grad )
                        
            # for param_group in self.BackwardCriticOptimizers[i].param_groups:
                # param_group['lr'] = 0.997*param_group['lr']
            
            

        ''' To constitute the forward-backward loss of the actor'''
        Advantages_FW_avg = 0
        for i in range(N_MCS):
            Advantages_FW_avg += Advantages_FW[i]/N_MCS
        Loss_FW_Actor = -torch.mean( Advantages_FW_avg.detach() * Log_Policies_FW[:,None], dim=0 ) 
        Gradients_FW_Actor = [[] for _ in range(N_forward_rewards)]
        for k in range(N_forward_rewards):
            self.ActorOptimizer.zero_grad()
            Loss_FW_Actor[k].backward(retain_graph=True)
            Gradient_FW_Actor = []
            for params in self.Actor.parameters():
                if (params.grad != None):
                    Gradient_FW_Actor = np.append(Gradient_FW_Actor, torch.flatten(params.grad))
            Gradients_FW_Actor[k] = Gradient_FW_Actor
            

        Advantages_BW_avg = 0
        for i in range(N_MCS):
            Advantages_BW_avg += Advantages_BW[i]/N_MCS
        Loss_BW_Actor = -torch.mean( Advantages_BW_avg.detach() * torch.flip(Log_Policies_FW, dims=[0])[:,None], dim=0 )
        Gradients_BW_Actor = [[] for _ in range(N_backward_rewards)]
        for k in range(N_backward_rewards):
            self.ActorOptimizer.zero_grad()
            Loss_BW_Actor[k].backward(retain_graph=True)
            Gradient_BW_Actor = []
            for params in self.Actor.parameters():
                if (params.grad != None):
                    Gradient_BW_Actor = np.append(Gradient_BW_Actor, torch.flatten(params.grad))
            Gradients_BW_Actor[k] = Gradient_BW_Actor


        Loss_Actor = torch.cat((Loss_FW_Actor, Loss_BW_Actor), dim = 0)
        Gradients_Actor = np.stack(Gradients_FW_Actor + Gradients_BW_Actor)
        
        self.Gradient_Actor_mean = (1-self.SmoothingFactor) * self.Gradient_Actor_mean + self.SmoothingFactor * Gradients_Actor

        if N_backward_rewards+N_forward_rewards == 1:
            OptimumAlphaMOO = 1.0
            
        elif N_backward_rewards+N_forward_rewards==2:
            OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO2(self.Gradient_Actor_mean)
            
        elif N_backward_rewards+N_forward_rewards==3:
            OptimumAlphaMOO = Optimize_Alpha.Optimize_alphaMOO3(self.Gradient_Actor_mean)
            
        else:
            raise NotImplementedError()

        Loss_ActorMOO = (OptimumAlphaMOO * Loss_Actor).sum() + self.betta * torch.mean(Entropys_FW)




        ''' To update the Forward-Backward Actor using the common descent direction '''
        self.ActorOptimizer.zero_grad()
        Loss_ActorMOO.backward()
        if Optimizer == "Adam":
            self.ActorOptimizer.step()
        elif Optimizer == "SGD":
            for (_,params) in enumerate(self.Actor.parameters()):
                if (params.grad != None):
                    params.data.copy_( params -self.LearningRate * params.grad )

        # for param_group in self.ActorOptimizer.param_groups:
            # param_group['lr'] = 0.997*param_group['lr']
            
            
            

    def Get_Action(self, stateFW, Resampling_flag, evaluations=False):
        StateFW = torch.from_numpy(stateFW).float().to(self.train_device)
    
        PolicyDist_Dirichlet, PolicyDist_logNormal, PolicyDist_Categorical, PolicyDist_Normal, PolicyDist_Beta,\
        Mu_Dirichlet,         Mu_logNormal,         Mu_Categorical,         Mu_Normal,         Mu_Beta           = self.Actor.Forward(StateFW)
        
        isEmpty_logNormal = len(Mu_logNormal)   == 0
        isEmpty_Normal    = len(Mu_Normal)      == 0
        isEmpty_Categoric = len(Mu_Categorical) == 0
        isEmpty_Beta      = len(Mu_Beta)        == 0
        
        if evaluations:
            DirichletActions  = [self.MaxDirichletAction[i] * Mu_Dirichlet[i] for i in range(len(PolicyDist_Dirichlet))]
            PositiveAction    = Mu_logNormal
            CategoricalAction = Mu_Categorical
            NormalAction      = Mu_Normal
            BetaAction        = Mu_Beta
            
        else:
            if Resampling_flag:
                DirichletActions = [self.MaxDirichletAction[i] * PolicyDist_Dirichlet[i].rsample() for i in range(len(self.MaxDirichletAction)) ]
            else:
                DirichletActions = [self.MaxDirichletAction[i] * PolicyDist_Dirichlet[i].sample() for i in range(len(self.MaxDirichletAction)) ]
            
            PositiveAction = torch.zeros(size=[0])
            if not isEmpty_logNormal:
                if Resampling_flag:
                    PositiveAction = PolicyDist_logNormal.rsample()                    
                else:
                    PositiveAction = PolicyDist_logNormal.sample()
                
            CategoricalAction = torch.zeros(size=[0])
            if not isEmpty_Categoric:
                CategoricalAction = PolicyDist_Categorical.sample().unsqueeze(0)
                
            NormalAction = torch.zeros(size=[0])
            if not isEmpty_Normal:
                if Resampling_flag:
                    NormalAction = PolicyDist_Normal.rsample()
                else:
                    NormalAction = PolicyDist_Normal.sample()

            BetaAction = torch.zeros(size=[0])
            if not isEmpty_Beta:
                if Resampling_flag:
                    BetaAction = PolicyDist_Beta.rsample()
                else:
                    BetaAction = PolicyDist_Beta.sample()
    
        DirichletActions_ = torch.cat(DirichletActions, dim=0)
        Action = torch.cat([DirichletActions_, PositiveAction, CategoricalAction + 1, NormalAction, BetaAction], 0)
    
         
        Log_Policy = 0  # Initialize as zero
        Log_Policy += sum([PolicyDist_Dirichlet[i].log_prob(DirichletActions[i]/self.MaxDirichletAction[i]) for i in range(len(self.MaxDirichletAction)) ])
        
        if not isEmpty_logNormal:
            Log_Policy += PolicyDist_logNormal.log_prob(PositiveAction).sum()
            
        if not isEmpty_Categoric:
            Log_Policy += PolicyDist_Categorical.log_prob(CategoricalAction)[0]
            
        if not isEmpty_Normal:
            Log_Policy += PolicyDist_Normal.log_prob(NormalAction).sum()
            
        if not isEmpty_Beta:
            Log_Policy += PolicyDist_Beta.log_prob(BetaAction).sum()
    
    
        Entropy = 0  # Initialize as zero
        Entropy += sum([dist.entropy() for dist in PolicyDist_Dirichlet])
        
        if not isEmpty_logNormal:
            Entropy += PolicyDist_logNormal.entropy().sum()
            
        if not isEmpty_Categoric:
            Entropy += PolicyDist_Categorical.entropy()
            
        if not isEmpty_Normal:
            Entropy += PolicyDist_Normal.entropy()
            
        if not isEmpty_Beta:
            Entropy += PolicyDist_Beta.entropy().sum()
    
    
        return Action, Log_Policy, Entropy
    

 

    def Get_ForwardStateValue(self, stateFW, N_MCS):
        StateFW = torch.from_numpy(stateFW).float().to(self.train_device)
        StateValue_FW   =  self.ForwardCriticBase.forward(StateFW)
        
        return StateValue_FW



    def Get_BackwardStateValue(self, stateBW, N_MCS):
        StateBW = torch.from_numpy(stateBW).float().to(self.train_device)
        StateValueBW  =  self.BackwardCriticBase.forward(StateBW)
            
        return StateValueBW



    def AccumulateFW(self, ForwardReward, Log_Policy, StateValue_FW,  NewStateValue_FW, Entropy, N_MCS):
        self.ForwardRewards.append(torch.Tensor(ForwardReward))
        self.Log_Policies_FW.append(Log_Policy)
        
        for i in range(N_MCS):
            self.StateValues_FW[i].append(StateValue_FW[i])
            self.NewStateValues_FW[i].append(NewStateValue_FW[i])
        
        self.Entropys_FW.append(Entropy)



    def AccumulateBW(self, Reward, StateValue, NewStateValue, N_MCS):
        self.Rewards_BW.append(torch.Tensor(Reward))
        
        for i in range(N_MCS):
            self.StateValues_BW[i].append(StateValue[i])
            self.NewStateValues_BW[i].append(NewStateValue[i])
