# In the Name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:58:19 2021

@author: amidzam1
"""
import torch

class Buffer():
    def __init__(self):
        super().__init__()
                

    def discount_rewards(r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    
    
    
class Optimize_Alpha():
    def __init__(self):
        super().__init__()
        
        
    def Optimize_alphaMOO2(Gradients):
        Gradients = torch.tensor(Gradients)
        G1 = Gradients[0,:].unsqueeze(dim=1)
        G2 = Gradients[1,:].unsqueeze(dim=1)
        Alphas = torch.zeros(size=[2])
        
        Alpha = (G2 - G1).T @ G2 / (G1.T@G1 + G2.T@G2 - 2* G2.T @ G1 + 1e-20)
        Alpha = (Alpha<0)* 0 + (Alpha >=0 and Alpha<1)*Alpha + (Alpha >=1)*1
        Alphas[0] = Alpha
        Alphas[1] = 1 - Alpha
        return Alphas





    def Optimize_alphaMOO3(Gradients):

        J = torch.tensor(Gradients.T)       

        J0 = J
        Ones3 = torch.ones([3,1]).type(torch.double)
        Ones2 = torch.ones([2,1]).type(torch.double)

        alpha0 = torch.linalg.solve(J0.T@J0, Ones3)/(Ones3.T @ torch.linalg.solve(J0.T@J0 ,Ones3))
        flag = (alpha0>0).sum()
        Loss = alpha0.T @ J0.T@J0 @ alpha0

        J1 = J[:,0:2]
        alpha1 = torch.linalg.solve(J1.T@J1, Ones2)/(Ones2.T @ torch.linalg.solve(J1.T@J1 ,Ones2))
        Loss = torch.cat( [alpha1.T @ J1.T@J1 @ alpha1, Loss], 0)
        alpha1 = torch.cat([alpha1, torch.zeros([1,1])],0)
        flag = torch.stack([flag, (alpha1>=0).sum()],0)

        J2 = J[:,1:3]
        alpha2 = torch.linalg.solve(J2.T@J2, Ones2)/(Ones2.T @ torch.linalg.solve(J2.T@J2 ,Ones2))
        Loss = torch.cat([alpha2.T @ J2.T@J2 @ alpha2, Loss ], 0)
        alpha2 = torch.cat([torch.zeros([1,1]), alpha2],0)
        flag = torch.cat([flag, (alpha2>=0).sum().unsqueeze(0)],0)

        J3 = J[:,[0,2]]
        alpha3 = torch.linalg.solve(J3.T@J3, Ones2)/(Ones2.T @ torch.linalg.solve(J3.T@J3 ,Ones2))
        Loss = torch.cat([alpha3.T @ J3.T@J3 @ alpha3, Loss], 0 )
        alpha3 = torch.tensor([alpha3[0], 0.0, alpha3[-1]]).unsqueeze(1)
        flag = torch.cat([flag, (alpha3>=0).sum().unsqueeze(0)],0)

        alpha = torch.cat([alpha0, alpha1, alpha2, alpha3], 1)

        MxValue = torch.max(flag)
        if MxValue == 3:
            aux = Loss*(flag.unsqueeze(1)==3)
            aux[aux==0] = 1e100
            auxindx = torch.argmin( aux )
            Alpha = alpha[:,auxindx]
        else:
            auxVal = torch.argmin(torch.diag(J.T@J))
            Alpha = torch.zeros(3)
            Alpha[auxVal] = 1.0

        return Alpha