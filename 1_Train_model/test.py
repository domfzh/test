from numpy import mod
import torch
import torch.nn.functional as F

t4d=torch.empty(1,3,5,3)
print(t4d.shape)
t1=F.pad(t4d,(0,1,0,1),mode="reflect")
print(t1.shape)