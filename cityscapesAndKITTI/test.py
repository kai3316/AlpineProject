import numpy as np
import torch
output_branch1 = torch.ones([1,22,5,6])
result_branch1 = torch.sum(output_branch1, 1)[0,:,:]
print(result_branch1.shape)