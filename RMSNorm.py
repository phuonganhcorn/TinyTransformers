import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    '''
    https://arxiv.org/abs/1910.07467
    https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964
    '''
    def __init__(self, n_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_dim))

    ''' Inspired by LLaMA '''
    def _norm(self, x):
        # x × {1 / [sqrt(mean(x^2)+ϵ)]}
        result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return result
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        final_output = output * self.weight
        return final_output