
import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY
    

class SVGD:
    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optim = optimizer

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)


        return phi

    def step(self, X):
        self.optim.zero_grad()
        X.grad = -self.phi(X)
        self.optim.step()


class MoG():#torch.distributions.Distribution):
    def __init__(self, loc, covariance_matrix):
        self.num_components = loc.size(0)
        self.loc = loc
        self.covariance_matrix = covariance_matrix

        self.dists = [
            torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
            for mu, sigma in zip(loc, covariance_matrix)
        ]

        #super(MoG, self).__init__(torch.Size([]), torch.Size([loc.size(-1)]))

    @property
    def arg_constraints(self):
        return self.dists[0].arg_constraints

    @property
    def support(self):
        return self.dists[0].support

    @property
    def has_rsample(self):
        return False

    def precision_matrix(self):
        pass

    def log_prob(self, value):
        return torch.cat(
            [p.log_prob(value).unsqueeze(-1) for p in self.dists], dim=-1).logsumexp(dim=-1)

    def enumerate_support(self):
        return self.dists[0].enumerate_support()
  

class MoG2(MoG):
    def __init__(self, device=None):
        loc = torch.Tensor([[-5.0, 0.0], [5.0, 0.0]]).to(device)
        cov = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1).to(device)

        super(MoG2, self).__init__(loc, cov)
    

def main():
  

    # Let us initialize a reusable instance right away.
    K = RBF()

    mog2 = MoG2(device=device)
        
    n = 100
    X_init = (5 * torch.randn(n, 2)).to(device)
        

    X = X_init.clone()
    svgd = SVGD(mog2, K, optim.Adam([X], lr=1e-1))
    for _ in range(1000):
        svgd.step(X)

        print(svgd.phi(X).mean())



if __name__ == '__main__':
    main()