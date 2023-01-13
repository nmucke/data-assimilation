import hamiltorch
import torch
import matplotlib.pyplot as plt

def log_pdf(x):

    H = 0.5 * x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0]**2*x[2]**2

    return -H

params_init = torch.zeros(4, requires_grad=True)


num_samples = 25000
step_size = .3
num_steps_per_sample = 5

sampler = hamiltorch.Sampler.HMC_NUTS
integrator = hamiltorch.Integrator.IMPLICIT

hamiltorch.set_random_seed(123)
samples = hamiltorch.sample(
    log_prob_func=log_pdf, 
    burn=10000,
    params_init=params_init,  
    num_samples=num_samples, 
    step_size=step_size, 
    num_steps_per_sample=num_steps_per_sample,
    integrator=integrator,
    sampler=sampler,
    )

samples = torch.stack(samples)
samples = samples.detach().numpy()

plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.hist(samples[:, 0], bins=50)
plt.subplot(1, 4, 2)
plt.hist(samples[:, 1], bins=50)
plt.subplot(1, 4, 3)
plt.hist(samples[:, 2], bins=50)
plt.subplot(1, 4, 4)
plt.hist(samples[:, 3], bins=50)
plt.show()

