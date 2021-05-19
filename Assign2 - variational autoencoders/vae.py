import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import os 
import numpy as np
import pandas as pd 

data = pd.read_csv('data.csv')

# print(data.shape)
# (1163, 9)

# converting to tensor
data = torch.tensor(data.values)
data = data.float()

N_EPOCHS = 10           
INPUT_DIM = 9           
HIDDEN_DIM = 7         
LATENT_DIM = 4           
lr = 1e-4               


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # non linear function of input
        hidden = F.relu(self.linear(x))
       
        # mean of latent variable distribution
        z_mu = self.mu(hidden)

        # variance of latent variable distribution
        z_var = self.var(hidden)

        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # non linear function of latent variable
        hidden = F.relu(self.linear(x))

        predicted = self.out(hidden)

        return predicted


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode the input, generate mean and variance of latent variable distribution
        z_mu, z_var = self.enc(x)

        # standard deviation
        std = torch.exp(z_var / 2)
        # small epsilon value sampled from N(0,I)
        eps = torch.randn_like(std)
        
        # sample from the distribution having latent parameters z_mu, z_var
        # using reparametrisation trick to simplify gradient descent
        x_sample = eps.mul(std).add_(z_mu)

        # decode the sample
        predicted = self.dec(x_sample)

        return predicted, z_mu, z_var
    
    def gen(self, x):
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # using reparametrisation trick to simplify gradient descent
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)

        # generate new_sample
        x_sample = eps.mul(std).add_(z_mu)
        return x_sample

# encoder
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# decoder
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

# vae
model = VAE(encoder, decoder)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i in range(list(data.size())[0]):
        x = data[i,:]

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss, squared loss
        recon_loss = torch.sum((x-x_sample)**2)
        # print(recon_loss)

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        # print(kl_loss)
        
        # total loss
        loss = recon_loss + kl_loss

        # backward propagation
        loss.backward()

        # adding total loss
        train_loss += loss.item()
        # print(train_loss)

        # update the weights of NN
        optimizer.step()

    return train_loss


# train for N_EPOCHS
for epoch in range(N_EPOCHS):
    train_loss = train()
    train_loss /= list(data.size())[0]
    print("%.2f"%(train_loss), end=' ')


new_X = np.zeros((data.shape[0], LATENT_DIM), dtype=float)
print(data[0,:].shape)
for i in range(list(data.size())[0]):
    x = data[i,:]
    x_sample = model.gen(x)
    x_sample_np = x_sample.detach().numpy()
    new_X[i] = x_sample_np

    
print(data.shape)
print(new_x.shape)
