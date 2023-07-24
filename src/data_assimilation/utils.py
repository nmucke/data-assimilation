
import os
import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder
from latent_time_stepping.AE_models.autoencoder import Autoencoder
from latent_time_stepping.AE_models.encoder_decoder import Decoder, Encoder
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder
from latent_time_stepping.time_stepping_models.time_stepping_model import TimeSteppingModel


def create_directory(directory):
    """Creates a directory if it doesn't exist"""

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)


def load_trained_AE_model(model_load_path, model_type, device):

    state_dict = torch.load(f'{model_load_path}/model.pt', map_location=device)

    with open(f'{model_load_path}/config.yml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if model_type == "VAE":
        encoder = VAEEncoder(**config['model_args']['encoder'])
    elif model_type == "WAE":
        encoder = Encoder(**config['model_args']['encoder'])
    elif model_type == "AE":
        encoder = Encoder(**config['model_args']['encoder'])

    decoder = Decoder(**config['model_args']['decoder'])

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(device)
    model.load_state_dict(state_dict['model_state_dict'])

    return model

def load_trained_time_stepping_model(
    model_load_path,
    model_type,
    device,
):

    state_dict = torch.load(f'{model_load_path}/model.pt', map_location=device)

    with open(f'{model_load_path}/config.yml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    pars_encoder = ParameterEncoder(**config['model_args']['parameter_encoder_args'])
    model = TimeSteppingModel(
        pars_encoder=pars_encoder,
        **config['model_args']['time_stepping_decoder'],
    )
    
    model = model.to(device)
    model.load_state_dict(state_dict['model_state_dict'])

    return model
