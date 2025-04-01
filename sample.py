from typing import Dict, Tuple
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sampling_config import get_config
from utils import get_alpha_bar_prev

def get_timesteps(training_timesteps: int, num_steps: int) -> Tuple[Tensor, Tensor]:
    """
    Generate timesteps for the diffusion process.

    Args:
        training_timesteps (int): Total number of training timesteps.
        num_steps (int): Number of inference steps.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the timesteps and previous timesteps.
    """
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    timesteps = torch.linspace(0, training_timesteps - 1, num_steps, device=config.device, dtype=torch.long)
    timesteps_prev = torch.cat([timesteps[1:], timesteps.new_zeros(1)], dim=0)
    return timesteps, timesteps_prev
    ### END CODE HERE ###

def predict_x0(
    predicted_noise: torch.Tensor,
    t: int,
    sample_t: torch.Tensor,
    scheduler_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict the original image from the noisy sample.

    Args:
        predicted_noise (torch.Tensor): The predicted noise tensor.
        t (int): Current timestep.
        sample_t (torch.Tensor): The noisy sample tensor.
        scheduler_params (Dict[str, torch.Tensor]): Scheduler parameters.

    Returns:
        torch.Tensor: The predicted original image tensor.
    """
    ### START CODE HERE ###
    alpha_bar_t = scheduler_params['alphas_bar'][t]
    predicted_x0 = (sample_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    return predicted_x0
    ### END CODE HERE ###

def compute_forward_posterior_mean(
    predicted_x0: Tensor,
    noisy_image: Tensor,
    scheduler_params: Dict[str, Tensor],
    t: int,
    t_prev: int,
) -> Tensor:
    """Compute the mean of the forward posterior distribution.

    Args:
        predicted_x0 (Tensor): The predicted original image tensor.
        noisy_image (Tensor): The noisy image tensor.
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current timestep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed mean of the forward posterior distribution.
    """
    ### START CODE HERE ###
    alpha_bar_t = scheduler_params['alphas_bar'][t]
    alpha_bar_t_prev = scheduler_params['alphas_bar'][t_prev]
    beta_t = 1 - alpha_bar_t / alpha_bar_t_prev
    mean = torch.sqrt(alpha_bar_t_prev) * predicted_x0 + torch.sqrt(1 - alpha_bar_t_prev) * noisy_image / torch.sqrt(1 - alpha_bar_t)
    return mean
    ### END CODE HERE ###

def compute_forward_posterior_variance(
    scheduler_params: Dict[str, Tensor], t: int, t_prev: int
) -> Tensor:
    """Compute the variance of the forward posterior distribution.

    Args:
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current tim estep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed variance of the forward posterior distribution.
    """
    ### START CODE HERE ###
    alpha_bar_t = scheduler_params['alphas_bar'][t]
    alpha_bar_t_prev = scheduler_params['alphas_bar'][t_prev]
    beta_t = 1 - alpha_bar_t / alpha_bar_t_prev
    variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    return variance
    ### END CODE HERE ###

def get_stochasticity_std(
    eta: float, t: int, t_prev: int, alphas_bar: torch.Tensor
) -> torch.Tensor:
    """Calculate the stochasticity standard deviation for DDIM sampling.

    Args:
        eta (float): The DDIM stochasticity parameter (0 = deterministic, 1 = full stochastic).
        t (int): Current timestep.
        t_prev (int): Previous timestep.
        alphas_bar (torch.Tensor): Cumulative product of (1 - beta).

    Returns:
        Tensor: The computed standard deviation.
    """
    ### START CODE HERE ###
    alpha_bar_t = alphas_bar[t]
    alpha_bar_t_prev = alphas_bar[t_prev]
    std = torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev) * eta**2)
    return std
    ### END CODE HERE ###

def predict_sample_direction(
    alphas_bars_prev: float, predicted_noise: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Predict the direction for the next sample in DDIM.

    Args:
        alphas_bars_prev (float): Alpha bar value for previous timestep.
        predicted_noise (torch.Tensor): Predicted noise from the model.
        std (torch.Tensor): Standard deviation for the step.

    Returns:
        Tensor: The predicted sample direction.
    """
    ### START CODE HERE ###
    direction = torch.sqrt(alphas_bars_prev) * predicted_noise + std * torch.randn_like(predicted_noise)
    return direction
    ### END CODE HERE ###


def stochasticity_term(std: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Compute the stochasticity term for DDIM sampling.

    Args:
        std (torch.Tensor): The computed standard deviation.
        noise (torch.Tensor): Random noise tensor.

    Returns:
        Tensor: The stochasticity term to be added to the sample.
    """
    ### START CODE HERE ###
    term = std * noise
    return term
    ### END CODE HERE ###
