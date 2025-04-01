from typing import Dict
import torch
from submission.score_matching_utils import (
    create_log_p_theta,
    compute_score_function,
    compute_trace_jacobian,
    compute_frobenius_norm_squared,
    add_noise,
    compute_score
)

# Objective Function for Denoising Score Matching
def denoising_score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor], noise_std: float = 0.1
) -> torch.Tensor:
    """Objective function for denoising score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.
        noise_std (float): Standard deviation of the noise to add.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    noisy_x = add_noise(x, noise_std)
    score_noisy = compute_score(noisy_x, mean, log_var)
    score_clean = compute_score(x, mean, log_var)
    dsm_objective = torch.mean((score_noisy - score_clean) ** 2)
    
    return dsm_objective
    ### END CODE HERE ###


# Objective Function for Score Matching
def score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Objective function for score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    log_p_theta = create_log_p_theta(x, mean, log_var)
    score_function = compute_score_function(log_p_theta, x)
    frobenius_norm_squared = compute_frobenius_norm_squared(score_function)
    trace_jacobian = compute_trace_jacobian(score_function)
    
    objective = frobenius_norm_squared + 2 * trace_jacobian
    return objective
    
    ### END CODE HERE ###