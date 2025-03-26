import torch
import numpy as np

# Standard loss metric (e.g., MSE)
def mse_loss(output, target):
    return torch.nn.functional.mse_loss(output, target).item()

# Mean Absolute Error
def mae(output, target):
    return torch.mean(torch.abs(output - target)).item()

# Root Mean Squared Error
def rmse(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2)).item()

# R² score (coefficient of determination)
def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    return 1 - (ss_res / ss_tot).item()

# Dictionary of available metrics
metric_functions = {
    'mse': mse_loss,
    'mae': mae,
    'rmse': rmse,
    'r2': r2_score
}
