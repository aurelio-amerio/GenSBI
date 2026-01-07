import math


# ANSI Escape Codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_colored_value(val, thresholds=(1.1, 1.2)):
    """Returns the value wrapped in color codes based on thresholds.
    
    Parameters
    ----------
        val : float
            The value to color.
        thresholds : tuple of float
            Thresholds for coloring (red/yellow, yellow/green). Defaults to (1.1, 1.2).
            
    Returns
    -------
        str
            The colored string representation of the value.
    """
    if val < thresholds[0]:
        color = GREEN
    elif val < thresholds[1]:
        color = YELLOW
    else:
        color = RED
    return f"{color}{val:.4f}{RESET}"


def scale_lr(batch_size, base_lr=1e-4, reference_batch_size=256):
    """Scale learning rate based on batch size using square root scaling.

    Parameters
    ----------
        batch_size : int 
            The current batch size.
        base_lr : float
            The base learning rate for the reference batch size.
        reference_batch_size : int, optional
            The reference batch size. Defaults to 256.

    Returns
    -------
        float
            The adjusted learning rate.
    """
    return base_lr * math.sqrt(batch_size / reference_batch_size)
