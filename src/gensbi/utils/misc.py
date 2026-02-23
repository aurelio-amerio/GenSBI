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

