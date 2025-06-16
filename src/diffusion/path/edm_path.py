from abc import ABC, abstractmethod
import jax 
from jax import Array
from jax import numpy as jnp


from diffusion.path.path import ProbPath

class EDMPath(ProbPath):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        assert self.scheduler.name in ["EDM", "EDM-VP", "EDM-VE"] , f"Scheduler must be one of ['EDM', 'EDM-VP', 'EDM-VE'], got {self.scheduler.name}."
        return