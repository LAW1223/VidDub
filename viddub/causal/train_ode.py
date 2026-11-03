"""Distill a few-step ODE VidDub student from frozen trajectories."""
from .training.engine import main


if __name__ == "__main__":
    main("ode")
