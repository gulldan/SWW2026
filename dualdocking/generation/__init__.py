"""FREED++ Docker-based molecule generation.

Wraps the FREED++ RL generator (Python 3.7 / PyTorch 1.12 / DGL 0.9) in a
Docker container, since its dependencies are incompatible with the project
environment (Python >= 3.11). UniDock GPU docking provides the RL reward.

Modules
-------
freedpp
    Docker subprocess wrapper for FREED++ training and molecule extraction.

    - ``GenerationConfig``       -- dataclass with all generation parameters
    - ``generate()``             -- run FREED++ in Docker, return generated SMILES
"""
