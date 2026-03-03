"""Data fetching utilities for PDB, FASTA, MSA, SMILES, and Docker images.

Modules
-------
pdb
    Download PDB structures from RCSB and FASTA sequences from UniProt.

msa
    Pre-compute MSA via Boltz + ColabFold API.

smiles
    Fetch calibration SMILES from HuggingFace or URLs.

docker
    Build the FREED++ Docker image and fetch fragment data.
"""
