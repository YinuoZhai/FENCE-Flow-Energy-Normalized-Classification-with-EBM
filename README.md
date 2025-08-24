# FENCE-Flow-Energy-Normalized-Classification-with-EBM

## Introduction
We propose a sample/feature-level discriminator that combines **energy + energy gradient**, and use **Flow Matching** for latent space preheating, combined with DSE/NCE, to improve the robustness-accuracy trade-off. The repository contains a complete implementation on CIFAR-10, a routing calibrator, and visualization tools.

## Running environment
```bash
python= 3.8.20 
pytorch= 2.4.1
cuda = 12.6
numpy= 1.24.4
```
 
## Quick start
```bash
git clone https://github.com/YinuoZhai/FENCE-Flow-Energy-Normalized-Classification-with-EBM.git
cd FENCE-Flow-Energy-Normalized-Classification-with-EBM
pip install -e .
python -m ebm_at.main
```




