# Robustness Verification with $\alpha, \beta$-CROWN

This repository provides an end-to-end pipeline to train a simple CNN on the MNIST dataset and verify its robustness against adversarial perturbations using **$\alpha, \beta$-CROWN**.

## Project Structure
* `environment.yml` - Conda environment setup file.
* `Dockerfile` - Containerization support with NVIDIA CUDA integration.
* `test.py` - Main script that handles model training, weight saving, and triggers the verifier.
* `mnist_marabou_compare.yaml` - Configuration file for the bound propagation and BaB verification setup.
* `report.pdf` - Assignment report
---

## Setup & run: Containerized Installation (Docker)
To isolate dependencies and run seamlessly with GPU acceleration, use NVIDIA Container Toolkit.

1) Build the Docker image
```bash
docker build -t abcrown-mnist .
```
2) Run the container with GPU resources enabled:
```bash
docker run --gpus all -it --rm abcrown-mnist
```
