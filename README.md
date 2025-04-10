# Benchmark study on peak shaving and valley filling
The goal of this project is to provide a benchmark study on different implementations about using electric vehicles for peak shaving and valley filling in the low voltage grid.

## Getting started
# Setup of Conda Environment
As a prerequisites, we assume to have a Linux distribution as operating system
1. Download a [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) version to be installed on your machine. 
2. Create a new Python enviroment named `benchmark-v2g`
```bash
conda create -n benchmark-v2g
```
3. To be sure that the correct environment is active:
```bash
conda activate benchmark-v2g
```
4. Then install `pip`
```bash
conda install pip
```

# Install Dependencies
Now we can install some required project dependencies, which are defined in the `requirement.txt`.
```bash
# Make sure your benchmark-v2g python enviroment is active!
cd <project-root>
pip install -r requirements.txt
```
