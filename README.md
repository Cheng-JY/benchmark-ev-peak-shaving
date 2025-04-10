# Benchmark study on peak shaving and valley filling
The goal of this project is to provide a benchmark study on different implementations about using electric vehicles for peak shaving and valley filling in the low voltage grid.

## Getting started
## Setup of Conda Environment
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

## Install Dependencies
Now we can install some required project dependencies, which are defined in the `requirement.txt`.
```bash
# Make sure your benchmark-v2g python enviroment is active!
cd <project-root>
pip install -r requirements.txt
```

## License
MIT License

Copyright (c) [2025] [Benchmark-v2g]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
