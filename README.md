# Odyssey

> *Odyssey*: A journey through robotic infrastructure; the ILIAD lab's stack for robotic demonstration collection,
> policy training, perception, and natural language.

Repository containing package source for Odyssey, the ILIAD lab's stack for real-world robotics, including perception and
natural language handling. Built with [Polymetis](https://facebookresearch.github.io/fairo/polymetis/),
[PyTorch](https://pytorch.org/), using [Anaconda](https://www.anaconda.com/) for python dependencies and sane quality
defaults (`black`, `isort`, `flake8`, `precommit`).

Note: This will eventually be rewritten as a PyPI-enabled package, with a `setup.py` to handle dependency management. The
source here is mostly for nightly development.

---

## Contributing

Before committing to the repository, *make sure to set up your dev environment and pre-commit (`pre-commit install`)!*
Here are the basic contribution guidelines:

+ Install and activate the Conda Environment using the `QUICKSTART` instructions below.

+ On installing new dependencies (via `pip` or `conda`), please make sure to update the `environment-<ID>.yaml` files
via the following command (note that you need to separately create the `environment-cpu.yaml` file by exporting from
your local development environment!):

  `make serialize-env --arch=<cpu | gpu>`

*More detailed instructions for intricate set up (e.g., simulators, experiment tooling, etc.) can be found in
[`CONTRIBUTING.md`](./CONTRIBUTING.md).*

---

## Quickstart

Clones `odyssey` to the working directory, then walks through dependency setup, mostly leveraging the
`environment-<arch>.yaml` files.

# Shared Environment (for Clusters w/ Centralized Conda)

Project-specific conda environments have already been setup for both the Stanford-NLP and ILIAD clusters, under the
name `odyssey`. The only necessary steps to take are cloning the repo, activating the appropriate
environment, and running `pre-commit install` to start developing (if you develop on the remote).

### Local Development - Linux w/ GPU & CUDA 11.3

Note: Assumes that `conda` (Miniconda, MiniForge, or Anaconda are all fine) is installed and on your path.

Ensure that you're using the appropriate `environment-<gpu | cpu>.yaml` file --> if PyTorch doesn't build properly for
your setup, checking the CUDA Toolkit is usually a good place to start. We have `environment-<gpu>.yaml` files for CUDA
11.3 (and any additional CUDA Toolkit support can be added -- file an issue if necessary).

```bash
git clone https://github.com/Stanford-ILIAD/odyssey
cd odyssey
conda env create -f environments/environment-gpu.yaml  # Choose CUDA Kernel based on Hardware - by default use 11.3!
conda activate odyssey
pre-commit install  # Important!
```

### Local Development - CPU (Mac OS & Linux)

Note: Assumes that `conda` (Miniconda, MiniForge or Anaconda are all fine) is installed and on your path. Use the `-cpu`
environment file.

```bash
git clone https://github.com/Stanford-ILIAD/odyssey
cd odyssey
conda env create -f environments/environment-cpu.yaml
conda activate odyssey
pre-commit install  # Important!
```

## Usage

This repository comes with sane defaults for `black`, `isort`, and `flake8` for formatting and linting. It additionally
defines a bare-bones Makefile (to be extended for your specific build/run needs) for formatting/checking, and dumping
updated versions of the dependencies (after installing new modules).

Other repository-specific usage notes should go here (e.g., training models, running a saved model, running a
visualization, etc.).

## Repository Structure

High-level overview of repository file-tree (expand on this as you build out your project). This is meant to be brief,
more detailed implementation/architectural notes should go in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

+ `environments` - Serialized Conda Environments for both CPU and GPU (CUDA 11.3). Other architectures/CUDA toolkit
environments can be added here as necessary.
+ `odyssey/` - Package Source - has all functionality for robot interfaces, demo collection, perception, etc.
    + `robot/` - Core robot interface implementation.
    + `demonstration/` - Useful utilities for demonstration collection.
+ `tests/` - Tests - please unit test (& integration test) your code when possible.
+ `Makefile` - Top-level Makefile (by default, supports `conda` serialization, and linting).
+ `.flake8` - Flake8 Configuration File (Sane Defaults).
+ `.pre-commit-config.yaml` - Pre-Commit Configuration File (Sane Defaults).
+ `pyproject.toml` - Black and isort Configuration File (Sane Defaults).
+ `ARCHITECTURE.md` - [WIP] Write up of repository architecture/design choices, how to extend and re-work for different
applications.
+ `CONTRIBUTING.md` - [WIP] Detailed instructions for contributing to the repository, in furtherance of the default
instructions above.
+ `README.md` - You are here!
+ `LICENSE` - By default, research code is made available under the GPLv3 License. Change as you see fit, but think
deeply about why!

---

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary typically since original
repository gets set up once, but I like to keep this in the README in case things break in the future).

Generally, if you're just trying to run/use this code, look at the Quickstart section above.

### GPU & Cluster Environments (CUDA 11.3)

```bash
conda create --name odyssey python=3.8
conda activate odyssey
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install ipython jupyter

pip install black flake8 gym isort matplotlib pre-commit wandb

# Install other dependencies via pip below -- conda dependencies should be added above (always conda before pip!)
...
```

### CPU Environments (Usually for Local Development -- Geared for Mac OS & Linux)

Similar to the above, but installs the CPU-only versions of Torch and similar dependencies.

```bash
conda create --name odyssey python=3.8
conda activate odyssey
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter

pip install black flake8 gym isort matplotlib pre-commit

# Install Polymetis dependencies...
pip install "git+https://github.com/siddk/polymetis-dev#egg=polymetis&subdirectory=polymetis"


# Install other dependencies via pip below -- conda dependencies should be added above (always conda before pip!)
...
```

### Containerized Setup

Support for running `odyssey` inside of a Docker or Singularity container is TBD. If this support is
urgently required, please file an issue.
