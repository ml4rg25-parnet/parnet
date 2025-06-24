# PanRBPNet: A RBP-binding-informed RNA Foundation Model

PanRBPNet - or `parnet` - is a multi-task extension of our previous [RBPNet](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03015-7) model for prediction RNA-protein binding at nucleotide resolution.

> [!WARNING]
> This package is under heavy development and while it's API is somewhat stable, it can change at any time and without warning. If you are using this package in your own research, it's highly recommended to either 1) fork the respository or 2) pin the version / commit of the package you are using.

## Installation

```bash
git clone https://github.com/ml4rg25-parnet/parnet.git
cd parnet
```

### Using pixi

```bash
# For GPU support
pixi install --environment parnet-gpu
# For CPU only
pixi install --environment parnet
```

For development, you have the `-dev` suffixed environments: `parnet-gpu-dev` or `parnet-dev`.

You can then activate the desired environment with:

```bash
# Example here with GPU support
pixi shell --environment parnet-gpu
```

### Using Conda

NOTE: conda YAML files are generated using `pixi-pack` ; see the [Updating the environment](#updating-the-environment) section below.

```
git clone https://github.com/ml4rg25-parnet/parnet.git
cd parnet
conda env create -f envs/pixi-pack.parnet.linux-64.yml -n parnet
```

## Updating the environment

> [!WARNING]
> We stronly advise using the `pixi` package manager to manage the environment.
> If you are using `conda` to create the environment and update it, please consider propagating the changes to the
> `pixi.toml` and `pyproject.toml`.

Updates to the `pixi.toml` can either be done manually or using the `pixi` commands (e.g. `pixi add <package>`).
After any modification, proceed through the following steps (hereafter with the example of the `parnet` environment):


1. manually update the `pyproject.toml` file with the newly added, version-resolved package.
2. run the `pixi-pack` command to produce a tarball of the environment.

  ```bash
  # Here with the example of the CPU-only environment for linux-64 ; you can generate multiple tarballs for different platforms.
  pixi-pack pack --environment parnet --platform linux-64 --output-file envs/pixi-pack.parnet.linux-64.tar.gz ./pixi.toml
  ```

3. Untar the archive:

  ```bash
  cd envs/ && mkdir pixi-pack.parnet.linux-64/ && tar -xzf pixi-pack.parnet.linux-64.tar.gz -C pixi-pack.parnet.linux-64/
  ```

4. Copy the `environment.yml` file from the unpacked directory into the `envs/` directory: `cp envs/pixi-pack.parnet.linux-64/environment.yml envs/pixi-pack.parnet.linux-64.conda.yml`
5. Edit the copy `envs/pixi-pack.parnet.linux-64.conda.yml` to apply the following changes:
    - replace the channels section with the channels list (available with `pixi config list`)
    - under the `- pip` section, remove the two lines `- --no-index` and `- --find-links ./pypi`

This resulting yaml file can then be used to create a new conda environment with:

```bash
conda env create -f envs/pixi-pack.parnet.linux-64.conda.yml -n parnet
```


> [!WARNING]
> Again: please consider using `pixi` to manage the environment.


If updating your environment through `conda`, you can export the currently activated environment to a lock file with:

```bash
conda env export --from-history --no-builds -n $(ENV_NAME) -f environment.lock.yml
```


## Dataset

Parnet's training datasets are stored in Huggingface's dataset format (HFDS). The primary training dataset of this study, which is composed of 223 eCLIP tracks from the ENCODE project, can be obtained via:

### Download compressed HFDS dataset
```
wget https://zenodo.org/records/14176118/files/encode.filtered.5.hfds.tar.gz
```

### Unpack HFDS dataset
```
tar -xJf encode.hfds.tar.xz
```

## Usage

```
Usage: parnet [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build-dataset
  predict
  train
```
