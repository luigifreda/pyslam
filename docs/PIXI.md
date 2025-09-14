# Install under pixi 

Here you can find the instructions to install `pyslam` under `pixi` and a very concise `pixi` primer.

<!-- TOC -->

- [Install under pixi](#install-under-pixi)
  - [Install `pyslam` with pixi](#install-pyslam-with-pixi)
    - [Install pixi](#install-pixi)
    - [Activate pixi shell](#activate-pixi-shell)
    - [Launch the `pyslam` install script](#launch-the-pyslam-install-script)
    - [Launch a main `pyslam` script](#launch-a-main-pyslam-script)
  - [Pixi primer](#pixi-primer)
    - [Initialize a Project](#initialize-a-project)
    - [Add dependancies](#add-dependancies)
    - [Run Commands in Environment](#run-commands-in-environment)
    - [Enter an interactive shell](#enter-an-interactive-shell)
    - [Add specific platform targets](#add-specific-platform-targets)
    - [Remove the local pixi environment](#remove-the-local-pixi-environment)
    - [Table of commands](#table-of-commands)

<!-- /TOC -->

---


## Install `pyslam` with pixi

Follow the steps reported below: 

### Install pixi 
```
curl -fsSL https://pixi.sh/install.sh | sh
```

Reference: https://pixi.sh/latest/#installation


### Activate pixi shell 

From the root folder of this repository, run
```
pixi shell 
```

### Launch the `pyslam` install script 
Then run
```
./scripts/install_all_pixi.sh
```

### Launch a main `pyslam` script

Once you have activate the pixi shell in your terminal, you're ready to run any main script.


--- 

## Pixi primer

###  Initialize a Project

From within your project folder:
```bash
pixi init
```
This creates a pixi.toml (like pyproject.toml) describing your environment.


### Add dependancies 

Conda packages (from conda-forge):
```bash
pixi add numpy scipy matplotlib
```

Pip packages:
```bash
pixi add pip
pixi run pip install opencv-python
```
Then manually edit pixi.toml to reflect pip-installed packages:

```toml
[tool.pixi.pip-dependencies]
opencv-python = "*"
```

### Run Commands in Environment


To test the environement: 
```bash
pixi run python -c "import numpy; print(numpy.__version__)"
pixi run python -c "import cv2; print(cv2.__version__)" 
``` 

Other examples: 
```bash
pixi run python script.py
pixi run jupyter notebook
```

### Enter an interactive shell 

```bash
pixi shell
```
This opens a shell with pixi enviornment variable already set and you don't need anymore to use `pixi run ...<python command>`.


### Add specific platform targets 

Add specific platform targets (e.g., cross-platform builds):
```toml
[tool.pixi]
platforms = ["linux-64", "osx-arm64", "win-64"]
```

### Remove the local pixi environment 

To delete the environment associated with the current project (i.e., clean slate):
```bash
pixi clean --all
``` 

This removes:
- The current environment’s packages
- The build cache
- Any temporary/lock artifacts

You can then rebuild it cleanly with:
```bash
pixi install
```

### Table of commands

This is a recap table:

| Command                     | Description                           |
| --------------------------- | ------------------------------------- |
| `pixi init`                 | Initialize a new project              |
| `pixi add <pkg>`            | Add a dependency                      |
| `pixi install`              | Re-resolve and install dependencies   |
| `pixi run <cmd>`            | Run a command in the environment      |
| `pixi shell`                | Enter an interactive shell            |
| `pixi remove <pkg>`         | Remove a package                      |
| `pixi list`                 | List installed packages               |
| `pixi export --format toml` | Export environment definition as TOML |
| `pixi update`               | Update dependencies and re-lock       |
| `pixi clean --all`          | Remove the local pixi environment     |   


