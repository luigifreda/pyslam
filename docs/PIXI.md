# Install under pixi 

Here you can find the instructions to install `pyslam` under `pixi`.

<!-- TOC -->

- [Install under pixi](#install-under-pixi)
    - [1. Install pixi](#1-install-pixi)
    - [2. Activate pixi shell](#2-activate-pixi-shell)
    - [3. Launch the install script](#3-launch-the-install-script)
    - [4. Launch a main script](#4-launch-a-main-script)

<!-- /TOC -->

Follow the steps described below: 
- [Install under pixi](#install-under-pixi)
  - [Install pixi](#install-pixi)
  - [Activate pixi shell](#activate-pixi-shell)
  - [Launch the install script](#launch-the-install-script)
  - [Launch a main script](#launch-a-main-script)


## Install pixi 
```
curl -fsSL https://pixi.sh/install.sh | sh
```

Reference: https://pixi.sh/latest/#installation


## Activate pixi shell 

From the root folder of this repository, run
```
pixi shell 
```

## Launch the install script 
Then run
```
./scripts/install_all_pixi.sh
```

## Launch a main script

Once you have activate the pixi shell in your terminal, you're ready to run any main script.

<!-- ---
## Additional manual pip install steps

```
pip install tensorflow==2.13
pip install tensorflow_hub  # required by VPR
pip install tf_slim==1.1.0

pip install protobuf==3.20.3 # delf

pip install "numpy<2"
```  -->