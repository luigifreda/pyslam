# Install under pixi 

Here you can find the instructions to install `pyslam` under `pixi`.

<!-- TOC -->

- [Install under pixi](#install-under-pixi)
  - [Install pixi](#install-pixi)
  - [Activate pixi shell](#activate-pixi-shell)
  - [Launch the install script](#launch-the-install-script)

<!-- /TOC -->

Follow the steps described below: 
- [Install under pixi](#install-under-pixi)
  - [Install pixi](#install-pixi)
  - [Activate pixi shell](#activate-pixi-shell)
  - [Launch the install script](#launch-the-install-script)


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

<!-- ---
## Additional manual pip install steps

```
pip install tensorflow==2.13
pip install tensorflow_hub  # required by VPR
pip install tf_slim==1.1.0

pip install protobuf==3.20.3 # delf

pip install "numpy<2"
```  -->