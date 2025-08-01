# Install under Anaconda or Miniconda

<!-- TOC -->

- [Install under Anaconda or Miniconda](#install-under-anaconda-or-miniconda)
    - [1. Installation](#1-installation)
    - [2. Usage](#2-usage)
    - [3. Create a pyslam conda environment](#3-create-a-pyslam-conda-environment)
    - [4. Activate the created pyslam conda environment](#4-activate-the-created-pyslam-conda-environment)
    - [5. Deactivate pyslam conda environment](#5-deactivate-pyslam-conda-environment)
    - [6. Delete pyslam conda environment](#6-delete-pyslam-conda-environment)
- [General Notes About Conda](#general-notes-about-conda)
    - [1. Install packages/env from file](#1-install-packagesenv-from-file)
    - [2. Deleting an environment](#2-deleting-an-environment)
    - [3. Creating an environment](#3-creating-an-environment)

<!-- /TOC -->

I successfully tested `pyslam` with [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (SHA256 hash *4b3b3b1b99215e85fd73fb2c2d7ebf318ac942a457072de62d885056556eb83e*) under Linux.   
Please, follow the instructions below. I assume you already installed Anaconda or Miniconda, and correctly initialized your conda python environment. 

## Installation 

In order to use `pyslam` under conda, check you have activated `conda` in your terminal. Get in the root of this repository and run the following command:
```bash
$ ./scripts/install_all_conda.sh  
```
This will compile the required thirdparty packages and will also create a `pyslam` conda environment.      
Please, discard any error like this: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed....`. See [TROUBLESHOOTING](./TROUBLESHOOTING.md) file for further details.

## Usage 

Now, from the same terminal, you can run: 
```bash
$ . scripts/pyenv-conda-activate.sh  # Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_vo.py               # test visual odometry
```
or
```bash
$ . scripts/pyenv-conda-activate.sh  # Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_slam.py             # test full SLAM
```
If you want to use a new terminal, you need to activate the `pyslam` environment as explained in this [section](#activate-the-created-pyslam-conda-environment).

## Create a `pyslam` conda environment 

You already see this above. In order to create a custom `pyslam` conda environment, get in the root of this repository and run the following command: 
```
$ . pyenv-conda-create.sh 
```

## Activate the created `pyslam` conda environment 

Run the following command (**N.B.**, do not forget the dot!): 
```
$ . scripts/pyenv-conda-activate.sh 
```
or 
```
$ conda activate pyslam 
```
Now, you can launch pySLAM scripts. 

## Deactivate `pyslam` conda environment 

To deactivate the `pyslam` environment, run
```
$ conda deactivate
```

## Delete `pyslam` conda environment 

To delete the `pyslam` environment, run
```
$ . scripts/pyenv-conda-delete.sh 
```


--- 
# General Notes About Conda

Below, you can find some useful details. The scripts mentioned above make the work for you. 

## Install packages/env from file 

You can generate a `requirements.txt` file by running: 
```
$ conda list -e > requirements-conda.txt
```
You can create and environment from such a file by runnning: 
```
$ conda create --name <env> --file requirements-conda.txt
```

**N.B.**: the file `requirements.txt` generated by conda cannot be used with pip3 (and viceversa)! 

Another approach is to use `.yml` files. In order to create a file `requirements-conda.yml` run:   
```
$ conda env export > requirements-conda.yml
```
or
```
$ conda env export --no-builds > requirements-conda-nobuilds.yml
```
for generating a requirements file without build numbers.    
You can create an environment from it by running: 
```
$ conda env create -f requirements.yml
```

## Deleting an environment 

To delete an environment, in your terminal window or an Anaconda Prompt, run:
```
$ conda remove --name myenv --all
```
this command will also return you some conda infos.  

You may instead use the simpler command:  
```
$ conda env remove --name myenv
```
To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run:
```
$ conda info --envs
```
The environments list that displays should not show the removed environment.

## Creating an environment 
In order to create a new conda environment `opencvenv`, activate it  and install OpenCV in it, run the following commands:  
```
$ conda create -yn opencvenv python=3.6.9
$ conda activate opencvenv
$ conda install -c menpo opencv3
```
This should install OpenCV 3.4.1 and everything you need to run SIFT and SURF. 

In order to install pytorch and torchvision: 
```
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
or (if you do not have an NVIDIA GPU)
```
$ conda install -c pytorch torchvision
```

To deactivate the `opencvenv` environment, use
```
$ conda deactivate
```
This command will bring you back to your default conda environment.

To re-activate the conda `opencvenv` environment, use
```
$ conda activate opencvenv
```
