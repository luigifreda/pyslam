
# Install pyslam under macOS 

<!-- TOC -->

- [Install pyslam under macOS](#install-pyslam-under-macos)
    - [1. Install steps](#1-install-steps)
    - [2. Notes about macOS](#2-notes-about-macos)
        - [2.1. Install homebrew](#21-install-homebrew)
    - [3. Known issues](#3-known-issues)
        - [3.1. First run](#31-first-run)
        - [3.2. Issues found with OpenCV and pyenv](#32-issues-found-with-opencv-and-pyenv)
        - [3.3. Issues found with boost serialization](#33-issues-found-with-boost-serialization)

<!-- /TOC -->


The following procedure has been tested under *Sequoia 15.1.1* and *Xcode 16.1*. 

## Install steps

Please, follow these install steps: 

1. first you need to install homebrew and Xcode CLT (check the [section below](#notes-about-macos))
2. download this repo: 
   ```bash
   $ git clone --recursive https://github.com/luigifreda/pyslam.git 
   $ cd pyslam 
   ```
3. change your default shell type to `bash`: 
   ```bash
   $ chsh -s /bin/bash 
   ```
   (if you want to set `zsh` back then run: `$ chsh -s /bin/zsh`)
4. launch the macOS install script
   ```bash
   $ ./install_all_venv.sh
   ```
5. in order to run `main_vo.py` run 
   ```bash
   $ . pyenv-activate.sh   # Activate pyslam environment. This is just needed once in a new terminal.
   ```
6. in order to run `main_slam.py` run 
   ```bash
   $ . pyenv-activate.sh   # Activate pyslam environment. This is just needed once in a new terminal. 
   ```

**NOTE 1**: the above procedure will install a virtual python environment `pyslam` in your system. That virtual environment can be easily activated by using the command: 
```bash
$ . pyenv-activate.sh 
```
(do not forget the dot! without '/' ! )
You can find further details about python virtual environments [here](./PYTHON-VIRTUAL-ENVS.md).

<!-- **NOTE 2**: the launch scripts `./scripts/launch_main_xxx.sh ` will automatically activate the `pyslam` virtual enviroment for you and launch the scripts with the necessary environment variable setting (explained below):
```bash
$ OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 main_xxx.py  # Deprecated: Not needed anymore. 
```

 **NOTE 3**: In order to make things running under macOS, I had to use some tricks (for matplotlib processes in particular, further details below). Please, consider that pyslam has been designed under Linux (Ubuntu 18.04), where you can get it in its 'best shape'.  -->

## Notes about macOS 

### Install homebrew

From https://brew.sh/, run the following command:
```bash
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

You may also need to install gcc by using XCode Command Line Tools. Run 
```bash
$ xcode-select --install
```

## Known issues


### First run

On my machine, I've noticed that, the first time one of main scripts is run, it takes a while before it actually starts.  

### Issues found with OpenCV and pyenv 

When you launch one of the scripts above, you get a warning: 
```bash
objc[6169]: Class CaptureDelegate is implemented in both /Users/luigi/.python/venvs/pyslam/lib/python3.7/site-packages/cv2/cv2.cpython-37m-darwin.so (0x11923d590) and /usr/local/opt/opencv/lib/libopencv_videoio.4.3.dylib (0x13021d0c8). One of the two will be used. Which one is undefined.
```
This is an **open issue** which needs to be solved. In a few words, this is an "interference" between the OpenCV libs of the installed virtual python environment and the OpenCV libs installed by homebrew.  


### Issues found with boost serialization 

On my mac, boost deserialization is very slow. On the other hand, under linux, it is very fast.

<!-- ### Issues found with dynamic matlplotlib 

**NEWS**: Under mac, the old classes `Mplot2d` and `Mplot3d` (based on `matplotlib`) are automatically replaced by `Qplot2d` and `Qplot3d` (based on `pyqtgraph`), which do not present the problems reported below. 

I found the following problems with python multi-processing (see https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr). The proposed solution to run this command in the open shell 
```
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  
```
does not work. In another thread https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr#comments-52230415, I found something that does work with both pangolin processes and mplot processes: launch the main scripts by setting the same environment variable one the same line 
```
$ OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 xxx.py
```

I found other issues with matplotlib due to `plt.ion()` (interactive mode) that does not work on mac. In order to make the matplotlib processes working, I had to apply some other tricks that make the matplot figures being refreshed in an inelegant way (being activated and refreshed in turn one over the other). But it works! :-)   
At the present time, `pyslam` is still experimental on macOS!  -->

