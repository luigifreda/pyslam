
# Install pyslam under macOS 

The following procedure has been tested under *catalina 10.15.5* and *Xcode 11.5*. In order to make things running under macOS, I had to use some tricks (for matplotlib threads in particular, further details below). Please, consider that pyslam has been designed under Linux (Ubuntu 18.04), where you can get it in its 'best shape'. 

Please, follow these install steps: 

1. first you need to install homebrew (check the section below)
2. download this repo and move to the **experimental** `mac` branch: 
   ```
   $ git checkout mac 
   ```
3. launch the install script `install_all_mac_venv.sh`
4. in order to run `main_vo.py` run 
   ```
   $ ./launch_main_vo.py 
   ```
5. in order to run `main_slam.py` run 
   ```
   $ ./launch_main_slam.py 
   ```

**NOTE 1**: note that the above procedure will install a virtual python environment `pyslam` in your system. That virtual environment can be easily activated by using the command: 
```
$ . pyenv-create.sh 
```
(do not forget the dot! without '/' ! )
You can find further details about python virtual environments [here](./PYTHON-VIRTUAL-ENVS.md).

**NOTE 2**: the launch scripts above `launch_main_xxx.py ` will automatically activate the `pyslam` virtual enviroment for you and launch the scripts with the necessary extension (explained below):
```
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 main_xxx.py
```

## Notes about macOS 

### Install homebrew and wget 

from https://www.cyberciti.biz/faq/howto-install-wget-om-mac-os-x-mountain-lion-mavericks-snow-leopard/ 
```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
.. and install wget 
```
$ brew install wget
```

You may also need to install gcc by using XCode CLI. Run 
```
$ xcode-select --install
```


### Issues found with matlplotlib 

In order to avoid problems with multi-threading (see https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr), run 
```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  
```
Indeed, this solution does not work. You have to use the following one: 

from https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr#comments-52230415
indeed, what works for pangolin and mplot_thread is launching the main scripts with the variable set before 
```
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 xxx.py
```

I found other issues with matplotlib due to `plt.ion()` (interactive mode) not working on mac. In order to make the matplotlib threads working, I add to apply some tricks that make the matplot lib being refreshed in an inelegant way (being activated and refreshed in turn one over the other). But it works! :-)   
At the present time, `pyslam` is still experimental on macOS! 