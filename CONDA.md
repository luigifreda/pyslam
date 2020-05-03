# Test under Anaconda 

I've successfully tested `main_vo.py` under Anaconda (version 2019.10). Please, follow the instructions below.

In order to create a new conda environment `opencvenv`, activate it  and install OpenCV in it, run the following commands:  
```
$ conda create -yn opencvenv python=3.6
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

Now, you can run 
```
$ python3 -O main_vo.py
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

## TODO
Test the following script under Anaconda 
``` 
$ ./install_thirdparty.sh 
```