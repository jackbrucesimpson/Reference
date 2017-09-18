# CUDA, cuDNN, TensorFlow, and Keras Setup on Ubuntu 16.04

## Install CUDA
Go to the [CUDA download](https://developer.nvidia.com/cuda-downloads) page and download the local .deb file.

Install the file with the following command (you may have to change the filename).  

```
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

Add the following lines to your `.profile` file in your home directory and then reload it with `source .profile`

```
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Install cuDNN

Download cuDNN 5.1 (Tensorflow has issues with version 6+), and once you've extracted the file, navigate into the extracted directory and run the following commands

```
sudo cp -P include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P lib64/libcudnn* /usr/local/cuda-8.0/lib64
```

## Install TensorFlow and Keras

```
pip install tensorflow-gpu keras
```