# Setting up Keras with Tensorflow backend using CUDA on Mac

- Homebrew install CUDA
    - `brew cask install cuda`
- [Register & download libCudnn](https://developer.nvidia.com/cudnn)
    - Copy files in cuda/lib into /usr/local/cuda/lib
    - Copy files in cuda/include into /usr/local/cuda/include
- Update .bash_profile
    - `export PATH=/usr/local/cuda/bin:$PATH`
    - `export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH`
- Install TensorFlow
    - pip install tensorflow-gpu
- Weird [bug](https://github.com/tensorflow/tensorflow/issues/3263) in CUDA install, fix:
    - `sudo ln -sf /usr/local/cuda/lib/libcuda.dylib /usr/local/cuda/lib/libcuda.1.dylib`
- Install Keras
    - pip install keras