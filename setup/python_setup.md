# Python Setup

## Update all Python packages

`pip3 freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip3 install -U`

## Install scientific libraries

```
pip3 install numpy matplotlib scipy sklearn pandas keras h5py cython
pip3 install tensorflow
```
## Install Jupyter notebook

```
pip3 install  jupyter jupyterlab

python2 -m pip install ipykernel
python2 -m ipykernel install --user
```

## Version information

`pip3 install version_information`

## Linting

`pip3 install flake8 flake8-docstrings`