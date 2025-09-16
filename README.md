## Overview

Jetsons style cooking robot

Simple test of the openVLA model with Isaac Sim 5.0


## VS Code Installation for intellisense

https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html


### 1. Install Miniconda
https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### 2. install isaacsim 5.0

```
conda create -n cookbot python=3.11 -y
conda activate cookbot
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
```

### 3. initialize isaacsim

From the terminal, run:
```
isaacsim
```
Then WAIT. It does a lot of stuff on inital startup. If you cancel this early you will break the isaacsim install and have to reinstall it. IDK what it's doing, but it takes a while and it's important.

### 4. setup vscode

```
python -m isaacsim --generate-vscode-settings
```

Now you should have intellisense in vscode.

