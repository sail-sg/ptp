## Install PyTorch and the other dependencies for PTP

The code has been test on '1.9.0+cu102' and python3.8.



```bash
conda create -n ptp python==3.8
conda activate ptp
# CUDA 10.2
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
cd [PATH_TO_PTP]
pip install -r requirements.txt
```
