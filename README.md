# Pre-training Spatio-Temporal Masked Graph Neural Networks for Traffic Forecasting
<img width="1260" height="714" alt="image" src="https://github.com/user-attachments/assets/b405f676-1551-46bf-93fb-2a9d98724deb" /># PSTM

# We have provided the following information for reproducibility
1. Hyper-parameter settings in pre-training and fine-tuning phases
2. Pre-processed data of the open-source PeMS03 dataset
3. Detailed model structure under the framework of PyTorch, EasyTorch, and BasicTS.

# Requirement
The code is built based on Python 3.9, PyTorch 1.10.0, and EasyTorch. You can install PyTorch following the instruction in PyTorch. For example:

    `pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`
# Data Preparation
You can download all the raw datasets at Google Drive or Baidu Yun, and unzip them to datasets/raw_data/.

# Train PSTM based on a Pre-trained STFormer
    python pstm/run.py --cfg='pstm/STFormer_$DATASET.py' --gpus='0, 1'
    # python pstm/run.py --cfg='pstm/STFormer_PEMS03.py' --gpus='0, 1'
    # python pstm/run.py --cfg='pstm/STFormer_PEMS04.py' --gpus='0, 1'
# Train STEP from Scratch
All the training logs, including the config file, training log, and checkpoints, will be saved in checkpoints/MODEL_EPOCH/MD5_of_config_file. For example,   checkpoints/STFormer_100/5afe80b3e7a3dc055158bcfe99afbd7f. Then change the parameter of checkpoints file location. 

    python pstm/run.py --cfg='pstm/PSTM_$DATASET.py' --gpus='0, 1'
    # python pstm/run.py --cfg='pstm/PSTM_PEMS03.py' --gpus='0, 1'
    # python pstm/run.py --cfg='pstm/PSTM_PEMS04.py' --gpus='0, 1'
# Acknowledgement
We appreciate the EasyTorch and BasicTS toolboxes to support this work.

# More details is on the way
1. Detailed results of several experiments.
2. Visualization result.
