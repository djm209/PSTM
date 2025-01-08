import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(2) # aviod high cpu avg usage


def parse_args():

    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="pstm/STFormer_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="pstm/PSTM_METR-LA.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="pstm/STFormer_PEMS03.py", help="training config")
    parser.add_argument("-c", "--cfg", default="pstm/PSTM_PEMS03.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="pstm/STFormer_PEMS07.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="pstm/PSTM_PEMS07.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="pstm/STFormer_PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="pstm/PSTM_PEMS-BAY.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="pstm/STFormer_PEMS08.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="pstm/PSTM_PEMS08.py", help="training config")
    
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    launch_training(args.cfg, args.gpus)
