# ライブラリ
import numpy as np
import pickle as pkl
import pandas as pd 
import os
import warnings
import logging
import configparser
import datetime as dt
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.multi_scale_ori import *


# Config parser
config = configparser.ConfigParser()
config.read(os.path.join('config.ini'))

# ENV
batch_size= int(config['PARAMETER']['infer_batch_size'])
data_path = config['PATH']['infer_data']
model_param_path = config['PATH']['infer_model_param']
output_path = os.path.join('..', '..','output', 'infer_output')


def now2string():
    now_t = dt.datetime.now()
    now_s = now_t.strftime('%Y_%m_%d_%H_%M')
    return now_s

# logging
def log(path, file):
    log_file = os.path.join(path, file)
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
    
    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    handler = logging.FileHandler(log_file)

    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

log_path = output_path
logger = log(path=log_path, file="inference.logs")

# load model
torch.cuda.empty_cache()
n_class = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
msresnet = MSResNet(input_channel=3, layers=[1,1,1,1], num_classes=n_class)
msresnet.to(device)

if device.type=='cpu':
    msresnet.load_state_dict(torch.load(model_param_path,map_location=torch.device('cpu')))
else:
    msresnet.load_state_dict(torch.load(model_param_path))
# inference
if __name__ == "__main__":
    now_s = now2string()
    data_name = data_path.split('/')[-1]
    model_param_name = model_param_path.split('/')[-1]

    inputs_data = pd.read_pickle(data_path)
    inputs_dataset = TensorDataset(torch.from_numpy(inputs_data))
    inputs_dataloader = torch.utils.data.DataLoader(inputs_dataset,batch_size=batch_size)

    logger.info("Inference Start")
    logger.info("model paramters : {}".format(model_param_name))
    logger.info("data: {}".format(data_name))
    msresnet.eval()
    my_inputsiter = iter(inputs_dataloader)
    length = len(inputs_dataloader)
    output_list = list()
    for i in range(length):
        inputs = next(my_inputsiter)
        inputs = inputs[0]
        inputs = inputs.float().cuda()
        outputs, _ = msresnet(inputs.permute(0, 2, 1))
        outputs = torch.sigmoid(outputs)
        tag_one = torch.ones_like(outputs)
        tag_zero = torch.zeros_like(outputs)
        preds = torch.where(outputs >= 0.5, tag_one, tag_zero)
        preds = preds.data.cpu().numpy()
        preds_expr = preds.tolist()
        output_list.append(preds_expr)
    logger.info("result: {}".format(output_list))
    np_output_list = np.array(output_list)
    
    save_name = now_s+'_'+'result.pkl'
    save_path = os.path.join(output_path, save_name)
    with open(save_path, 'wb') as f:
        pkl.dump(np_output_list, f)
    logger.info("Inference End")
    logger.info("--"*10)