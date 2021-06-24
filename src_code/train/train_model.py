import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from sklearn.preprocessing import MaxAbsScaler
import shutil

import visdom
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from model.multi_scale_ori import *
from imblearn.over_sampling import SMOTE
from PIL import Image
from utils import *
from preprocessing import pre
from config import cfg
import datetime
import warnings
warnings.simplefilter("ignore")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def main():
    vis = visdom.Visdom(env="resnet1D_focal_loss_2", port=8097)
    
    data = pre(train_data_path=cfg.train_data_path, 
           test_data_path=cfg.test_data_path,
           y_train_path=cfg.train_label, 
           y_val_path=cfg.test_label,
           batch_size=cfg.batch_size, 
           aspect_ratio=cfg.aspect_ratio)
    
    #clear gpu_cache
    torch.cuda.empty_cache() 

    #set parameter 
    #class　
    n_class = cfg.n_class

    # Number of epochs to train for
    num_epochs = cfg.num_epochs

    # Flag for feature extracting. When False, we finetune the whole model,
    # when True we only update the reshaped layer params
    feature_extract = False

    #use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #learning rate 
    lr=cfg.learn_rate
    msresnet = MSResNet(input_channel=cfg.channel, layers=cfg.resnet_layer, num_classes=n_class)

    # use GPU
    msresnet.to(device)

    #weight decay
    weight_decay=cfg.weight_decay
    
    params_to_update = msresnet.parameters()

    #Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(msresnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
    criterion = FocalLoss(logits=True)
     
    #train_model
    currunt_turn = 0
    loss_list = []

    for epoch in range(num_epochs):
    
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
     
        for epoch_i, (inputs_nonacc, labels_nonacc) in enumerate(data.train_nonacc_dataloader):
        
            msresnet.train() 
            currunt_turn += 1
        
            #sampling_acc, labels_acc
            inputs_acc, labels_acc = next(iter(data.train_acc_dataloader))
        
            #concat_sampling 
            inputs = torch.cat([inputs_nonacc, inputs_acc], axis=0) 
            labels = torch.cat([labels_nonacc, labels_acc], axis=0) 
        
            #shuffle inputs and labels 
            shuffled_index = torch.randperm(cfg.batch_size + cfg.aspect_ratio * cfg.batch_size)
            inputs = inputs[shuffled_index]
            labels = labels[shuffled_index]
        
            #use GPU
            inputs, labels = Variable(inputs.type(torch.FloatTensor).cuda()), Variable(labels.float().cuda())
        
            #init optimizer
            optimizer.zero_grad()
        
            #calc output
            print(inputs.shape)
            p1d = (0, 0, 0, 735)
            #outputs, _ = msresnet(inputs.permute(0, 2, 1))
            inputs = F.pad(inputs, p1d, "constant", 0.5)
            print(inputs.shape)
            
            outputs, _ = msresnet(inputs.permute(0, 2, 1))
            loss = criterion(outputs, labels)
        
            #calc_pred_label
            outputs = torch.sigmoid(outputs)
            tag_one= torch.ones_like(outputs)
            tag_zero= torch.zeros_like(outputs)
            preds = torch.where(outputs >= 0.5, tag_one, tag_zero)
        
            preds = preds.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            #calc f1 scores
            precision = precision_score(labels, preds)
            recall = recall_score(labels, preds)
            f1 = f1_score(labels, preds)
        
            visulization(vis, ptype= "line",X=currunt_turn, Y=f1, win_name="f1")       
            visulization(vis, ptype= "line",X=currunt_turn, Y=loss.item(), win_name="loss")
        
            #backward 
            loss.backward()
            optimizer.step() 
            if epoch_i % 50 == 0: 
                dt_now = datetime.datetime.now()
                torch.save(msresnet.state_dict(), cfg.training_model_path + "model_{}_{}.pth".format(str(epoch) + "_" + str(epoch_i), str(dt_now.month) + "_" + str(dt_now.day)))
                outputs_val_all, preds_val_all, labels_val_all, loss_val_all = cal_train_data(msresnet=msresnet, dataloader=data.test_dataloader, criterion=criterion)
                outputs_train_all, preds_train_all, labels_train_all, loss_train_all = cal_train_data(msresnet=msresnet, dataloader=data.train_dataloader, criterion=criterion)
            
                loss_mean_test, f1_val = eval_metric(loss_val_all, labels_val_all, preds_val_all, epoch, epoch_i, phaze="test")
                loss_mean_train, f1_train = eval_metric(loss_train_all, labels_train_all, preds_train_all, epoch, epoch_i, phaze="train")
                
                loss_list.append({"model"+ "_" + str(epoch) + "_" + str(epoch_i) + "_" + str(dt_now.month) + "_" + str(dt_now.day) + ".pth": f1_val})
            
                visulization(vis, ptype= "line",X=currunt_turn, Y=loss_mean_test, win_name="loss_test")
                visulization(vis, ptype= "line",X=currunt_turn, Y=f1_val, win_name="f1_test")
                visulization(vis, ptype= "line",X=currunt_turn, Y=loss_mean_train, win_name="loss_train")
                visulization(vis, ptype= "line",X=currunt_turn, Y=f1_train, win_name="f1_train")
    
    index = np.argmax([[v for k, v in d.items()][0] for d in loss_list])
    print("best_model:", loss_list[index])
    shutil.copy(cfg.training_model_path+list(loss_list[index].keys())[0], cfg.infer_model_path+list(loss_list[index].keys())[0])

if __name__ == "__main__":
    main()