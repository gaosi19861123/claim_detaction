import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
import matplotlib.pyplot as plt
import itertools
import visdom
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

def plot_confusion_matrix(outputs, label, fig_title="train", therhold=0.5):
    predict = list(map(lambda x: x >=therhold, outputs))
    outputs = list(map(lambda x: float(x), outputs))
    
    cm = confusion_matrix(label, predict) 
    f1 = f1_score(label, predict)
    fpr, tpr, thr = roc_curve(label, outputs)
    AUC = auc(fpr, tpr)
    print("f1:{:.2f}, AUC:{:.2f}".format(f1, AUC))
    
    plt.figure(fig_title)
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.colorbar()
    normalize=False

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def cal_train_data(msresnet, dataloader, criterion):                
    msresnet.eval()
    #test_phaze
    outputs_val_all = []
    preds_val_all = []
    labels_val_all = []
    loss_val_all = []

    with torch.no_grad():
        for i, (inputs_val, labels_val) in enumerate(dataloader):
                inputs_val, labels_val = Variable(inputs_val.type(torch.FloatTensor).cuda()), Variable(labels_val.float().cuda())
                
                if True:
                    p1d = (0,0,0,735) # pad last dim by 1 on each side
                    inputs_val = F.pad(inputs_val, p1d, "constant", 0)
                    
                outputs_val_epoch, _ = msresnet(inputs_val.permute(0, 2, 1))
                loss_val_epoch = criterion(outputs_val_epoch, labels_val)
                    
                outputs_val_epoch= F.sigmoid(outputs_val_epoch)
                tag_one_val= torch.ones_like(outputs_val_epoch)
                tag_zero_val= torch.zeros_like(outputs_val_epoch)
                preds_val = torch.where(outputs_val_epoch >=0.5, tag_one_val, tag_zero_val)
            
                outputs_val_epoch = outputs_val_epoch.cpu().numpy()
                preds_val = preds_val.data.cpu().numpy()
                labels_val = labels_val.data.cpu().numpy()
                loss_val_epoch = loss_val_epoch.data.cpu().numpy()
            
                outputs_val_all.extend(outputs_val_epoch)
                preds_val_all.extend(preds_val)
                labels_val_all.extend(labels_val)
                loss_val_all.append(loss_val_epoch)
                
        return outputs_val_all, preds_val_all, labels_val_all, loss_val_all
    
def visulization(vis, ptype, X, Y, win_name):
    """
    vis:visdom instance
    ptype:plot type, "line", "scattor"
    X:X aixs label
    Y:Y aixs label
    win_name: name of pane
    """
    if ptype == "line":
        vis.line(
            X=np.array([X]), 
            Y=np.array([Y]), 
            win='loss_' + win_name, 
            update="append",
            opts={
                'title':win_name,
                'xlabel':'epoch',
                'ylabel':'loss', 
                } 
            )
    
#画像を読みこみ関数
def get_pic_path(root_path, hash_name):
    root_path = root_path
    folder_name = hash_name[0][:2] + "/"
    file_name = hash_name
    img = root_path + folder_name + file_name + ".png"
    return img

def eval_metric(loss_val_all, labels_val_all, preds_val_all, epoch, epoch_i, phaze):    
    loss_mean_test = np.mean(loss_val_all)
    
    if loss_mean_test > 1:
        loss_mean_test = 1
        
    f1_val = f1_score(labels_val_all, preds_val_all)
    precision_val = precision_score(labels_val_all, preds_val_all)
    recall_val = recall_score(labels_val_all, preds_val_all)
    print("{} => epoch:{}, f1:{:.3f}, loss:{:.3f}".format(phaze, str(epoch) + "_" + str(epoch_i) ,f1_val, loss_mean_test))
    print("{} => epoch:{}, precision_val:{:.3f}, recall_val:{:.3f}".format(phaze, str(epoch) + "_" + str(epoch_i) ,precision_val, recall_val))
    return loss_mean_test, f1_val

def plot_predict_probability(predict, preds_label, labels, fig_text="train"):
    #plot all predict probability
    plt.figure(fig_text + str(1))
    plt.title("plot all predict probability")
    plt.hist([float(i) for i in predict], bins=10)

    #plot the predict probability of misclassification label 
    plt.figure(fig_text + str(2))
    plt.title("plot the predict probability of misclassification label")
    plt.hist(np.array(predict)[np.array(preds_label) != np.array(labels)], bins=10)
    
def plt_pr_roc(labels, outputs, fig="train"):
    precision, recall, thresholds = precision_recall_curve(labels, outputs)
    f1 = 2 * precision * recall / (precision + recall)   
    plt.figure(fig)
    plt.plot(recall, precision, label="PR_curve")
    plt.plot(recall, f1, label="f1")
    plt.legend()
    plt.title(fig + '_PR curve & f1')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    