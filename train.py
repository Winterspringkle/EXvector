import os
import pickle

import torch
import copy
import torch.nn as nn
from PIL.Image import Image
from matplotlib import pyplot as plt
from numpy.ma import copy
from sklearn import metrics
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time
import numpy as np
import torch.utils.data as DataSet
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from EXvector import EXvector
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix
import datetime
import random
import os
import time

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        # nn.init.normal_(m.weight, mean=0,  std=np.sqrt(2 / 256))

def compute_eer( answers, scores,name=None):
    """
    计算模型的评价指标
    :param answers: 训练数据对每一类的值(one-hot值)
    :param scores: 训练数据对每一类的预测值(softmax输出)
    :return: 准确率、等错误率、dcf
    """
    fpr, tpr, thresholds = roc_curve(answers, scores,pos_label=1)

    far, tar = fpr, tpr
    frr = 1 - tar
    if name:
        display = metrics.DetCurveDisplay(fpr=far, fnr=frr)
        display.plot()
        plt.show()
        np.save("./det_c/"+name+"_far.npy",far)
        np.save("./det_c/"+name + "_frr.npy", frr)
    cr = 10
    ca = 1
    pt = 0.01

    min_dirr = min([abs(far[i] - frr[i]) for i in range(len(far))])

    for i in range(len(far)):
        if abs(far[i] - frr[i]) == min_dirr:
            eer = (far[i] + frr[i]) / 2
            dcf = cr * frr[i] * pt + ca * far[i] * (1 - pt)
            break
    return eer, dcf

epochs=1000
BATCH_SIZE=256
# torch.manual_seed(2)
LR = 1e-3

if_use_gpu = True
crop_size=256
image_size=256,
method="train"
model_name="x_dann_DCN"
# 获取训练集dataset
#X-vector extracted from kaldi
train_data = np.load('./data/x_vector_transfer_68_2/train1.npz',allow_pickle=True)['features']
vali_data = np.load('./data/x_vector_transfer_68_2/train2.npz',allow_pickle=True)['features']
test_data = np.load('./data/x_vector_transfer_68_2/test.npz',allow_pickle=True)['features']

#speaker label
train_label = np.load('./data/x_vector_transfer_68_2/train1_label.npy')
vali_label = np.load('./data/x_vector_transfer_68_2/train2_label.npy')
test_label = np.load('./data/x_vector_transfer_68_2/test_label.npy')
#emotion domain label
e_train_label = np.load('./data/x_vector_transfer_68_2/e_train1_label.npy')
e_vali_label = np.load('./data/x_vector_transfer_68_2/e_train2_label.npy')
e_test_label = np.load('./data/x_vector_transfer_68_2/e_test_label.npy')

train_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(train_data, dtype=float)),
                                     torch.LongTensor(np.array(train_label)),torch.LongTensor(np.array(e_train_label)))
vali_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(vali_data, dtype=float)),
                                     torch.LongTensor(np.array(vali_label)),torch.LongTensor(np.array(e_vali_label)))
test_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(test_data, dtype=float)),
                                     torch.LongTensor(np.array(test_label)),torch.LongTensor(np.array(e_test_label)))

# 通过torchvision.datasets获取的dataset格式可直接可置于DataLoader
train_loader = DataSet.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
vali_loader = DataSet.DataLoader(vali_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataSet.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

old_acc=0
print(model_name)
print("train_data",train_data.shape)
print("train_label",train_label.shape)
print("train_emotion_label",e_train_label.shape)
print("vali_data",vali_data.shape)
print("vali_label",vali_label.shape)
print("vali_emotion_label",vali_label.shape)
print("test_data",test_data.shape)
print("test_label",test_label.shape)
print("test_emotion_label",e_test_label.shape)

loss_function = nn.CrossEntropyLoss()
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()
start_time = time.time()
if method=="train":
    cadan = EXvector()
    # print(cnn)
    cadan.apply(init_weights)
    if if_use_gpu:
        cadan = cadan.cuda()

    optimizer = torch.optim.Adam(cadan.parameters(), lr=LR)
    min_loss_val = 0  # 任取一个大数

    for epoch in range(1,epochs+1):
        if epoch>int(epochs*0.8):
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * 0.1
        step=0
        start = time.time()
        train_loss_list = []
        full_preds = []
        full_gts = []
        cadan=cadan.train()
        len_dataloader = max(len(train_loader), len(vali_loader))
        data_source_iter = iter(train_loader)
        data_target_iter = iter(vali_loader)
        while step < len_dataloader:
            p = float(step + epoch * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            if step % len(train_loader) == 0:
                data_source_iter = iter(train_loader)
            data_source = data_source_iter.next()
            s_img, s_label,e_label = data_source
            # print("e_label_len",e_label.shape)

            # training model using source data

            cadan.zero_grad()
            batch_size = len(s_label)

            input = torch.FloatTensor(batch_size, 512)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.LongTensor(batch_size)

            if if_use_gpu:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input = input.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            input.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)
            domain_label.resize_as_(e_label).copy_(e_label)

            class_output,domain_output = cadan(input_data=input, alpha=alpha)
            err_s_label = loss_class(class_output, class_label)

            # print(err_s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            if step%len(vali_loader)==0:
                data_target_iter = iter(vali_loader)
            data_target = data_target_iter.next()

            t_img, t_label ,e_label= data_target
            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 512)
            domain_label = torch.LongTensor(batch_size)

            class_label = torch.LongTensor(batch_size)
            if if_use_gpu:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)
            domain_label.resize_as_(e_label).copy_(e_label)
            class_label.resize_as_(t_label).copy_(t_label)
            class_output, domain_output = cadan(input_data=input_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err_t_label = loss_domain(class_output, class_label)

            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()
            step+=1
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]

        if epoch % 1 == 0 or epoch == epochs - 1:
            cadan=cadan.eval()
            test_loss_list = []
            full_preds = []
            full_gts = []
            embedding = []
            acc = [0, 0, 0, 0, 0]
            for step, (x, y,_) in enumerate(test_loader):
                b_x = Variable(x, requires_grad=False)
                b_y = Variable(y, requires_grad=False)
                if if_use_gpu:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                # output,r_vec = resnet(b_x)
                output,_ = cadan(b_x,alpha)
                loss = loss_function(output, b_y)
                predictions = torch.max(output, 1)[1].data.squeeze()
                test_loss_list.append(loss.item())
                # accuracy = sum(pred_y == b_y) / b_y.size(0)
                for pred in predictions.detach().cpu().numpy():
                    full_preds.append(pred)
                for lab in b_y.detach().cpu().numpy():
                    full_gts.append(lab)
            mean_acc = accuracy_score(full_gts, full_preds)
            mean_loss = np.mean(np.asarray(test_loss_list))
            print('Epoch:', epoch, '|valiloss:%.4f' % mean_loss, '|vali accuracy:%.4f' % mean_acc)
            if old_acc<mean_acc:
                old_acc=mean_acc
                model_save_path = os.path.join('model/' + model_name+'_best_model')
                state_dict = {'model': cadan.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, model_save_path)
            if epoch==epochs:
                model_save_path = os.path.join('model/' + model_name + '_last_model')
                state_dict = {'model': cadan.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, model_save_path)
cadan =EXvector()
cadan.load_state_dict(torch.load("./model/"+model_name+"_best_model")["model"])

print("test mode")
if if_use_gpu:
    cadan = cadan.cuda()
cadan = cadan.eval()
test_loss_list = []
full_preds = []
full_gts = []
embedding=[]
acc=[0,0,0,0,0]
scores = []
answers = []
full_emo=[]
predictions_list=[]
eer_label=[]
emo_eer_label=[[],[],[],[],[]]
emo_predictions_list=[[],[],[],[],[]]
for step, (x,y,y_emo) in enumerate(test_loader):
    b_x = Variable(x, requires_grad=False)
    b_y = Variable(y, requires_grad=False)
    if if_use_gpu:
        b_x = b_x.cuda()
        b_y = b_y.cuda()
    # output,r_vec = resnet(b_x)
    output,_ = cadan(b_x,0)

    predictions = torch.max(output, 1)[1].data.squeeze()

    loss = loss_function(output, b_y)
    test_loss_list.append(loss.item())
    # accuracy = sum(pred_y == b_y) / b_y.size(0)

    for pred in predictions.detach().cpu().numpy():
        full_preds.append(pred)
    for lab in b_y.detach().cpu().numpy():
        full_gts.append(lab)
    res1 = np.argmax(output.detach().cpu().numpy(), axis=1)
    # print(res1)
    for index1, i in enumerate(output.detach().cpu().numpy()):
        for index, j in enumerate(i):
            # print(index,y)
            predictions_list.append(j)
            emo_predictions_list[y_emo[index1]].append(j)
            # print(index,testing_labels[index1])
            if index == y[index1]:
                eer_label.append(1)
                # print("yes")
                emo_eer_label[y_emo[index1]].append(1)
            else:
                eer_label.append(0)
                emo_eer_label[y_emo[index1]].append(0)

    for i in output.detach().cpu().numpy():
        # print(i)
        target=np.max(i)
        scores.append(target)
    for i in y_emo:
        full_emo.append(i)
scores = np.array(scores)
answers = np.array(answers)
mean_acc = accuracy_score(full_gts, full_preds)
mean_loss = np.mean(np.asarray(test_loss_list))
print('|test phase ac:%.4f' % mean_acc,'|test phase loss:%.4f' % mean_loss,)
eer_score, dcf = compute_eer(np.array(eer_label), np.array(predictions_list))
print("acc:",mean_acc, "eer_score:",eer_score, "dcf:",dcf)
score=[0,0,0,0,0]
true_score=[0,0,0,0,0]
emotion=["ang","ela","neu","pan","sad"]

for i in range(len(full_preds)):
    if full_preds[i]==full_gts[i]:
        score[full_emo[i]]=score[full_emo[i]]+1
    true_score[full_emo[i]] = true_score[full_emo[i]] + 1
res = []
count=[0,0,0,0,0]
for i in range(len(test_label)):
    if test_label[i]==full_preds[i]:
        acc[e_test_label[i]]=acc[e_test_label[i]]+1
    count[e_test_label[i]]=count[e_test_label[i]]+1
for i in range(len(acc)):
    res.append(acc[i]/count[i])
for i in range(5):
    eer_score, dcf = compute_eer(np.array(emo_eer_label[i]),np.array(emo_predictions_list[i]) )
    print(emotion[i],"acc",res[i],"eer",eer_score,"dcf",dcf)
print(model_name)