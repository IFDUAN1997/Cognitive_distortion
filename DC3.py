# 在DC2的测试集固定基础之上(estSat1是正负数据1:1，testSet2是正负数据2:3)将训练集的负样本随机选取
#来源 https://cloud.tencent.com/developer/article/1792496
import torch
import time 
import torch.nn as nn
import torch.nn.functional as F 
from transformers import RobertaModel, BertTokenizer
import pandas as pd 
import numpy as np 
from tqdm import tqdm,trange 
from torch.utils.data import TensorDataset,RandomSampler,DataLoader,SequentialSampler
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = "./"
bert_path = "../pretrained_models/chinese-roberta-wwm-ext/"
tokenizer = BertTokenizer(vocab_file=bert_path+"vocab.txt")  # 初始化分词器

input_ids = []     # input char ids
test_ids=[]        # 测试集id

input_types = []   # segment ids
input_masks = []   # attention mask
label = []         # 标签
xOri=[]            # 原文
fpId=[]            # 预测结果fp的原id，用来找出错误原句子
fnId=[]            # 预测结果fn的原id，用来找出错误原句子
positiveNum=0
positive_order=[]
negative_order=[]
pad_size = 67      # 也称为 max_len (前期统计分析，文本长度最大值为66，取67即可覆盖100%)
 
with open(path + "testSet1.txt", encoding='utf-8') as f:
    for i, l in tqdm(enumerate(f)): 
        x1, y = l.strip().split('\t')
        Ori=x1
        x1 = tokenizer.tokenize(x1)   # 将文字拆分
        tokens = ["[CLS]"] + x1 + ["[SEP]"]
        
        # 得到input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)   # 将文字转换为编码
        types = [0] * len(ids)
        masks = [1] * len(ids)
        # 短则补齐，长则切断
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(types) == pad_size
        label.append([int(y)])
        xOri.append(Ori)


# 随机打乱索引
# 最后200条是测试数据
train_order = list(range(len(input_ids)-200))
test_order = list(range(len(input_ids)-200,len(input_ids)))

# 计算正负样本数量
for i in train_order:
    if label[i]==[1]:
        positiveNum+=1
        positive_order=positive_order+[i] # 所有的正样本
    else:
        negative_order=negative_order+[i] # 所有的负样本

per=positiveNum/(len(input_ids)-200)  #计算正样本的比例，负样本也要同样多 
np.random.seed(2020)   # 固定种子
np.random.shuffle(negative_order)  # 将负样本顺序打乱
train_order=positive_order+negative_order[:positiveNum]  # 把正负样本合在一起
np.random.shuffle(train_order)
np.random.shuffle(test_order)


# 训练集
input_ids_train = np.array([input_ids[i] for i in train_order])
input_types_train = np.array([input_types[i] for i in train_order])
input_masks_train = np.array([input_masks[i] for i in train_order])
y_train = np.array([label[i] for i in train_order])

# 测试集
input_ids_test = np.array([input_ids[i] for i in test_order])  
input_types_test = np.array([input_types[i] for i in test_order])
input_masks_test = np.array([input_masks[i] for i in test_order])
y_test = np.array([label[i] for i in test_order])
text_test = np.array(test_order)  #储存测试集的句子原id

BATCH_SIZE = 128
train_data = TensorDataset(torch.LongTensor(input_ids_train), 
                           torch.LongTensor(input_types_train), 
                           torch.LongTensor(input_masks_train), 
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)  
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_types_test), 
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test),
                          torch.LongTensor(text_test))  
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = RobertaModel.from_pretrained(bert_path,return_dict=False)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, 2)   # 768 -> 2  768是模型里的层数，不要修改。2是最后的线性层，只有2个输出。需要几分类要在这边改

    def forward(self, x): # (ids, seq_len, mask)
        context = x[0]  # 输入的句子   
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types, attention_mask=mask)
        # print(_.shape, pooled.shape) # torch.Size([128, 32, 768]) torch.Size([128, 768])
        # print(_[0,0] == pooled[0]) # False 注意是不一样的 pooled再加了一层dense和activation
        out = self.fc(pooled)   # 得到2分类
        return out

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #先用cuda跑，不然用cpu
model = Model().to(DEVICE)
#print(model) 

# param_optimizer = list(model.named_parameters())  # 模型参数名字列表
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

NUM_EPOCHS = 20  #epoch次数

def tpfptnfn(pred, y, textId, threshold=0.5):
    batchsize = y.shape[0]
    tp_samples, fp_samples, tn_samples, fn_samples = (
        0,
        0,
        0,
        0,
    )
    for sample in range(batchsize):
        if y.ndim == 1:   #判断输入标签的维度
            groundtruth_value = y[sample]
        elif y.ndim == 2:
            groundtruth_value = y[sample, 0]

        inference_value = pred[sample, 1]
        if (groundtruth_value > threshold) and (inference_value > threshold):
            tp_samples += 1
        elif (groundtruth_value <= threshold) and (inference_value > threshold):
            fp_samples += 1  
            fpId.append(textId[sample])       # 把fp的原id加到矩阵里面          
        elif (groundtruth_value <= threshold) and (inference_value <= threshold):
            tn_samples += 1
        elif (groundtruth_value > threshold) and (inference_value <= threshold):
            fn_samples += 1
            fnId.append(textId[sample])       # 把fn的原id加到矩阵里面
           
    return tp_samples, fp_samples, tn_samples, fn_samples


def train(model, device, train_loader, optimizer, epoch):   # 训练模型
    model.train()
    best_acc = 0.0 
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1, x2, x3])  # 得到预测结果
        optimizer.zero_grad()             # 梯度清零

        weightD10=torch.tensor([0.25,0.75]).to(device)   #加入权重
        loss = F.cross_entropy(y_pred, y.squeeze(), weight=weightD10).to(device)  # 得到loss

        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 100 == 0:    # 打印loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(x1), 
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), 
                                                                           loss.item()))  # 记得为loss.item()
            

def testOri(model, device, test_loader):    # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0 
    acc = 0 
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader): #每次跑一个batch

        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    
        # 记得加item()只取数字资料 ，view_as()是设定数据格式
        #预测标签与实际标签相同时累加
        


    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, acc, len(test_loader.dataset),
          100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)
 

def test(model, device, test_loader):    # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0 
    # acc = 0  #预测与实际相同的次数
    samples_TP = 0  # 预测正确 预测为正样本（原本为正）
    samples_FP = 0  # 预测错误 预测为正样本（原本为负）
    samples_TN = 0  # 预测正确 预测为负样本（原本为负）
    samples_FN = 0  # 预测错误 预测为负样本（原本为正）
    for batch_idx, (x1, x2, x3, y, textId) in enumerate(test_loader): #每次跑一个batch，y是结果，x1,2,3不用管
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3]) #模型预测的分数结果，不能直接视为概率，所以后续使用softmax将其转化为概率
        test_loss += F.cross_entropy(y_, y.squeeze()) 
        sfm = torch.nn.Softmax(dim=1) 
        pred = sfm(y_) #softmax会得到概率值
        # pred = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        # acc += pred.eq(y.view_as(pred)).sum().item()    
        # 记得加item()只取数字资料 ，view_as()是设定数据格式
        #预测标签与实际标签相同时累加
        tp_samples, fp_samples, tn_samples, fn_samples = tpfptnfn( pred, y,textId, threshold=0.5)  #用来计算对应的样本数量，用0.5做阈值划分。不写的默认值是0.5
        samples_TP += tp_samples
        samples_FP += fp_samples
        samples_TN += tn_samples
        samples_FN += fn_samples
  
    if ((samples_TP + samples_FP) != 0) and ((samples_TP + samples_FN) != 0):
        precision = samples_TP / (samples_TP + samples_FP)
        recall = samples_TP / (samples_TP + samples_FN)
        if (precision != 0) and (recall != 0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
    else:
        precision, recall, f1_score = 0, 0, 0

    test_loss /= len(test_loader)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #       test_loss, acc, len(test_loader.dataset),
    #       100. * acc / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}'.format( test_loss.item()))
    acc = (samples_TP + samples_TN) /(samples_TP + samples_FP + samples_TN + samples_FN)
    print('\nPrecision:{:.3f}. Recall:{:.3f}. F1 score:{:.3f}. acc score:{:.3f}'.format(precision, recall, f1_score, acc))
    print("ture       0   1")
    print("predict 0  {:} {:}".format(samples_TN, samples_FN))
    print("        1  {:} {:}".format(samples_FP, samples_TP))
    
    return acc, precision, recall, f1_score

best_acc = 0.0 
PATH = 'roberta_modelPD10Auto.pth'  # 定义模型保存路径
for epoch in trange(NUM_EPOCHS):  # 看几个epoch
    fpId.clear()         # 将fpid清零
    fnId.clear()         # 将fnid清零
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc, precision, recall, f1_score = test(model, DEVICE, test_loader)


    if best_acc < acc: 
        best_acc = acc 
        torch.save(model.state_dict(), PATH)  # 保存最优模型
        print("fn:")
        for i in range(len(fnId)-1):
            print(fnId[i],xOri[fnId[i]],sep=" ：")
        print("fp:")
        for i in range(len(fpId)-1):
            print(fpId[i],xOri[fpId[i]],sep=" ：")
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

model.load_state_dict(torch.load(PATH))
acc, precision, recall, f1_score = test(model, DEVICE, test_loader)

# Test set: Average loss: 0.4429, Accuracy: 387/476 (81.30%)
# acc is: 0.8130, best acc is 0.8130

# PsyQA 2500 XEMAC 1377)
# roberta_modelD10Psy  Precision:0.705. Recall:0.677. F1 score:0.691. acc score:0.846
# roberta_modelD10PsyEmma  Precision:0.617. Recall:0.671. F1 score:0.643. acc score:0.846
# roberta_modelD5Psy   Precision:0.615. Recall:0.661. F1 score:0.637. acc score:0.837
# roberta_modelD5PsyEmma   Precision:0.701. Recall:0.432. F1 score:0.535. acc score:0.879
# roberta_modelD10Psy862   Precision:0.742. Recall:0.697. F1 score:0.719. acc score:0.862