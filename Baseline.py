#%%
import random
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparmeter Setting
CFG = {
    'TRAIN_WINDOW_SIZE':90, # 90일치로 학습
    'PREDICT_SIZE':21, # 21일치 예측 
    'EPOCHS':10,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':4096,
    'SEED':41
}

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

# seed_everything(CFG['SEED']) # Seed 고정
sales_data = pd.read_csv('./sales.csv') #가격
train_data = pd.read_csv('./train.csv').drop(columns=['ID'])
sales = sales_data.iloc[:,6:]/train_data.iloc[:,6:]
sales=sales.fillna(0)
sales_data.iloc[:,6:]=sales


# Data Scaling Preprocessing
scale_max_dict = {}
scale_min_dict = {}
# Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['제품','대분류', '중분류', '소분류', '브랜드']

for col in categorical_columns:
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
numeric_cols = train_data.columns
# 각 column의 min 및 max 계산
min_values = train_data[numeric_cols].min(axis=1)
max_values = train_data[numeric_cols].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges = max_values - min_values
ranges[ranges == 0] = 1
# min-max scaling 수행
train_data[numeric_cols] = (train_data[numeric_cols].subtract(min_values, axis=0)).div(ranges, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict = min_values.to_dict()
scale_max_dict = max_values.to_dict()
    
indexs_bigcat={}
for bigcat in train_data['대분류'].unique():
    indexs_bigcat[bigcat] = list(train_data.loc[train_data['대분류']==bigcat].index)
def PSFA(pred, target): 
    PSFA = 1
    for cat in range(5):
        ids = indexs_bigcat[cat]
        for day in range(21):
            total_sell = np.sum(target[ids, day]) # day별 총 판매량
            pred_values = pred[ids, day] # day별 예측 판매량
            target_values = target[ids, day] # day별 실제 판매량
            
            # 실제 판매와 예측 판매가 같은 경우 오차가 없는 것으로 간주 
            denominator = np.maximum(target_values, pred_values)
            diffs = np.where(denominator!=0, np.abs(target_values - pred_values) / denominator, 0)
            
            if total_sell != 0:
                sell_weights = target_values / total_sell  # Item별 day 총 판매량 내 비중
            else:
                sell_weights = np.ones_like(target_values) / len(ids)  # 1 / len(ids)로 대체
                
            if not np.isnan(diffs).any():  # diffs에 NaN이 없는 경우에만 PSFA 값 업데이트
                PSFA -= np.sum(diffs * sell_weights) / (21 * 5)
            
            
    return PSFA
def make_train_data(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE']):
    '''
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    '''

    num_rows = len(data)
    window_size = train_size + predict_size
    
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1),len(data.iloc[0, :5]) + 1,train_size))
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :5])
        sales_data = np.array(data.iloc[i, 5:])
        
        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            temp_data = np.concatenate((np.tile(encode_info.reshape(-1,1), (1,train_size)), window[:train_size].reshape(1,-1)),axis=0)
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window[train_size:]
    
    return input_data, target_data

def make_predict_data(data, train_size=CFG['TRAIN_WINDOW_SIZE']):
    '''
    평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
    data : 일별 판매량
    train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
    '''
    num_rows = len(data)
    
    input_data = np.empty((num_rows, len(data.iloc[0, :5]) + 1, train_size))
    
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :5])
        sales_data = np.array(data.iloc[i, -train_size:])
        
        window = sales_data[-train_size : ]
        temp_data = np.concatenate((np.tile(encode_info.reshape(-1,1), (1,train_size)), window[:train_size].reshape(1,-1)),axis=0)
        input_data[i] = temp_data
    
    return input_data

train_input, train_target = make_train_data(train_data)
test_input = make_predict_data(train_data)

# Train / Validation Split
data_len = len(train_input)
val_input = train_input[-int(data_len*0.2):]
val_target = train_target[-int(data_len*0.2):]
train_input = train_input[:-int(data_len*0.2)]
train_target = train_target[:-int(data_len*0.2)]
# dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])
    
    def __len__(self):
        return len(self.X)
    
train_dataset = CustomDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_input, val_target)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
#model
class BaseModel(nn.Module):
    def __init__(self, input_feature=6, output_size=CFG['PREDICT_SIZE']):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_feature,out_channels=64,kernel_size=7,padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.maxpool= nn.MaxPool1d(2,2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.avg_pool= nn.AdaptiveAvgPool1d(21)
        self.conv4=nn.Conv1d(in_channels=256,out_channels=1,kernel_size=1)
 

    
    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.maxpool(x)
        x=self.conv3(x)
        x=self.avg_pool(x)
        x=self.conv4(x)
        return x.squeeze(1)
    

# Training
def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_metric=99999
    best_loss_model = None
    best_metric_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss,val_metric = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}], Val metric:[{val_metric:.5f}]')
        
        if best_loss > val_loss:
            best_loss = val_loss
            best_loss_model = model
            print('Model Updated')
        if best_metric > val_metric:
            best_metric = val_metric
            best_metric_model = model
            print('Model Updated')
    if best_loss_model is not None:
        torch.save(best_loss_model.state_dict(),"./best_loss_model.pth")
        print('Best Model Saved')
    if best_metric_model is not None:
        torch.save(best_metric_model.state_dict(),"./best_loss_model.pth")
        print('Best Model Saved')
    
    return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    pred = []
    target = []
    
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            output = model(X)
            loss = criterion(output, Y)
            Y = Y.cpu().numpy()
            target.extend(Y)
            output = output.cpu().numpy()
            pred.extend(output)

            val_loss.append(loss.item())
    pred = np.array(pred)
    target = np.array(target)
    # for idx in range(len(pred)):
    #     pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    #     target[idx, :] = target[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]

    # 결과 후처리
    pred = np.round(pred, 0).astype(int)
    target = np.round(target, 0).astype(int)
    
    return np.mean(val_loss),PSFA(pred, target)

model = BaseModel()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
infer_model = train(model, optimizer, train_loader, val_loader, device)