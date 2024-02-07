import os

import akshare as ak
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Arg

args = Arg( )

class LSTM(nn.Module):
    
    def __init__( self, input_size, rnn_unit, output_size, layer_num, dropout ):
        super(LSTM, self).__init__( )
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_unit, num_layers=layer_num, dropout=dropout)
        self.relu = nn.ReLU( )
        self.fc = nn.Linear(rnn_unit, output_size)
        self.sigmoid = nn.Sigmoid( )
    
    def forward( self, x ):
        out, _ = self.rnn(x)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out

# 定义数据加载器
class StockDataset(torch.utils.data.Dataset):
    
    def __init__( self, data, labels ):
        self.data = torch.Tensor(np.array(data))
        self.labels = torch.Tensor(np.array(labels))
    
    def __len__( self ):
        return len(self.data)
    
    def __getitem__( self, idx ):
        return self.data[idx], self.labels[idx]

class Stock:
    
    def __init__( self, file_path, ratio=0.8, random_seed=221213 ):
        self.file_path = file_path
        self.ratio = ratio
        self.random_seed = random_seed
    
    def get_data( self, code ):
        df = ak.fund_etf_hist_sina(symbol=code)
        p_change = df['close'].pct_change( )
        df['p_change'] = p_change

        forecast = self.set_target(df)
        df['forecast'] = pd.DataFrame(forecast).shift(-1)
        df = df.iloc[1:, :]

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        path = os.path.join(self.file_path, code + '.csv')
        df.to_csv(path, index=False)

        print("已获取全部数据")
    
    def upgrade_data( self, code ):
        path = os.path.join(self.file_path, code + '.csv')
        total_data = pd.read_csv(path)
        total_data['date'] = pd.to_datetime(total_data['date'])

        new_data = ak.fund_etf_hist_sina(symbol=code)
        p_change = new_data['close'].pct_change( )
        new_data['p_change'] = p_change

        forecast = self.set_target(new_data)
        new_data['forecast'] = pd.DataFrame(forecast).shift(-1)
        new_data = new_data.iloc[1:, :]
        new_data['date'] = pd.to_datetime(new_data['date'])

        missing_data = new_data[~new_data['date'].isin(total_data['date'])]

        if missing_data.empty:
            print("没有可更新的数据")
        else:
            total_data = pd.concat([total_data, new_data], ignore_index=True)
            total_data = total_data.sort_values(by='date')
            total_data.to_csv(path, index=False)

            print("数据已更新")

    def set_target( self, df ):
        p_changes = df['p_change'].values
        forecast = []

        for p_change in p_changes:
            if p_change * 100 >= 0:
                forecast.append(1)
            else:
                forecast.append(0)

        return forecast
    
    def split_data( self, code ):
        path = os.path.join(self.file_path, code + '.csv')
        df = pd.read_csv(path)
        forecast = self.set_target(df)
        df = df.iloc[:-1, :]
        
        ratio = args.ratio
        time_step = args.time_step
        
        data = df.iloc[:, 1:].values
        label = np.array(forecast[1:])
        
        normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        data_x, data_y = [], []
        
        for k in range(len(data) - time_step - 1):
            x = normalized_data[k:k + time_step, :6]
            y = label[k + time_step, np.newaxis]
            
            data_x.append(x.tolist( ))
            data_y.append(y)
        
        train_len = int(len(data_x) * ratio)
        train_x, train_y = data_x[:train_len], data_y[:train_len]
        val_x, val_y = data_x[train_len:], data_y[train_len:]
        
        return train_x, train_y, val_x, val_y
    
    def acc_cutoff_class( self, y_valid, y_pred_valid ):
        y_valid = np.array(y_valid)
        y_pred_valid = np.array(y_pred_valid)
        
        fpr, tpr, threshold = roc_curve(y_valid, y_pred_valid)
        pred_valid = pd.DataFrame({ 'label': y_pred_valid })
        thresholds = np.array(threshold)
        
        pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
        acc_scores = (pred_labels == y_valid).mean(axis=1)
        
        acc_df = pd.DataFrame({ 'threshold': threshold, 'test_acc': acc_scores })
        acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
        cutoff = acc_df.iloc[0, 0]
        
        y_pred_valid = np.where(y_pred_valid < float(cutoff), 0, 1)
        return y_pred_valid
    
    def train_model( self, code ):
        device = torch.device("cuda" if torch.cuda.is_available( ) else "cpu")
        
        if os.path.exists(f'./model/best_{code}_model.pth'):
            model = torch.load(f'./model/best_{code}_model.pth', map_location=device)
        else:
            model = LSTM(
                input_size=args.input_size,
                rnn_unit=args.rnn_unit,
                output_size=args.output_size,
                layer_num=args.layer_num,
                dropout=args.dropout
                ).to(device)
        
        loss_object = nn.BCELoss( ).to(device)
        optimizer = optim.Adam(model.parameters( ), lr=args.lr, betas=(0.9, 0.999), eps=1e-07)
        
        train_x, train_y, val_x, val_y = self.split_data(code)
        
        train_dataset = StockDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        best_accuracy = 0
        
        for epoch in range(args.epoch + 1):
            model.train( )
            for train_batch, labels_batch in train_loader:
                train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
                
                optimizer.zero_grad( )
                predictions = model(train_batch)
                
                predictions = predictions.view(-1)
                labels_batch = labels_batch.view(predictions.shape)
                
                loss = loss_object(predictions, labels_batch)
                loss.backward( )
                
                torch.nn.utils.clip_grad_norm_(model.parameters( ), max_norm=1)
                optimizer.step( )
            
            if epoch % 10 == 0:
                model.eval( )
                
                with torch.no_grad( ):
                    X_test_tensor = torch.Tensor(np.array(val_x)).to(device)
                    y_val_pred = model(X_test_tensor).cpu( ).numpy( )[:, 0]

                val_y = np.squeeze(val_y)
                accuracy = accuracy_score(val_y, self.acc_cutoff_class(val_y, y_val_pred))

                print(f'Epoch {epoch + 1}, Loss: {loss.item( )}, Validation Accuracy: {accuracy}')
                
                torch.save(model, f'./model/{code}_model_{epoch}.pth')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model, f'./model/best_{code}_model.pth')
    
    def test_model( self, code ):
        cutoff = args.cutoff
        
        if not os.path.exists(f'./model/best_{code}_model.pth'):
            self.train_model(code)
        
        model = torch.load(f'./model/best_{code}_model.pth', map_location='cpu')
        
        path = os.path.join(self.file_path, code + '.csv')
        test = pd.read_csv(path)
        val_x = test.iloc[-30:, 1:-1]
        
        with torch.no_grad( ):
            X_val_tensor = torch.Tensor(np.array(val_x))
            y_val_pred = model(X_val_tensor).numpy( )[-1, 0]
            
            y_val_pred = np.where(y_val_pred < float(cutoff), 0, 1)
            
            print(f'明天股票' + '上涨' if y_val_pred else '跌' + f', 正确率在{y_val_pred * 100}%左右')

if __name__ == '__main__':
    file_path = args.data_path
    codes = args.codes
    etf_path = path = os.path.join(file_path, 'all_etf.csv')
    
    if not os.path.exists(etf_path):
        all_etf = ak.fund_etf_category_sina(symbol="ETF基金")
        all_etf.to_csv(etf_path)
    
    stock = Stock(file_path=file_path)
    
    for code in codes:
        path = os.path.join(file_path, code + '.csv')
        
        if os.path.exists(path):
            stock.upgrade_data(code)
            # output = stock.test_model(code)
        else:
            stock.get_data(code)
            # stock.train_model(code)
