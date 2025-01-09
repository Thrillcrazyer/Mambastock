import pandas as pd
import torch
from torch.utils.data import IterableDataset
import os
import numpy as np



def data_formatting(data,start_idx,batch_size,seq_length,d_type,orig=False,is_multi=True):
        batches=[]
        eps=1e-9
        for i in range(batch_size):
            pd_data= data[start_idx+i:start_idx+i + seq_length]
            if is_multi:
                cols=["Open","High","Low","Close","Volume"]
            else:
                cols=["Close"]
            pd_data=pd_data[cols]
            numpy_arrays = pd_data.values
            if orig:
                new_tensor = torch.tensor(numpy_arrays, dtype=torch.float64).to("cuda")
                batches.append(new_tensor)
            else:    
                new_tensor = torch.tensor(numpy_arrays, dtype=torch.float64).to("cuda")+eps
                first_row = new_tensor[0, :].view(1, -1).clone()
                first_row[:, 1:4] = first_row[:, 0]
                result = new_tensor / first_row *100
                batches.append(result)
        batch = torch.stack(batches, dim=0)
        return batch.to(d_type)


class ConstantAllLengthDataset(IterableDataset):
    """
    한 폴더 내에 csv 파일을 seq_length 만큼 슬라이딩한다.
    """
    
    def __init__(self,batch=1, seq_length=128, folder_path='./data',d_type=torch.float32,is_multi_feature=True):
        super(ConstantAllLengthDataset).__init__()
        files = self.find_all_csv_files(folder_path)
        print(f'{len(files)} is exist')
        self.folder_path=folder_path
        self.csv_files = [file for file in files if file.endswith('.csv')]
        self.seq_length = seq_length    
        self.batch=batch
        self.d_type=d_type
        self.multi=is_multi_feature
        
    
    def find_all_csv_files(self, directory):
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        return csv_files
    def __iter__(self):
        # 각 CSV 파일을 순회
        for file in self.csv_files:
            file_path = file
            data = pd.read_csv(file_path).dropna()
            data = data[~(data == 0).any(axis=1)]
            # seq_length 크기의 청크로 데이터를 나눔
            for start_idx in range(0, len(data)-self.seq_length-self.batch,self.batch):
                yield data_formatting(data,start_idx,self.batch,self.seq_length,self.d_type,is_multi=self.multi)
                
    
class ConstantLengthDataset(IterableDataset):
    """
    한 폴더 내에 csv 파일을 seq_length 만큼 슬라이딩한다.
    """
    
    def __init__(self,batch=1, seq_length=128, folder_path='./test_data',d_type=torch.float32,is_multi_feature=True):
        super(ConstantLengthDataset).__init__()
        files = os.listdir(folder_path)
        print(f'{len(files)} is exist')
        self.folder_path=folder_path
        self.csv_files = [file for file in files if file.endswith('.csv')]
        self.seq_length = seq_length    
        self.batch=batch
        self.d_type=d_type
        self.multi=is_multi_feature
        
    def __iter__(self):
        # 각 CSV 파일을 순회
        for file in self.csv_files:
            file_path = os.path.join(self.folder_path, file)
            print(file,"data")
            data = pd.read_csv(file_path).dropna()
            data = data[~(data == 0).any(axis=1)]
            # seq_length 크기의 청크로 데이터를 나눔
            for start_idx in range(0, len(data)-self.seq_length-self.batch,self.batch):
                yield data_formatting(data,start_idx,self.batch,self.seq_length,self.d_type,is_multi=self.multi)
                
class OriginalConstantLengthDataset(IterableDataset):
    """
    한 폴더 내에 csv 파일을 seq_length 만큼 슬라이딩한다.
    """
    
    def __init__(self,batch=1, seq_length=128, folder_path='./test_data',d_type=torch.float32,is_multi_feature=True):
        super(OriginalConstantLengthDataset).__init__()
        files = os.listdir(folder_path)
        print(f'{len(files)} is exist')
        self.folder_path=folder_path
        self.csv_files = [file for file in files if file.endswith('.csv')]
        self.seq_length = seq_length    
        self.batch=batch
        self.d_type=d_type
        self.multi=is_multi_feature
    def __iter__(self):
        # 각 CSV 파일을 순회
        for file in self.csv_files:
            file_path = os.path.join(self.folder_path, file)
            print(file,"data")
            data = pd.read_csv(file_path).dropna()
            data = data[~(data == 0).any(axis=1)]
            # seq_length 크기의 청크로 데이터를 나눔
            for start_idx in range(0, len(data)-self.seq_length-self.batch,self.batch):
                yield data_formatting(data,start_idx,self.batch,self.seq_length,self.d_type,orig=True, is_multi=self.multi)
                
    
        
    
 
if __name__ == "__main__":
    # test code
    iterator=ConstantAllLengthDataset(folder_path='./data',seq_length=10,batch=2)
    iterator=ConstantLengthDataset(folder_path='./data/crypto',seq_length=10,batch=2)
    for i,dataset in enumerate(iterator):
        if i==10:
            break
        print(dataset)
        