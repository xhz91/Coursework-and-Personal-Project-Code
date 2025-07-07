import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from BertModel.PreProcessing import BertDataset, generate_batch
import numpy as np
from sklearn import metrics

loss_criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertAnalyzer:

    def __init__(self, model, batch_size, max_seq_len, epochs, lr):
        self.net = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_seq_len = max_seq_len

    def train(self, data_file, save_path):
        train_data = BertDataset.from_data(data_file)
        train_loader = DataLoader(train_data,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  collate_fn = lambda batch: generate_batch(batch, max_seq_len = self.max_seq_len))
        self.net.to(device)
        optimiser = optim.Adam(self.net.parameters(), lr = self.lr)

        for epoch in range(self.epochs):
            batch_loss = 0.0
            for i, batch in enumerate(train_loader):
                input_ids, attn_mask, labels = tuple(i.to(device) for i in batch)
                optimiser.zero_grad()
                outputs = self.net(input_ids, attn_mask).squeeze(dim = 1)
                l = loss_criterion(outputs, labels.float())
                l.backward()
                optimiser.step()
                batch_loss += l.item()
                if i % 10 == 9:
                    print(f'''epoch:, {epoch + 1} -- batch: {i +1} -- avg loss: {batch_loss/10:.4f}''')
                    batch_loss = 0.0
        if save_path is not None:
            torch.save(self.net.state_dict(), save_path)
 
    def evaluate(self, data_file):
        test_data = BertDataset.from_data(data_file)
        test_loader = DataLoader(test_data,
                                 batch_size = self.batch_size,
                                 shuffle = True,
                                 num_workers = 4,
                                 collate_fn = lambda batch: generate_batch(batch, max_seq_len = self.max_seq_len))
        predicted = []
        truths = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attn_mask, labels = tuple(i.to(device) for i in batch)
                outputs = self.net(input_ids, attn_mask).squeeze(dim = 1)
                pred = (outputs >=0).int()
                predicted += pred.tolist()
                truths += labels.tolist()  
        accuracy = metrics.accuracy_score(truths, predicted)
        print(f'Accuracy: {accuracy:.4f}')
        cm = metrics.confusion_matrix(truths, predicted, labels = [0,1])
        print('Confusion Matrix:')
        print(cm)
        f1 = metrics.f1_score(truths, predicted)
        print("f1")
        print(f1)
        return f1

if __name__ == "__main__":
    path = 'dontpatronizeme_pcl.tsv'
    titles = ['par_id', 'art_id', 'keyword','country_code','text','label']
    raw_data = pd.read_csv(path, skiprows = 4, sep = '\t',
                           names = titles)
    raw_data = raw_data.dropna()
    raw_data['label'] = np.where(raw_data['label'] > 1, 1, 0)
    raw_data_shuffled = raw_data.sample(frac = 1, random_state = 1).reset_index(drop = True)
    split_index = int(0.8 * len(raw_data_shuffled))
    train_df = raw_data_shuffled.iloc[:split_index]
    test_df = raw_data_shuffled.iloc[split_index:]
    BertModel = BertAnalyzer()
    # BertModel.laod_saved('BertModel.pkl')
    BertModel.evaluate(test_df)
 