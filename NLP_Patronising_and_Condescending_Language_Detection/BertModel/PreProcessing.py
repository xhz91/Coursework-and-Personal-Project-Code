import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, XLNetTokenizer
#from transormers import Robert
TRAINED_WEIGHTS = "xlnet-base-cased"
MAX_SEQ_LEN = 128
tokenizer = XLNetTokenizer.from_pretrained(TRAINED_WEIGHTS)
class BertDataset(Dataset):
    def __init__(self):
        self.data = []

    @staticmethod
    def from_data(raw_data):
        dataset = BertDataset()
        dataset.data = [(text, label) for text, label in zip(raw_data['text'], raw_data['label']) ]
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return {'text': text,
                'label': label}

def generate_batch(batch, max_seq_len):
    encoded = tokenizer([entry['text'] for entry in batch],
                         padding = True,
                         add_special_tokens = True,
                         max_length = max_seq_len,
                         truncation = True,
                         return_tensors = 'pt')

    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']
    labels = torch.tensor([entry['label'] for entry in batch])
    return input_ids, attn_mask, labels

if __name__ == '__main__':
    path = 'dontpatronizeme_pcl.tsv'
    titles = ['par_id', 'art_id', 'keyword','country_code','text','label']
    raw_data = pd.read_csv(path, skiprows = 4, sep = '\t',
                           names = titles)
    raw_data = raw_data.dropna()
    BD = BertDataset()
    print(BD.from_data(raw_data)[0])