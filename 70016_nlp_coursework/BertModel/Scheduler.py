import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
if str(os.getcwd()).endswith('BertModel'):
    os.chdir("..")


path = '/vol/bitbucket/ql1024/70016_nlp_coursework/dontpatronizeme_pcl.tsv'
titles = ['par_id', 'art_id', 'keyword','country_code','text','label']
raw_data = pd.read_csv(path, skiprows = 4, sep = '\t',
                       names = titles)
raw_data = raw_data.dropna()
raw_data['label'] = np.where(raw_data['label'] > 1, 1, 0)


train = pd.read_csv("/vol/bitbucket/ql1024/70016_nlp_coursework/semeval-2022/practice_splits/train_semeval_parids-labels.csv")
test = pd.read_csv("/vol/bitbucket/ql1024/70016_nlp_coursework/semeval-2022/practice_splits/dev_semeval_parids-labels.csv")
train_df = raw_data[raw_data["par_id"].isin(train['par_id'])]
test_df = raw_data[raw_data["par_id"].isin(test['par_id'])]

train_df = raw_data[raw_data["par_id"].isin(train['par_id'])]
test_df = raw_data[raw_data["par_id"].isin(test['par_id'])]



from Sampling import DataSampling
datasampling = DataSampling()
downsample = datasampling.downsample(train_df)


downsample.reset_index(inplace=True, drop=True)
print(downsample.label.value_counts())


from Analyzer import BertAnalyzer
from PreTrainedBert import model

scheduler_types = ["linear_schedule_with_warmup", "cosine_schedule_with_warmup", "exponential_schedule_with_warmup"]
results = {}

for scheduler_type in scheduler_types:
    xlnet_model = model("xlnet-base-cased")
    BertModel = BertAnalyzer(xlnet_model, 
                            batch_size=64,
                            max_seq_len=128,
                            epochs=3,
                            lr=4e-5)

    BertModel.train(downsample, None)

print(f"scheduler_type={scheduler_type}")