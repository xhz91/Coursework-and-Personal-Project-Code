import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import BertConfig, BertModel
from transformers import AutoConfig, AutoModel
#from BertModel.PreProcessing import TRAINED_WEIGHTS
HIDDEN_OUTPUT_FEATURES = 768
class model(nn.Module):
    def __init__(self, pretrained_weight):
        super(model, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_weight)
        self.bert_base = AutoModel.from_pretrained(pretrained_weight,
                                                   config = config)
        self.fc = nn.Linear(HIDDEN_OUTPUT_FEATURES, 1)

    def forward(self, input_ids, attn_mask):
        outputs = self.bert_base(input_ids = input_ids,
                                 attention_mask=attn_mask)
        bert_output = outputs.last_hidden_state
        pooled_output = torch.max(bert_output, dim = 1)[0]
        x = self.fc(pooled_output)
        return x