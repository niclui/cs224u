from collections import defaultdict, Counter
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer

# label split
def print_label_dist(dataset, labelname='gold_label', splitnames=('train', 'validation')):
    for splitname in splitnames:
        print(splitname)
        dist = sorted(Counter(dataset[splitname][labelname]).items())
        for k, v in dist:
            print(f"\t{k:>14s}: {v}")

# convert SST split to match dynasent
def convert_sst_label(s):
    return s.split(" ")[-1]

def get_batch_token_ids(batch, tokenizer):
    out = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch, padding='max_length', max_length=512,
                                       truncation=True, return_tensors='pt')
    return {"input_ids": out["input_ids"], "attention_mask":out["attention_mask"]}

class BertClassifierModule(nn.Module):
    def __init__(self, 
            n_classes, 
            hidden_activation, 
            weights_name="prajjwal1/bert-mini"):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = AutoModel.from_pretrained(self.weights_name)
        self.bert.train()
        self.hidden_activation = hidden_activation
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim

        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes)
        )

    def forward(self, indices, mask):
        out = self.bert(indices, mask)
        logits = self.classifier_layer(out["last_hidden_state"][:,0,:])
        return logits
