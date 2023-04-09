from collections import defaultdict, Counter
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer

from src.a1_code.a1_processing import convert_sst_label, get_batch_token_ids, BertClassifierModule
from src.cs224u_original.torch_shallow_neural_classifier import TorchShallowNeuralClassifier

# data
dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all')
dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all')
sst = load_dataset("SetFit/sst5")
for splitname in ('train', 'validation', 'test'):
    dist = [convert_sst_label(s) for s in sst[splitname]['label_text']]
    sst[splitname] = sst[splitname].add_column('gold_label', dist)
    sst[splitname] = sst[splitname].add_column('sentence', sst[splitname]['text'])

# model
class BertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']

    def build_graph(self):
        return BertClassifierModule(
            self.n_classes_, self.hidden_activation, self.weights_name)

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)
        return dataset

def train():
    # configure params here
    weights_name = "prajjwal1/bert-mini"

    # set params
    bert_finetune = BertClassifier(
    weights_name=weights_name,
    hidden_activation=nn.ReLU(),
    eta=0.00005,          # Low learning rate for effective fine-tuning.
    batch_size=8,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5)   # params.

    # combine data
    X_train = dynasent_r1['train']['sentence'] + \
                dynasent_r2['train']['sentence'] + \
                sst['train']['sentence']

    Y_train = dynasent_r1['train']['gold_label'] + \
                dynasent_r2['train']['gold_label'] + \
                sst['train']['gold_label']

    # finetune on comb data
    _ = bert_finetune.fit(
        X_train,
        Y_train)

    # save best model
    fn = "a1_bakeoff_" + weights_name
    torch.save(bert_finetune.model, fn + ".pt")

train()