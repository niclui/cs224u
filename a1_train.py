from src.a1_code.args import get_train_test_args
from collections import defaultdict, Counter
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer
import wandb
import os
from sklearn.metrics import classification_report
import json

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

# load prompting data
easy_zero = open("datasets/combined_zeroshot.json")
hard_zero = open("datasets/combined_hard_zeroshot.json")
fewshot = open("datasets/combined_fewshot.json")

# model
class BertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, model=None, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']
        self.model = model

    def build_graph(self):
        if self.model is None:
            return BertClassifierModule(
                self.n_classes_, self.hidden_activation, self.weights_name)
        else:
            return self.model

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
            self.classes_ = ['negative', 'neutral', 'positive']
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
    # initialize wandb
    wandb.init(entity=os.getenv("helengu"), 
               project=os.getenv("bakeoff1-nlu"))
    
    # save dir
    save_dir = "checkpoints/a1_bakeoff/"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
    save_name = args.model.split('/')[-1]
    save_name += "_" + args.prompting
    print(save_name)
    save_path = os.path.join(save_dir, save_name)
    
    # set params
    bert_finetune = BertClassifier(
    weights_name=args.model,
    hidden_activation=nn.ReLU(),
    eta=args.lr,          # Low learning rate for effective fine-tuning.
    batch_size=16,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 64.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5,
    save_path=save_path) # the checkpoint path

    if args.prompting == "r2":
        print("r2")
        X_train = dynasent_r2["train"]["sentence"][:500]
        Y_train = dynasent_r2["train"]['gold_label'][:500]
    else:
        if args.prompting == "easy_zero":
            dataset = easy_zero
            print("easy zero")
        elif args.prompting == "hard_zero":
            dataset = hard_zero
            print("hard zero")
        elif args.prompting == "fewshot":
            dataset = fewshot
            print("fewshot")

        X_train = []
        Y_train = []
        dataset = json.load(dataset)

        if len(dataset)>500:
            dataset=dataset[:500]
        
        for data in dataset:
            X_train.append(data['sentence'])
            Y_train.append(data['label'])

    # X_train = dataset['sentence']

    # Y_train = dataset['label']

    # finetune on comb data
    _ = bert_finetune.fit(
        X_train,
        Y_train)

    # save best model
    save_path = os.path.join(save_dir, save_name)
    torch.save(bert_finetune.model, save_path + ".pt")
    
def predict():
    finetuned = BertClassifier(
    weights_name=args.model,
    model = torch.load(args.checkpoint),
    hidden_activation=nn.ReLU(),
    eta=args.lr,          # Low learning rate for effective fine-tuning.
    batch_size=8,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5)   # params.

    bakeoff_df = pd.read_csv(args.val_data)
    preds = finetuned.predict(bakeoff_df['sentence'].tolist())
    bakeoff_df['prediction'] = preds 
    bakeoff_df.to_csv(args.pred_file, index=False)
    
def eval():

    finetuned = BertClassifier(
    weights_name=args.model,
    model = torch.load(args.checkpoint),
    hidden_activation=nn.ReLU(),
    eta=args.lr,          # Low learning rate for effective fine-tuning.
    batch_size=8,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5)   # params.

    for val_set in [dynasent_r2, dynasent_r1, dynasent_r1, sst]:
        preds = finetuned.predict(val_set['validation']['sentence'])
        print(classification_report(val_set['validation']['gold_label'], preds, digits=3))


# python a1_train.py --model "roberta-base"
if __name__ == '__main__':
    args = get_train_test_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()
    elif args.mode == 'eval':
        eval()

# train("prajjwal1/bert-mini")
# train("roberta-base")
# train("microsoft/deberta-v3-base")
