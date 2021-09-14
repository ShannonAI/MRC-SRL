import torch
import torch.nn as nn
from transformers import AutoModel


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size+2)
        self.hidden_size = self.bert.config.hidden_size
        num_labels = 28 if config.dataset_tag == 'conll2012' else 20
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.dropout = nn.Dropout(p=config.dropout)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, target=None):
        _, rep = self.bert(input_ids, attention_mask, return_dict=False)
        rep = self.dropout(rep)
        rep = self.linear(rep)
        if target is not None:
            loss = self.loss_func(rep.view(-1), target.view(-1))
            return loss
        else:
            return torch.sigmoid(rep).cpu()
