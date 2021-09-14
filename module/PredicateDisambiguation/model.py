import torch
import torch.nn as nn
from transformers import AutoModel


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.bert = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size+2)
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, target=None):
        """
        Args:
            input_ids: (batch,seq_len)
            token_type_ids： (batch,seq_len)
            attention_mask: (batch,seq_len)
            target: (batch,)
        """
        if 'roberta' in self.bert.config._name_or_path:
            _, rep = self.bert(input_ids, attention_mask, return_dict=False)
        else:
            _, rep = self.bert(input_ids, attention_mask,token_type_ids, return_dict=False)
        rep = self.dropout(rep)
        rep = self.linear(rep)  # (batch,1)
        if target is not None:
            loss = self.loss_func(rep.view(-1), target.view(-1))
            return loss
        else:
            predict_prob = torch.sigmoid(rep)
            return predict_prob.view(-1).cpu()
