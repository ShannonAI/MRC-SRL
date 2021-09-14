import torch
import torch.nn as nn
from transformers import AutoModel


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.loss_func = nn.CrossEntropyLoss()
        self.bert = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size+2)
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, 7) 

    def forward(self, input_ids, token_type_ids, attention_mask, target=None):
        """
        Args:
            input_ids: (batch,seq_len)
            token_type_ids： (batch,seq_len)
            attention_mask: (batch,seq_len)
            target: (batch,seq_len)
        """
        if 'roberta' in self.bert.config._name_or_path:
            rep, _ = self.bert(input_ids, attention_mask, return_dict=False)
        else:
            rep, _ = self.bert(input_ids, attention_mask,
                               token_type_ids, return_dict=False)
        rep = self.linear(self.dropout(rep))  # (batch,seq_len,7)
        context_mask = torch.logical_and(
            attention_mask.bool(), token_type_ids == 1)  # (batch,seq_len)
        if target is not None:
            loss = self.loss_func(rep[context_mask], target[context_mask])
            return loss
        else:
            return rep.softmax(dim=-1).cpu(), context_mask.cpu()
