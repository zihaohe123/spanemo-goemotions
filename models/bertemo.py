import torch
import torch.nn as nn


class BertEMO(nn.Module):
    def __init__(self):
        super(BertEMO, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 27)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        embs = outputs.pooler_output
        logits = self.fc(embs)
        return logits
