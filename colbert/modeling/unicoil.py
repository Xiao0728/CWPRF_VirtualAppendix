import torch 
from torch import nn
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import PreTrainedModel, BertModel, BertTokenizer, BertConfig
import math

class UniCOILEncoder(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = 'coil_encoder'
    load_tf_weights = None
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config 
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)

    def _init_weights(self, module):
            """ Initialize the weights (needed this for the inherited from_pretrained method to work) """
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self._init_weights)

    def forward(self, **kargs):
        outputs = self.bert(**kargs)
        sequence_output = outputs.last_hidden_state 
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return tok_weights