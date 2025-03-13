import torch

from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertAttention, BertSelfOutput, BertIntermediate, BertOutput, BertLMPredictionHead, BertPredictionHeadTransform, BertSdpaSelfAttention, BertSelfAttention, BertOnlyMLMHead
from transformers.models.bert.configuration_bert import BertConfig

class TacitBertConfig(BertConfig):
    def __init__(self):
        super().__init__()


class TacitBert(BertForMaskedLM):
    def __init__(self, config: TacitBertConfig):
        super().__init__(config)

