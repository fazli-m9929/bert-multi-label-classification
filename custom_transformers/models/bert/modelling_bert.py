# from transformers.models.bert.modeling_bert import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     BERT_START_DOCSTRING,
#     BERT_INPUTS_DOCSTRING,
#     BertPreTrainedModel,
#     BertModel,
#     BertConfig,
#     BertForSequenceClassification
# )
# import torch.nn as nn
# from torch.nn import BCEWithLogitsLoss
# import torch
# from ...modelling_outputs import MultiLabelClassifierOutput
# from typing import Optional, Union, Tuple
from transformers import BertForSequenceClassification


class ModifiedBertForSequenceClassification(BertForSequenceClassification):
    def _get_no_split_modules(self, device_map: str):
        return self.bert._get_no_split_modules(device_map=device_map)

# @add_start_docstrings(
#     """
#     Bert Model transformer with a sequence classification head on top (a linear layer on top of the pooled
#     output) e.g. for GLUE tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class BertForMultiLabelClassification(BertPreTrainedModel):
#     def __init__(self, config: BertConfig):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
#         self.config.problem_type = 'multi_label_classification'
#         self.bert = BertModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.sigmoid = nn.Sigmoid()

#         self.loss_fct = BCEWithLogitsLoss()

#         self.post_init()

#     def _get_no_split_modules(self, device_map: str):
#         return self.bert._get_no_split_modules(device_map=device_map)

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], MultiLabelClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs.pooler_output

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         probabilities = self.sigmoid(logits)

#         loss = None
#         if labels is not None:
#             loss = self.loss_fct(logits, labels)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return MultiLabelClassifierOutput(
#             loss=loss,
#             logits=logits,
#             probabilities=probabilities,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )