from transformers import BertForSequenceClassification

class ModifiedBertForSequenceClassification(BertForSequenceClassification):
    def _get_no_split_modules(self, device_map: str):
        return self.bert._get_no_split_modules(device_map=device_map)
