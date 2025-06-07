from transformers import BertForSequenceClassification

class ModifiedBertForSequenceClassification(BertForSequenceClassification):
    """
    Subclass of BertForSequenceClassification to override _get_no_split_modules.

    This method controls which parts of the model should not be split across devices
    when using model parallelism or device mapping. Here, it delegates directly
    to the underlying BERT model’s method.
    """
    def _get_no_split_modules(self, device_map: str):
        """
        Return modules that must stay on the same device (not split).

        Args:
            device_map (str): The device map string specifying device placement.

        Returns:
            list or set: Modules names that should not be split.
        """
        return self.bert._get_no_split_modules(device_map=device_map)
