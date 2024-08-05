import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

class GenderBiasAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def get_masked_probability(self, sentence, target_word):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        softmax = torch.nn.functional.softmax(logits, dim=-1)
        mask_token_logits = logits[0, mask_token_index, :]
        target_id = self.tokenizer.convert_tokens_to_ids(target_word)
        return softmax[0, mask_token_index, target_id].item()

    def calculate_association(self, sentence, target_word, attribute_word):
        # Calculate target probability
        masked_sentence = sentence.replace(target_word, self.tokenizer.mask_token)
        target_prob = self.get_masked_probability(masked_sentence, target_word)
        
        # Calculate prior probability
        masked_sentence = masked_sentence.replace(attribute_word, self.tokenizer.mask_token)
        prior_prob = self.get_masked_probability(masked_sentence, target_word)
        
        # Calculate association
        association = np.log(target_prob / prior_prob)
        return association

    def analyze_bias(self, templates, targets, attributes):
        results = {}
        for template in templates:
            for target in targets:
                for attribute in attributes:
                    sentence = template.format(target=target, attribute=attribute)
                    association = self.calculate_association(sentence, target, attribute)
                    results[(target, attribute)] = association
        return results

# Usage example
analyzer = GenderBiasAnalyzer()

templates = [
    "The {target} is a {attribute}.",
    "{target} works as a {attribute}."
]

targets = ["he", "she"]
attributes = ["doctor", "nurse", "engineer", "teacher"]

results = analyzer.analyze_bias(templates, targets, attributes)

for (target, attribute), association in results.items():
    print(f"Association between {target} and {attribute}: {association}")