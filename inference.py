# inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class ReviewClassifier:
    def __init__(self, model_path="./final_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_names = {0: "GENUINE", 1: "PROMOTION", 2: "IRRELEVANT", 3: "RANT"}
        
    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return self.label_names[predictions.item()]

# Example usage
if __name__ == "__main__":
    classifier = ReviewClassifier()
    test_text = "Amazing service and delicious food! Highly recommend."
    prediction = classifier.predict(test_text)
    print(f"Prediction: {prediction}")