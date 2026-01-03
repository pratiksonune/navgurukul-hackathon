import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
import os

class LightweightSummarizer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", max_length=512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        
        # Load base model
        base_model = DistilBertModel.from_pretrained(model_name)
        
        self.encoder = base_model
        self.encoder.transformer.layer = self.encoder.transformer.layer[:3]
        
        # Summarization head
        self.summarization_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Get sentence embeddings (mean pooling)
        sentence_embeddings = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        # Score sentences
        scores = self.summarization_head(sentence_embeddings)
        return scores
    
    def summarize(self, text, num_sentences=3):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Get scores
        with torch.no_grad():
            scores = self.forward(inputs["input_ids"], inputs["attention_mask"])
            scores = scores.squeeze(-1).numpy()
        
        # Select top sentences
        import numpy as np
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        summary = ". ".join([sentences[i] for i in top_indices]) + "."
        return summary
    
    def get_model_size(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb