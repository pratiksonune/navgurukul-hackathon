import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

class ONNXLightweightSummarizer:
    def __init__(self, onnx_path, model_name, max_length=512):
        self.max_length = max_length
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

    def summarize(self, text, num_sentences=3):
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= num_sentences:
            return text

        inputs = self.tokenizer(
            sentences,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Convert to int64 for ONNX
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        scores = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )[0]

        # Handle different output shapes
        if len(scores.shape) > 1:
            scores = scores.squeeze(-1)

        top_idx = np.argsort(scores)[-num_sentences:]
        top_idx = sorted(top_idx)

        return ". ".join(sentences[i] for i in top_idx) + "."
    
    def get_model_size(self):
        """Calculate ONNX model file size in MB"""
        model_path = Path(self.onnx_path)
        if model_path.exists():
            size_bytes = model_path.stat().st_size
            size_mb = size_bytes / (1024 ** 2)
            return size_mb
        else:
            return 0.0