import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from transformers import DistilBertModel, DistilBertTokenizerFast


class ModelConverter:
    @staticmethod
    def convert_to_onnx(model, tokenizer, output_path, max_length=512):
        model.eval()
        model.cpu()

        dummy_text = "This is a sample sentence for conversion."

        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "output": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=True
            )

        print(f"Model exported to {output_path}")

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully")

        return output_path

    @staticmethod
    def optimize_onnx(input_path, output_path):
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8
        )

        original_size = Path(input_path).stat().st_size / (1024 ** 2)
        optimized_size = Path(output_path).stat().st_size / (1024 ** 2)

        print(f"Optimized model saved to {output_path}")
        print(f"Original size: {original_size:.2f} MB")
        print(f"Optimized size: {optimized_size:.2f} MB")
        print(f"Reduction: {(1 - optimized_size / original_size) * 100:.1f}%")

        return output_path

    @staticmethod
    def test_onnx_inference(model_path, tokenizer, text):
        import time
        import numpy as np

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        inputs = tokenizer(
            text,
            return_tensors="np",
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        # Convert to int64 for ONNX
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # warm-up
        session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        times = []
        for _ in range(10):
            start = time.perf_counter()
            outputs = session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )
            times.append((time.perf_counter() - start) * 1000)

        print("\nInference results:")
        print(f"Avg time: {sum(times)/len(times):.2f} ms")
        print(f"Output shape: {outputs[0].shape}")

        return outputs[0]