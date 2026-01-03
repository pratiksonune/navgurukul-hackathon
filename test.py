import argparse
from pathlib import Path
import torch
import time

from models.summarizer import LightweightSummarizer
from models.model_converter import ModelConverter
from models.onnx_summarizer import ONNXLightweightSummarizer
from models.filler_generator import FillerGenerator


def main():
    parser = argparse.ArgumentParser(description="Voice Interview AI System")
    parser.add_argument(
        "--mode",
        choices=["convert", "demo", "test"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument("--output-dir", type=str, default="./output")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "convert":
        print("Converting summarizer to ONNX...")
        print("Loading PyTorch model...")
        
        model = LightweightSummarizer()
        model.eval()

        onnx_path = output_dir / "summarizer.onnx"
        optimized_path = output_dir / "summarizer_int8.onnx"

        print(f"Converting to ONNX: {onnx_path}")
        ModelConverter.convert_to_onnx(
            model,
            model.tokenizer,
            str(onnx_path)
        )

        print(f"Optimizing ONNX model: {optimized_path}")
        ModelConverter.optimize_onnx(
            str(onnx_path),
            str(optimized_path)
        )

        print("ONNX conversion completed successfully")

    elif args.mode == "test":
        print("Testing PyTorch model...")
        
        model = LightweightSummarizer()
        model.eval()
        
        test_text = """
        I built a web application using React and Node.js that allows users
        to upload images and apply various filters. The backend uses Express
        and processes images with Sharp library. I implemented user authentication
        with JWT tokens and stored data in MongoDB. The biggest challenge was
        optimizing image processing for large files, which I solved by implementing
        a queue system with Bull.
        """
        
        print(f"\nModel size: {model.get_model_size():.2f} MB")
        
        start = time.perf_counter()
        summary = model.summarize(test_text)
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"\nInference time: {inference_time:.2f} ms")
        print(f"\nOriginal text ({len(test_text.split())} words):")
        print(test_text.strip())
        print(f"\nSummary ({len(summary.split())} words):")
        print(summary)
        
        generator = FillerGenerator()
        filler = generator.generate(test_text)
        print(f"\nGenerated filler: {filler}")

    elif args.mode == "demo":
        print("Running ONNX demo...")

        onnx_path = output_dir / "summarizer_int8.onnx"
        if not onnx_path.exists():
            print(f"ONNX model not found at {onnx_path}")
            print("Run with --mode convert first to create the ONNX model")
            return

        print(f"Loading ONNX model from {onnx_path}")
        summarizer = ONNXLightweightSummarizer(
            onnx_path=str(onnx_path),
            model_name="distilbert-base-uncased"
        )

        test_text = """
        I built a web application using React and Node.js that allows users
        to upload images and apply various filters. The backend uses Express
        and processes images with Sharp library. I implemented user authentication
        with JWT tokens and stored data in MongoDB. The biggest challenge was
        optimizing image processing for large files, which I solved by implementing
        a queue system with Bull.
        """

        print(f"\nOriginal text ({len(test_text.split())} words):")
        print(test_text.strip())
        
        start = time.perf_counter()
        summary = summarizer.summarize(test_text)
        inference_time = (time.perf_counter() - start) * 1000
        model_size = summarizer.get_model_size()

        print(f"=============================Model size: {model_size:.2f} MB")
        print(f"\n=============================Inference time: {inference_time:.2f} ms")
        print(f"\n=============================Summary ({len(summary.split())} words):")
        print(summary)

        generator = FillerGenerator()
        filler = generator.generate(test_text)
        detected_context = generator.detect_context(test_text)
        should_pause = generator.detect_pause(test_text)
        
        print(f"\nGenerated filler: {filler}")
        print(f"Detected context: {detected_context}")
        print(f"Should pause: {should_pause}")

        print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()