from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.onnx_summarizer import ONNXLightweightSummarizer
from models.filler_generator import FillerGenerator

app = Flask(__name__)
CORS(app)

summarizer = None
filler_generator = None
model_load_time = 0
model_size = 0

def initialize_models():
    global summarizer, filler_generator, model_load_time, model_size
    
    print("Initializing models...")
    start_time = time.time()
    
    # Initialize ONNX summarizer
    onnx_path = Path(__file__).parent.parent / "output" / "summarizer_int8.onnx"
    
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. "
            "Run 'python test.py --mode convert'"
        )
    
    summarizer = ONNXLightweightSummarizer(
        onnx_path=str(onnx_path),
        model_name="distilbert-base-uncased"
    )
    
    filler_generator = FillerGenerator()
    
    model_load_time = (time.time() - start_time) * 1000
    model_size = summarizer.get_model_size()
    
    print(f"Models loaded in {model_load_time:.2f}ms")
    print(f"Model size: {model_size:.2f}MB")
    
    return model_load_time, model_size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        start_time = time.time()
        summary = summarizer.summarize(text)
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'summary': summary,
            'inference_time': inference_time,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_filler', methods=['POST'])
def generate_filler():
    try:
        data = request.json
        text = data.get('text', '')
        context = data.get('context', None)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        filler = filler_generator.generate(text, context)
        should_pause = filler_generator.detect_pause(text)
        
        return jsonify({
            'filler': filler,
            'should_pause': should_pause,
            'detected_context': filler_generator.detect_context(text)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_conversation', methods=['POST'])
def process_conversation():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        start_summary = time.time()
        summary = summarizer.summarize(text)
        summary_time = (time.time() - start_summary) * 1000
        
        filler = filler_generator.generate(text)
        should_pause = filler_generator.detect_pause(text)
        
        return jsonify({
            'summary': summary,
            'filler': filler,
            'should_pause': should_pause,
            'metrics': {
                'summary_inference_time': summary_time,
                'total_processing_time': summary_time
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    try:
        return jsonify({
            'model_size_mb': summarizer.get_model_size(),
            'model_load_time_ms': model_load_time,
            'max_length': summarizer.max_length,
            'model_type': 'ONNXLightweightSummarizer',
            'status': 'loaded'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_time, size = initialize_models()
    app.run(host='0.0.0.0', port=5008)