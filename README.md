# AI Voice Interview Assistant

Ultra-lightweight client-side AI system for real-time voice interview summaries with contextual filler phrases.


## Project Structure

```
navgurukul-hackathon/
├── models/                          # AI model modules
│   ├── __init__.py
│   ├── summarizer.py                # PyTorch summarization model
│   ├── onnx_summarizer.py           # ONNX inference wrapper
│   ├── filler_generator.py          # Contextual filler generator
│   └── model_converter.py           # PyTorch to ONNX converter
│
├── web/                             # Web application
│   ├── app.py                       # Flask backend server
│   └── templates/
│       └── index.html               # Frontend HTML/CSS/JavaScript
│
├── output/                          # Generated models (created on first run)
│   ├── summarizer.onnx              # Original ONNX model
│   └── summarizer_int8.onnx         # Quantized ONNX model
│
├── test.py                          # Testing and conversion script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Features

### Core Capabilities
- Real-time voice-to-text transcription with live streaming
- Extractive text summarization (less than 50ms inference)
- Context-aware filler phrase generation
- Automatic pause detection for natural conversation flow
- Offline-capable after initial setup
- Complete conversation logging and export

### Technical Highlights
- Model size: Approximately 25MB (under 30MB requirement)
- Inference latency: 35-45ms (under 50ms requirement)
- Load time: 400-500ms
- ONNX Runtime for optimized inference
- DistilBERT-based architecture (3 layers)
- INT8 quantization for size reduction

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- 500MB free disk space

### Software Requirements
- Chrome or Edge browser (for Web Speech API support)
- Microphone access
- Internet connection (only for initial model download)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pratiksonune/navgurukul-hackathon.git
cd navgurukul-hackathon
```

### Step 2: Create Virtual Environment

```bash
uv venv ThoR --python 3.11
source ThoR/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- torch (PyTorch for model operations)
- transformers (Hugging Face transformers)
- onnx (ONNX model format)
- onnxruntime (ONNX inference engine)
- flask (Web server)
- flask-cors (CORS support)
- numpy (Numerical operations)

Installation may take 5-10 minutes depending on your internet speed.

## Quick Start

### Step 1: Convert Model to ONNX

Convert the PyTorch model to optimized ONNX format:

```bash
python test.py --mode convert
```

**Expected Output:**
```
Converting to ONNX: output/summarizer.onnx
Some weights of DistilBertModel were not initialized from the model checkpoint...
Model exported to output/summarizer.onnx
ONNX model verified successfully
Optimizing ONNX model: output/summarizer_int8.onnx
Optimized model saved to output/summarizer_int8.onnx
Original size: 42.15 MB
Optimized size: 24.89 MB
Reduction: 40.9%
ONNX conversion completed successfully
```

This creates two files in the `output/` directory:
- `summarizer.onnx` - Original ONNX model (approximately 42MB)
- `summarizer_int8.onnx` - Quantized model (approximately 25MB)


### Step 2: Test the Model

Verify the ONNX model works correctly:

```bash
python test.py --mode demo
```

**Expected Output:**
```
Running ONNX demo...
Loading ONNX model from output/summarizer_int8.onnx

Original text (67 words):
I built a web application using React and Node.js...

Inference time: 42.35 ms

Summary (34 words):
I built a web application using React and Node.js that allows users
to upload images and apply various filters. I implemented user authentication
with JWT tokens and stored data in MongoDB.

Generated filler: That's a solid technical approach...
Detected context: technical
Should pause: True

Demo completed successfully!
```

### Step 3: Start the Web Application

Launch the Flask server:

```bash
python web/app.py
```

**Expected Output:**
```
Initializing models...
Models loaded in 450.23ms
Model size: 24.89MB

Server ready!
Model Load Time: 450.23ms
Model Size: 24.89MB

Open http://localhost:5008 in your browser
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5008
 * Running on http://192.168.1.100:5008
```

### Step 4: Open in Browser

1. Open Chrome or Edge browser
2. Navigate to `http://localhost:5008`
3. You should see the AI Voice Interview Assistant interface

## Usage

### Starting an Interview Session

1. **Click "Start Interview" button**
   - Browser will request microphone permissions
   - Grant permissions when prompted

2. **Begin Speaking**
   - Speak clearly into your microphone
   - You'll see text appear in real-time in the "Live Transcript" section
   - Streaming text appears in purple/italic
   - Final text turns black when confirmed

3. **AI Processing**
   - After you pause (1 second of silence), AI processes your speech
   - Summary appears in "Real-time Summary" section
   - Contextual fillers may be generated and spoken back
   - All interactions logged in "Conversation Log"

4. **Stop Recording**
   - Click "Stop Interview" when done
   - All data remains available for review

5. **Download Report**
   - Click "Download Report" to export conversation data
   - Downloads JSON file with complete transcript and metadata

### Understanding the Interface

#### Metrics Dashboard
- **Model Load Time**: Time taken to load the AI model
- **Inference Time**: Time for AI to process and summarize text
- **Model Size**: Size of the ONNX model file
- **Status**: Current recording status (Ready/Recording)

#### Content Panels
- **Live Transcript (Streaming)**: Real-time speech-to-text with streaming display
- **Real-time Summary**: AI-generated extractive summary
- **Conversation Log**: Complete conversation history with timestamps
- **Generated Fillers**: Contextual phrases inserted by AI

### Tips for Best Results

1. **Speak clearly and at moderate pace**
2. **Pause between thoughts** (allows AI to process)
3. **Use good microphone quality**
4. **Minimize background noise**
5. **Speak in complete sentences** for better summaries

## API Documentation

### Base URL
```
http://localhost:5008/api
```

### Endpoints

#### 1. Get Model Information

```http
GET /api/model_info
```

**Response:**
```json
{
  "model_size_mb": 24.89,
  "model_load_time_ms": 450.23,
  "max_length": 512,
  "model_type": "ONNXLightweightSummarizer",
  "status": "loaded"
}
```

#### 2. Summarize Text

```http
POST /api/summarize
Content-Type: application/json

{
  "text": "Your text to summarize here..."
}
```

**Response:**
```json
{
  "summary": "Generated summary text...",
  "inference_time": 42.5,
  "original_length": 150,
  "summary_length": 45
}
```

#### 3. Generate Filler Phrase

```http
POST /api/generate_filler
Content-Type: application/json

{
  "text": "Context text for filler generation...",
  "context": "technical"
}
```

**Context Options:** `acknowledgment`, `thinking`, `clarification`, `technical`, `challenge`

**Response:**
```json
{
  "filler": "That's a solid technical approach...",
  "should_pause": true,
  "detected_context": "technical"
}
```

#### 4. Process Full Conversation

```http
POST /api/process_conversation
Content-Type: application/json

{
  "text": "Complete conversation text..."
}
```

**Response:**
```json
{
  "summary": "Conversation summary...",
  "filler": "Contextual filler phrase...",
  "should_pause": true,
  "metrics": {
    "summary_inference_time": 38.2,
    "total_processing_time": 45.7
  }
}
```
