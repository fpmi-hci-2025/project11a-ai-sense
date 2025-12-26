# AI Sense - Intelligent Text Composition Service

![Project Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

AI Sense is a high-performance AI-powered text composition service that generates thoughtful, engaging content including articles, social media posts, and creative writing. Built with C++ for optimal inference performance and featuring both REST API and Docker deployment options.

## 🌟 Key Features

- **High-Performance Inference**: C++ backend powered by ONNX Runtime for lightning-fast text generation
- **RESTful API**: Clean, well-documented API endpoints for easy integration
- **Content Generation**: Generate articles, social media posts, and creative content
- **Docker Ready**: Containerized deployment with multi-stage builds
- **Multiple Implementations**: Both C++ and Python (FastAPI) versions available
- **Extensible Architecture**: Modular design for easy feature additions

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd project11-ai-sense

# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:7070
```

### Manual Build (C++ Version)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake wget git ca-certificates libgomp1 libprotobuf-dev protobuf-compiler

# Download ONNX Runtime
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz

# Build SentencePiece
git clone --depth 1 https://github.com/google/sentencepiece.git
cd sentencepiece && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && sudo make install

# Build the project
cd ../../../
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.18.0
cmake --build . --config Release -j$(nproc)

# Run the server
./server
```

### Python Version (FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python src/main.py
```

## 📋 API Endpoints

### Health Check
```http
GET /api/health
```
Returns server status.

### Text Composition
```http
POST /api/compose
Content-Type: application/json

{
  "query": "Compose an article about the benefits of morning walks",
  "metadata": {
    "language": "english",
    "tone": "inspirational",
    "length": 500
  }
}
```

**Response:**
```json
{
  "response": "Morning walks aren't just exercise—they're meditation in motion..."
}
```

### Content Recommendations (Coming Soon)
```http
POST /api/recommend
Content-Type: application/json

{
  "id": "user_12345"
}
```

## 🏗️ Architecture

### Core Components

#### C++ Backend (`src/main.cpp`)
- **ONNX Runtime Integration**: High-performance neural network inference
- **SentencePiece Tokenizer**: Efficient text tokenization
- **HTTP Server**: Built-in REST API server using cpp-httplib
- **Text Generation Pipeline**: Complete inference pipeline with sampling and post-processing

#### Python API (`src/main.py`)
- **FastAPI Framework**: Modern, fast web framework
- **Pydantic Models**: Type-safe request/response validation
- **Uvicorn Server**: ASGI server for high-performance deployment

#### Model Architecture
- **Nexus Model**: Custom transformer-based language model
- **ONNX Format**: Optimized for cross-platform inference
- **SentencePiece**: Subword tokenization for efficient text processing

### Project Structure
```
├── src/
│   ├── main.cpp          # C++ inference server
│   └── main.py           # Python FastAPI server
├── storage/
│   ├── nexus.onnx        # ONNX model file
│   ├── nexus.onnx.data   # Model weights
│   ├── tokenizer.model   # SentencePiece tokenizer
│   └── dataset.json      # Training data sample
├── docs/
│   └── openapi.json      # API specification
├── Dockerfile            # Multi-stage build
├── docker-compose.yml    # Container orchestration
├── CMakeLists.txt        # C++ build configuration
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_ROOT` | Path to ONNX Runtime installation | Required for C++ build |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `7070` (C++), `8000` (Python) |

### Model Parameters

The inference engine supports configurable parameters:

- **Temperature**: Controls randomness (0.1-2.0)
- **Top-K**: Limits vocabulary sampling (1-50)
- **Max Context**: Maximum token context length (512-2048)
- **Model Path**: Custom model file location

## 📊 Model Training Data

The model is trained on a diverse dataset including:

- **Social Media Posts**: Concise, engaging content for platforms
- **Articles**: Thoughtful pieces on various topics
- **Creative Writing**: Short stories and creative content
- **Informational Content**: How-to guides and explanations

Sample data includes topics like:
- Nature and outdoor activities
- Personal development and wellness
- Technology and innovation
- Social issues and community
- Creative expression and arts

## 🐳 Docker Deployment

### Multi-Stage Build Strategy

1. **Builder Stage**: Compiles C++ dependencies and ONNX Runtime
2. **Runtime Stage**: Minimal Ubuntu image with compiled binaries
3. **Model Assets**: Includes pre-trained model and tokenizer

### Docker Compose Features

- **Volume Mounting**: Live code reloading for development
- **Port Mapping**: Automatic port forwarding
- **Restart Policy**: Automatic container recovery
- **Resource Limits**: Configurable CPU/memory constraints

## 🔍 API Documentation

### Request/Response Formats

#### Compose Request
```typescript
interface ComposeRequest {
  query: string;           // User's composition request
  metadata?: {            // Optional parameters
    language?: string;    // Target language
    tone?: string;        // Writing tone
    length?: number;      // Approximate length
    [key: string]: any;   // Additional metadata
  };
}
```

#### Error Handling
```typescript
interface ValidationError {
  detail: Array<{
    loc: string[];        // Error location
    msg: string;          // Error message
    type: string;         // Error type
  }>;
}
```

### Status Codes

- `200`: Success
- `422`: Validation error (invalid request format)
- `500`: Internal server error

## 🚦 Performance Benchmarks

### Inference Performance (C++ Backend)
- **Model Load Time**: ~2-3 seconds
- **Token Generation**: ~50-100ms per token
- **Memory Usage**: ~500MB base + model size
- **Concurrent Requests**: 10-50 simultaneous users

### API Response Times
- **Health Check**: <1ms
- **Short Generation**: 200-500ms
- **Long Generation**: 2-5 seconds

## 🔒 Security Considerations

- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Built-in protection against abuse
- **Container Security**: Minimal attack surface in Docker
- **Model Safety**: Filtered training data and output validation

## 🧪 Testing

### API Testing
```bash
# Health check
curl http://localhost:7070/api/health

# Text composition
curl -X POST http://localhost:7070/api/compose \
  -H "Content-Type: application/json" \
  -d '{"query": "Write about nature", "metadata": {}}'
```

### Load Testing
```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Test API performance
hey -n 1000 -c 10 http://localhost:7070/api/health
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ONNX Runtime**: High-performance ML inference
- **SentencePiece**: Efficient tokenization
- **cpp-httplib**: Lightweight HTTP server
- **FastAPI**: Modern Python web framework
- **Nlohmann JSON**: C++ JSON library

## 📞 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Documentation**: Comprehensive API docs and examples

---

**AI Sense** - Transforming ideas into words with the power of AI.
