# RayWhisper2

A cross-platform (Windows/macOS) voice-to-text tool with RAG-enhanced transcription using state-of-the-art speech recognition, vector database embeddings, and keyboard-driven interaction.

## Features

- 🎤 **High-Quality Speech Recognition**: Uses faster-whisper with support for large-v3 and distil-large-v3 models
- 🧠 **RAG-Enhanced Transcription**: Retrieves relevant context from your documents to improve transcription accuracy
- ⌨️ **Keyboard-Driven**: Global hotkeys for hands-free operation
- 🚀 **Fast & Efficient**: Optimized with INT8/FP16 quantization for GPU acceleration
- 🔄 **Cross-Platform**: Works on both Windows and macOS (not tested on macOS yet, but Claude promises it works 😄)
- 📝 **Smart Output**: Types transcribed text directly into any application

## Architecture

RayWhisper2 follows Clean Architecture principles with clear separation of concerns:

- **Domain Layer**: Core business logic (entities, value objects, interfaces)
- **Application Layer**: Use cases and application services
- **Infrastructure Layer**: External implementations (Whisper, ChromaDB, audio, keyboard)
- **Presentation Layer**: CLI and user interface

## Installation

### Prerequisites

- Python 3.11 or higher
- **For GPU acceleration:** 
  - NVIDIA GPU with CUDA 12 and cuDNN 9 support
  - If you don't have a compatible NVIDIA GPU, you must use CPU mode (see Configuration below)

### Install from source

```bash
# Clone the repository
git clone https://github.com/Fredrik-C/RayWhisper2.git
cd RayWhisper2

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Configuration

Copy the example configuration files:

```bash
cp .env.example .env
cp config/config.example.yaml config/config.yaml
```

Edit `.env` or `config/config.yaml` to customize settings:

- **Whisper Model**: Default is Systran/faster-whisper-medium.en (or use: tiny, base, small, medium, large-v2, large-v3, distil-large-v3)
- **Device**: Default is cuda (change to cpu if you don't have CUDA 12/cuDNN 9)
- **Compute Type**: Default is float16 (change to int8 for CPU mode)
- **Embedding Model**: Default is BAAI/bge-base-en-v1.5
- **Hotkeys**: Default is ctrl+space (customize as needed)

**⚠️ Important - CUDA 12 and cuDNN 9 Required for GPU Mode:**
- The default configuration uses GPU mode (`device: "cuda"`)
- **GPU mode requires CUDA 12 and cuDNN 9 with a compatible NVIDIA GPU**
- **If you don't have CUDA 12/cuDNN 9**, you must change to CPU configuration:

```yaml
whisper:
  model_size: "base"      # or "small" for better accuracy
  device: "cpu"           # REQUIRED if no CUDA 12/cuDNN 9
  compute_type: "int8"    # int8 is fastest on CPU
```

## Usage

### Populate the Vector Database

Before using RAG-enhanced transcription, populate the vector database with your documents:

```bash
raywhisper populate ./docs --clear
```

This will parse and embed all Markdown files from the specified directories.

### Run the Application

```bash
raywhisper run
```

**Enable Caps Lock** to start recording, **disable Caps Lock** to stop and transcribe. The transcribed text will be typed into the active application.

> **Note:** The Caps Lock LED provides visual feedback of the recording state - when the LED is on, you're recording.

## Development

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=raywhisper --cov-report=html

# Run specific test
pytest tests/unit/domain/test_transcription.py -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
ruff format src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
raywhisper2/
├── src/raywhisper/          # Source code
│   ├── domain/              # Domain layer
│   ├── application/         # Application layer
│   ├── infrastructure/      # Infrastructure layer
│   ├── presentation/        # Presentation layer
│   └── config/              # Configuration
├── tests/                   # Tests
├── config/                  # Configuration files
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## Technology Stack

- **Speech-to-Text**: faster-whisper (CTranslate2)
- **Vector Database**: ChromaDB
- **Embeddings**: BAAI/bge-small-en-v1.5 (sentence-transformers)
- **Reranking**: BAAI/bge-reranker-v2-m3
- **Audio**: sounddevice
- **Keyboard**: pynput
- **Configuration**: pydantic + pydantic-settings

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome!

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient Whisper inference
- [ChromaDB](https://www.trychroma.com/) for vector database
- [BAAI](https://huggingface.co/BAAI) for BGE embeddings and reranker models

