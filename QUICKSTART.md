# RayWhisper2 Quick Start Guide

This guide will help you get RayWhisper2 up and running quickly.

## Prerequisites

- Python 3.11 or higher
- **For GPU acceleration:** 
  - NVIDIA GPU with CUDA 12 and cuDNN 9 support
  - **If you don't have CUDA 12/cuDNN 9:** You must configure for CPU mode (see Configuration section)
- Microphone access
- On macOS: Accessibility permissions for keyboard control

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Fredrik-C/RayWhisper2.git
cd RayWhisper2
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

## Configuration

### 1. Create environment file

```bash
cp .env.example .env
```

### 2. Edit config/config.yaml

**⚠️ IMPORTANT - Check Your Configuration:**

The default configuration uses GPU mode. **You MUST change this if you don't have CUDA 12 + cuDNN 9!**

**Default configuration (requires CUDA 12 + cuDNN 9):**
```yaml
whisper:
  model_size: "Systran/faster-whisper-medium.en"
  device: "cuda"          # Requires CUDA 12 + cuDNN 9!
  compute_type: "float16"

keyboard:
  start_stop_hotkey: "super+o"  # Hold the hotkey to record, release to transcribe
```

**If you DON'T have CUDA 12/cuDNN 9, change to CPU mode:**
```yaml
whisper:
  model_size: "base"      # or "small" for better accuracy
  device: "cpu"           # REQUIRED without CUDA 12/cuDNN 9
  compute_type: "int8"    # int8 is fastest on CPU

keyboard:
  start_stop_hotkey: "super+o"  # Hold the hotkey to record, release to transcribe
```

> **Note:** The `start_stop_hotkey` setting is used by default. You can customize it to any supported key combination in `config/config.yaml`.

## Usage

### 1. Populate the Vector Database (Optional but Recommended)

To enable RAG-enhanced transcription, populate the vector database with your documents:

```bash
# Populate from your documentation
raywhisper populate ./docs2ingest --clear
```

This will:
- Parse all Markdown (.md) files
- Generate embeddings
- Store them in the vector database

### 2. Run the Application

```bash
raywhisper run
```

You should see:
```
Listening for hotkey: super+o (hold to record)
Hold super+o to record, release to transcribe
Press Ctrl+C to exit
```

### 3. Use Voice-to-Text

1. **Hold your configured hotkey** to start recording (default: `super+o` — Windows key + comma)
2. **Speak**: Say what you want to transcribe while holding the hotkey
3. **Release your configured hotkey** to stop recording and start transcription
4. **Wait**: The text will be transcribed and typed into your active application

> **Tip:** On platforms without a recording LED, consider a physical keyboard indicator or watch the app logs to verify recording state.

## Testing

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=raywhisper --cov-report=html
```

### Run specific tests

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/domain/test_transcription.py -v
```

## Code Quality

### Linting

```bash
ruff check src/
```

### Type Checking

```bash
mypy src/
```

### Format Code

```bash
ruff format src/
```

## Troubleshooting

### Audio Issues

**Problem**: No audio is being recorded

**Solutions**:
- Check microphone permissions
- Verify microphone is working in other applications
- Try listing available audio devices:
  ```python
  import sounddevice as sd
  print(sd.query_devices())
  ```

### Keyboard Hotkey Issues

**Problem**: Hotkeys not working

**Solutions**:
- **Windows**: Run as administrator if needed
- **macOS**: Grant Accessibility permissions:
  1. System Preferences > Security & Privacy > Privacy > Accessibility
  2. Add Terminal or your Python executable
- Try a different hotkey combination

### Model Loading Issues

**Problem**: Whisper model fails to load

**Solutions**:
- Check internet connection (models are downloaded on first use)
- Try a smaller model: `RAYWHISPER_WHISPER__MODEL_SIZE=tiny`
- Check available disk space (~1-3GB needed for models)

### GPU Issues

**Problem**: CUDA errors or slow performance

**Solutions**:
- **Verify Requirements**: GPU acceleration requires CUDA 12 and cuDNN 9 with a compatible NVIDIA GPU
- **Switch to CPU mode** if you don't have CUDA 12/cuDNN 9:
  ```yaml
  whisper:
    device: "cpu"
    compute_type: "int8"  # int8 is fastest on CPU
  ```
- Check GPU compatibility with faster-whisper documentation

## Next Steps

1. **Customize Configuration**: Edit `.env` to tune performance
2. **Add Your Documents**: Populate the vector database with your specific documents
3. **Test Different Models**: Try different Whisper models for accuracy vs. speed
4. **Integrate with Your Workflow**: Use in your favorite applications

## Getting Help

- Open an issue on GitHub for bugs or feature requests

## Performance Tips

1. **GPU Mode** (default - requires CUDA 12 + cuDNN 9):
   - Default uses `device: "cuda"` and `compute_type: "float16"`
   - Provides 5-10x speedup vs CPU
   - **Must have CUDA 12 and cuDNN 9 installed!**

2. **CPU Mode** (if no CUDA 12/cuDNN 9):
   - Change to `device: "cpu"` and `compute_type: "int8"`
   - Use smaller models like `"base"` or `"small"` for better speed

3. **Model Selection**:
   - Default: `"Systran/faster-whisper-medium.en"` (good accuracy, medium speed)
   - Faster: `"base"` or `"small"` (use with CPU)
   - Best quality: `"large-v3"` or `"distil-large-v3"` (requires GPU)

4. **Typing Speed**: Adjust `output.typing_speed` (default: 0.01, lower = faster)

## Advanced Usage

### View Configuration

```bash
raywhisper info
```

### Clear Vector Database

```bash
raywhisper populate ./docs2ingest --clear
```

### Use Clipboard Fallback

If keystroke simulation is unreliable, enable clipboard fallback:

```bash
RAYWHISPER_OUTPUT__USE_CLIPBOARD_FALLBACK=true
```

### Switching Back to Key Press Mode

If you want to configure the hotkey, change it in `config/config.yaml`:
```yaml
keyboard:
  start_stop_hotkey: "super+o"  # Windows key + comma; e.g., "super+o"
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

