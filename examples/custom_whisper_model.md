# Using Custom Whisper Models

RayWhisper supports both standard Whisper model sizes and custom HuggingFace models.

## Standard Model Sizes

The following standard model sizes are available:

- `tiny` - Fastest, lowest accuracy (~32MB, ~10x realtime)
- `base` - Good balance, default (~74MB, ~7x realtime)
- `small` - Better accuracy (~244MB, ~4x realtime)
- `medium` - High accuracy (~769MB, ~2x realtime)
- `large-v2` - Very high accuracy (~1.5GB, ~1x realtime)
- `large-v3` - Latest, best accuracy (~1.5GB, ~1x realtime)
- `distil-large-v3` - Distilled, fast with high accuracy (~756MB, ~6x realtime)

## Custom HuggingFace Models

You can use any CTranslate2-compatible Whisper model from HuggingFace by specifying the full model ID.

### Example: Systran/faster-whisper-medium.en

This is a distilled version of Whisper Large V3 that offers excellent speed/accuracy tradeoff.

**Configuration via config.yaml:**

```yaml
whisper:
  model_size: "Systran/faster-whisper-medium.en"
  device: "auto"              # Use GPU if available
  compute_type: "float16"     # FP16 for GPU, int8 for CPU
  language: null              # Auto-detect language
```

**Configuration via environment variables:**

```bash
export RAYWHISPER_WHISPER__MODEL_SIZE=Systran/faster-whisper-medium.en
export RAYWHISPER_WHISPER__DEVICE=auto
export RAYWHISPER_WHISPER__COMPUTE_TYPE=float16
```

**Windows PowerShell:**

```powershell
$env:RAYWHISPER_WHISPER__MODEL_SIZE="Systran/faster-whisper-medium.en"
$env:RAYWHISPER_WHISPER__DEVICE="auto"
$env:RAYWHISPER_WHISPER__COMPUTE_TYPE="float16"
```

## Other Compatible Models

Any model in CTranslate2 format on HuggingFace can be used:

### Systran Models

- `Systran/faster-whisper-tiny`
- `Systran/faster-whisper-base`
- `Systran/faster-whisper-small`
- `Systran/faster-whisper-medium`
- `Systran/faster-whisper-large-v2`
- `Systran/faster-whisper-large-v3`
- `Systran/faster-distil-whisper-small.en` (English-only, faster)
- `Systran/faster-distil-whisper-medium.en` (English-only, faster)
- `Systran/faster-whisper-medium.en` (Multilingual, fast)

### Custom Fine-tuned Models

If you have a custom Whisper model fine-tuned for your domain and converted to CTranslate2 format:

```yaml
whisper:
  model_size: "your-username/your-custom-whisper-model"
  device: "auto"
  compute_type: "float16"
```

## Performance Recommendations

### For CPU Usage

```yaml
whisper:
  model_size: "Systran/faster-whisper-medium.en"
  device: "cpu"
  compute_type: "int8"  # Quantized for faster CPU inference
```

### For GPU Usage (CUDA)

```yaml
whisper:
  model_size: "Systran/faster-whisper-medium.en"
  device: "cuda"
  compute_type: "float16"  # FP16 for GPU efficiency
```

### For Maximum Speed

```yaml
whisper:
  model_size: "Systran/faster-distil-whisper-small.en"  # English-only
  device: "cuda"
  compute_type: "int8_float16"  # INT8 weights, FP16 compute
```

### For Maximum Accuracy

```yaml
whisper:
  model_size: "Systran/faster-whisper-large-v3"
  device: "cuda"
  compute_type: "float16"
```

## Model Download

Models are automatically downloaded from HuggingFace on first use and cached locally.

**Cache location:**
- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

**First run will download the model:**

```bash
raywhisper info
# Output will show: Loading Whisper model: Systran/faster-whisper-medium.en...
# Model will be downloaded (may take a few minutes)
```

## Troubleshooting

### Model Not Found

If you get an error like "Model not found":

1. Check the model ID is correct on HuggingFace
2. Ensure the model is in CTranslate2 format (not PyTorch)
3. Check your internet connection

### Out of Memory

If you get OOM errors:

1. Use a smaller model (e.g., `small` instead of `large-v3`)
2. Use CPU instead of GPU: `device: "cpu"`
3. Use int8 quantization: `compute_type: "int8"`

### Slow Performance

If transcription is too slow:

1. Use GPU if available: `device: "cuda"`
2. Use a distilled model: `Systran/faster-whisper-medium.en`
3. Use int8_float16 on GPU: `compute_type: "int8_float16"`
4. Use a smaller model: `base` or `small`

## Verifying Configuration

Check your current configuration:

```bash
raywhisper info
```

Output will show:

```
RayWhisper Configuration:
  Whisper Model: Systran/faster-whisper-medium.en
  Device: auto
  Compute Type: float16
  Language: auto-detect
  ...
```

## References

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Systran Models on HuggingFace](https://huggingface.co/Systran)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

