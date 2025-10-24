# START HERE - Building RayWhisper Executable

## 🎯 Quick Build (3 Steps)

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Run build script
python build_exe.py

# 3. Test it
cd dist\raywhisper
raywhisper.exe --version
```

**That's it!** The executable is ready in `dist/raywhisper/`

---

## 📋 What You Need

- ✅ Virtual environment activated
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ `config\config.yaml` file exists (automatically verified during build)
- ✅ 5+ GB free disk space

**⚠️ Important Configuration Note:**
- The build includes your current `config\config.yaml`
- **Before distributing**: Ensure the config uses CPU mode if target users don't have CUDA 12/cuDNN 9
- See configuration requirements below

**Note:** The build scripts automatically verify that `config\config.yaml` exists and copy it to the distribution.

---

## 🔧 Recent Fixes

### ✅ All Issues Fixed!

1. **Import Error** - Fixed with `__main__.py` entry point
2. **SciPy Missing** - Fixed by including scipy/sklearn
3. **Transformers Missing** - Fixed by including transformers library
4. **Package Metadata Missing** - Fixed by copying metadata for tqdm, transformers, etc.
5. **Faster-Whisper Assets Missing** - Fixed by collecting ONNX models for VAD

**Status**: The executable now works correctly! Just rebuild to get all fixes.

```bash
# Clean old build first!
rmdir /s /q build dist
del raywhisper.spec

# Then build
python build_exe.py
```

---

## 📦 What You Get

After building, you'll have:

```
dist/raywhisper/
├── raywhisper.exe       ← The executable
├── run.bat              ← Quick launcher
├── README.txt           ← User guide
├── config/              ← Configuration
├── data/chroma/         ← ChromaDB (external, not in .exe)
└── logs/                ← Log files
```

**To distribute**: ZIP the entire `dist/raywhisper` folder

---

## ✅ Testing

```bash
cd dist\raywhisper

# Test 1: Version
raywhisper.exe --version

# Test 2: Help
raywhisper.exe --help

# Test 3: Populate embeddings
raywhisper.exe populate ..\..\docs --clear

# Test 4: Run
raywhisper.exe run
```

---

## 🚨 Troubleshooting

### Build fails?
1. Clean old builds: `rmdir /s /q build dist`
2. Delete spec file: `del raywhisper.spec`
3. Try again: `python build_exe.py`

### Import errors?
- Make sure virtual environment is activated
- Check all dependencies: `pip install -r requirements.txt`

### Executable crashes?
- Run from command line to see errors
- Check `logs/raywhisper.log`

---

## 🎁 Distribution

```bash
# Create ZIP
cd dist
tar -a -c -f raywhisper-v0.1.0-windows.zip raywhisper

# Or use PowerShell
Compress-Archive -Path raywhisper -DestinationPath raywhisper-v0.1.0.zip
```

Share the ZIP file with users!

---

## 💡 Key Points

✅ **ChromaDB is external** - Not embedded in .exe, stored in `data/chroma/`  
✅ **ML models download on first use** - Not included in package  
✅ **Single folder distribution** - Just ZIP and share  
✅ **User-configurable** - Edit `config/config.yaml`

## ⚙️ Configuration for Distribution

**Before building for distribution**, ensure `config\config.yaml` is set appropriately:

**For users WITHOUT CUDA 12/cuDNN 9 (recommended default):**
```yaml
whisper:
  model_size: "base"
  device: "cpu"
  compute_type: "int8"  # Fastest on CPU
```

**For users WITH CUDA 12/cuDNN 9:**
```yaml
whisper:
  model_size: "base"
  device: "cuda"
  compute_type: "int8_float16"
```

**⚠️ Users must have CUDA 12 and cuDNN 9 installed to use GPU mode!**  

---

## 🔄 Rebuild

If you need to rebuild:

```bash
# Clean
rmdir /s /q build dist
del raywhisper.spec

# Build
python build_exe.py
```

---

## ⚡ Alternative Build Methods

### Method 1: Python Script (Recommended)
```bash
python build_exe.py
```

### Method 2: Batch Script (Quick)
```bash
build_exe_simple.bat
```

**Ready to build?** Just run:
```bash
python build_exe.py
```

🎉 **Happy building!**
