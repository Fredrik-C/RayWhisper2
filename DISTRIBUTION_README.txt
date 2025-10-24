================================================================================
RayWhisper - Voice to Text with RAG-Enhanced Transcription
================================================================================

Version: 0.1.0
Platform: Windows 10/11 (64-bit)
License: MIT

================================================================================
QUICK START
================================================================================

1. FIRST TIME SETUP - Populate Embeddings
   
   Before using RayWhisper, you must populate the vector database with your
   documents. This creates embeddings for RAG-enhanced transcription.
   
   Open Command Prompt in this folder and run:
   
   raywhisper.exe populate <path-to-your-documents> --clear
   
   Example:
   raywhisper.exe populate C:\Users\YourName\Documents --clear
   
   This will:
   - Parse all Markdown (.md) files in the specified directory
   - Generate embeddings using AI models
   - Store them in the data/chroma/ directory
   
   Note: First run will download ML models (~1-2 GB). This is normal.

2. RUN THE APPLICATION
   
   Option A - Use the launcher:
   Double-click: run.bat
   
   Option B - Use command line:
   raywhisper.exe run
   
   You should see:
   "Listening for key hold: ctrl+shift+space"
   "Hold keys to record, release to transcribe"

3. USE VOICE-TO-TEXT
   
   a) Hold down: Ctrl + Shift + Space
   b) Speak clearly while holding the keys
   c) Release the keys when done speaking
   d) Wait a moment - the transcribed text will be typed automatically
      into whatever application has focus

================================================================================
CONFIGURATION
================================================================================

Edit: config/config.yaml

Common settings to customize:

- Whisper Model Size (accuracy vs speed):
  whisper:
    model_size: "base"    # Options: tiny, base, small, medium, large-v3
  
  Recommendations:
  - "tiny" or "base" - Fast, good for simple dictation
  - "small" - Balanced performance
  - "medium" or "large-v3" - Best accuracy, slower

- Keyboard Shortcuts:
  keyboard:
    start_stop_hotkey: "ctrl+shift+space"
  
  Change to your preferred key combination

- Device (CPU vs GPU):
  whisper:
    device: "cpu"         # Options: cpu, cuda, auto
  
  Use "cuda" if you have an NVIDIA GPU with CUDA installed

- Typing Speed:
  output:
    typing_speed: 0.01    # Seconds between keystrokes
  
  Increase for slower typing, decrease for faster

================================================================================
SYSTEM REQUIREMENTS
================================================================================

Minimum:
- Windows 10 (64-bit) or later
- 4 GB RAM
- 2 GB free disk space
- Microphone

Recommended:
- Windows 11 (64-bit)
- 8 GB RAM
- 5 GB free disk space (for ML models and embeddings)
- NVIDIA GPU with CUDA support (optional, for faster transcription)

================================================================================
DIRECTORY STRUCTURE
================================================================================

raywhisper/
├── raywhisper.exe           Main executable
├── run.bat                  Quick launcher script
├── README.txt               This file
├── config/
│   ├── config.yaml          Main configuration file
│   └── config.example.yaml  Example configuration
├── data/
│   └── chroma/              Vector database storage (your embeddings)
├── logs/
│   └── raywhisper.log       Application log file
└── _internal/               Python runtime and dependencies (don't modify)

================================================================================
TROUBLESHOOTING
================================================================================

Problem: "No embeddings found" or poor transcription quality
Solution: Make sure you've run the populate command first:
          raywhisper.exe populate <path-to-docs> --clear

Problem: Application crashes on startup
Solution: 1. Check logs/raywhisper.log for error messages
          2. Ensure you have enough disk space
          3. Try running from Command Prompt to see errors

Problem: Slow transcription
Solution: 1. Use a smaller Whisper model (e.g., "tiny" or "base")
          2. If you have an NVIDIA GPU, set device: "cuda" in config
          3. Close other applications to free up RAM

Problem: Text not being typed
Solution: 1. Make sure another application has focus (cursor is active)
          2. Try enabling clipboard fallback in config:
             output:
               use_clipboard_fallback: true

Problem: Microphone not working
Solution: 1. Check Windows microphone permissions
          2. Verify microphone is set as default recording device
          3. Test microphone in Windows Sound settings

Problem: First run is very slow
Solution: This is normal! The application downloads ML models on first use.
          Subsequent runs will be much faster.

Problem: Antivirus warning
Solution: This is a false positive common with PyInstaller executables.
          The application is safe. You can:
          1. Add an exception in your antivirus
          2. Verify the source/download location

================================================================================
ADVANCED USAGE
================================================================================

Command Line Options:

  raywhisper.exe --help
    Show all available commands

  raywhisper.exe --version
    Show version information

  raywhisper.exe populate --help
    Show options for populating embeddings

  raywhisper.exe populate <path> --clear
    Clear existing embeddings and repopulate

  raywhisper.exe populate <path>
    Add documents to existing embeddings (incremental)

  raywhisper.exe run
    Start the voice-to-text application

Logs:

  Check logs/raywhisper.log for detailed information about:
  - Application startup
  - Recording sessions
  - Transcription results
  - Errors and warnings

ML Models Location:

  Models are downloaded to your user cache directory:
  %USERPROFILE%\.cache\huggingface\
  %USERPROFILE%\.cache\torch\
  
  You can delete these to free space, but they'll be re-downloaded on next use.

================================================================================
UPDATING EMBEDDINGS
================================================================================

When you add new documents or want to update your knowledge base:

1. Add/modify your documents in your source directory

2. Re-run the populate command:
   raywhisper.exe populate <path-to-docs> --clear

The --clear flag removes old embeddings before adding new ones.
Omit --clear to add to existing embeddings (incremental update).

================================================================================
PRIVACY & DATA
================================================================================

- All processing is done locally on your computer
- No data is sent to external servers
- Your voice recordings are processed in memory and not saved
- Embeddings are stored locally in data/chroma/
- ML models are downloaded from HuggingFace (one-time)

================================================================================
SUPPORT & INFORMATION
================================================================================

Project: https://github.com/Fredrik-C/RayWhisper2
License: MIT License (see LICENSE file)
Author: Fredrik Claesson

For issues, questions, or contributions, visit the GitHub repository.

================================================================================
CREDITS
================================================================================

RayWhisper uses the following open-source technologies:

- Faster Whisper - Speech recognition
- ChromaDB - Vector database
- Sentence Transformers - Text embeddings
- PyTorch - Machine learning framework
- And many other excellent open-source libraries

================================================================================

Thank you for using RayWhisper!

================================================================================

