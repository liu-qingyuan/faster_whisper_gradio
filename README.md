# Faster Whisper Gradio
Real-time speech-to-text application using Faster Whisper with Gradio.

## Installation

### 1. Install PyTorch
Make sure to install the correct version of PyTorch with CUDA for GPU acceleration. Run the following command:

```bash
pip install torch==2.2.1+cu121
```

### Install Faster Whisper
Install Faster Whisper directly from the GitHub repository:

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

This ensures you get the latest version of Faster Whisper for optimized real-time processing.

## Running the Application

To start the Gradio interface for real-time speech-to-text transcription, execute the following command:

```bash
python app.py
```
This will launch the Gradio application.
