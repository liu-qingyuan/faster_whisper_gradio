# Gemini Project Context: faster_whisper_gradio

## Project Overview

This project is a real-time speech-to-text application. It uses the `faster-whisper` library for efficient transcription and `gradio` to provide a simple web-based user interface. The application also integrates with the DeeplX API to offer translation of the transcribed text. The primary language for translation appears to be from Chinese to English.

The architecture is straightforward:
- `app.py` is the main application file that contains the Gradio UI and the core transcription logic.
- `worker_deeplx.py` is a separate worker module that handles making API calls to an external DeeplX translation service.
- `requirements.txt` lists the necessary Python dependencies for the project.

## Building and Running

### 1. Install Dependencies

The project dependencies are listed in `requirements.txt`. They can be installed using pip:

```bash
pip install -r requirements.txt
```
*Note: The README mentions specific installation steps for PyTorch and `faster-whisper` that might be important for GPU acceleration.*

### 2. Running the Application

The application can be started by running the `app.py` script:

```bash
python app.py
```
This will launch a local Gradio web server, allowing you to interact with the speech-to-text engine.

## Development Conventions

- **Modularity:** The application follows a simple modular design by separating the translation logic into its own file (`worker_deeplx.py`). This keeps the main application file (`app.py`) focused on the UI and transcription workflow.
- **Dependencies:** Project dependencies are managed via a `requirements.txt` file.
- **User Interface:** Gradio is used for building the user interface, suggesting a focus on rapid prototyping and ease of use.
