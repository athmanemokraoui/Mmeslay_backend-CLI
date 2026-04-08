# 🎧 CLI Tool README

This project provides a simple command-line interface (CLI) for running audio inference, testing a model, or launching a Flask server.

---

## 📦 Features

- Run inference on a single audio file
- Batch test the model on a directory of audio files
- Launch a Flask API server

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

Run the CLI using:

```bash
python src/cli.py [OPTIONS]
```

---

## 🔍 Options

| Option             | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `--infer`          | Path to a single audio file for inference                      |
| `--test`           | Run model tests on a folder of audio files                     |
| `--audio_location` | Path to folder containing audio files (required with `--test`) |
| `--server`         | Launch the Flask server                                        |

---

## 🎯 Examples

### 1. Run Inference on a Single File

```bash
python src/cli.py --infer path/to/audio.wav
```

---

### 2. Run Model Tests on a Folder

```bash
python src/cli.py --test --audio_location path/to/audio_folder
```

- Validates the folder path
- Sets `AUDIO_LOCATION` as an environment variable
- Executes:

```bash
python src/model_test.py
```

⚠️ Note: `--audio_location` is required when using `--test`.

---

### 3. Launch Flask Server

```bash
python src/cli.py --server
```

- Starts the Flask server by running:

```bash
python src/flask_server.py
```

---
