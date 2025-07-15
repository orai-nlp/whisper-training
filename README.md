# Whisper fine-tuning scripts

This repository contains scripts and configurations to fine-tune OpenAI's Whisper model on custom datasets, especially for under-resourced languages.

## Installation

Install necessary system and Python dependencies:

```bash
apt install sox
pip install -r requirements.txt
```

## How to fine-tune a Whisper model

**1. Create a config**

Create a YAML configuration file specifying the training settings, dataset paths, and model details.
An example configuration to adapt Whisper to Asturian is available in configs/example_config.yaml.

**2. Launch fine-tuning**

Run the fine-tuning script with your configuration:

```bash
python scripts/finetune.py configs/example_config.yaml
```

**How to test a fine-tuned Whisper model**

Use the test_simpler.py script to run inference with a fine-tuned model:

```bash
python scripts/test.py configs/example_config.yaml finetuned-whisper-tiny-ast results_dir --device cuda --batch_size 64
```

**How to server your fine-tuned Whisper model**

Run the server

```bash
python scripts/run_server.py language model
```

Test with the client:

```bash
python scripts/server_client.py audio.wav
```
