# api-audio2txt-gemma4

API FastAPI compatible OpenAI `/v1/audio/transcriptions` et `/v1/audio/translations`
basée sur `google/gemma-4-E4B-it`.

## Caractéristiques

- compatible OpenAI audio endpoints
- accepte tout `model` en entrée mais l'ignore volontairement
- expose `/v1/models` avec `whisper-1` pour compatibilité
- convertit tout audio en mono 16 kHz WAV float32
- découpe automatiquement les audios > 30s
- parallélise le prétraitement et les chunks
- venv dans `~/venv/<basename project dir>`
- `install.sh` idempotent
- `source run.sh [IP] [PORT]`

## Fichiers

- `app.py`
- `requirements.txt`
- `install.sh`
- `run.sh`
- `.env.example`

## Installation

```bash
chmod +x install.sh
./install.sh
