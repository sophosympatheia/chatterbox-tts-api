# Chatterbox-tts-api

Serve a [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) TTS server with an [OpenAI Compatible API Speech endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech).

## Install

Chatterbox can't be installed on Python versions > 3.12. You can use `conda` to use another version. If you're already running Python 3.12 ou 3.11, you can just use a `venv` (`python -m venv venv && source venv/bin/activate`) and ignore the conda part.

```sh
conda create -n chatterbox python=3.12
conda activate chatterbox
pip install -r requirements.txt
```

## Usage

The server is now configured primarily through a YAML configuration file.

```sh
python server.py path/to/your/conf.yaml path/to/your/voices_dir
```

**Example:**
```sh
python server.py conf.yaml ./voices
```

*   `conf.yaml`: Path to your YAML configuration file (see "Configuration" section below).
*   `voices_dir`: Path to the directory containing your `.wav` voice prompt files. The server will automatically detect voices from the filenames (e.g., `alloy.wav` becomes voice `alloy`).

Server will run by default on `http://127.0.0.1:5001` (configurable in `conf.yaml`). The API endpoint is `/v1/audio/speech`.

Running `server.py -h` will display the following help message:
```
usage: server.py [-h] config_path voices_dir

positional arguments:
  config_path  Path to the YAML configuration file.
  voices_dir   Path to the audio prompt files dir.

options:
  -h, --help   show this help message and exit
```

## Configuration

The server's behavior is controlled by a YAML configuration file (e.g., `conf.yaml`). This file allows you to set:

*   **Server Settings**: `host` and `port`.
*   **Audio Generation Parameters**:
    *   `exaggeration`, `temperature`, `cfg_weight`: Control the characteristics of the generated speech. See comments in `conf.yaml` for details.
    *   `seed`: For reproducible audio output. `0` for random.
    *   `default_response_format`: The default audio format if not specified in the API request (e.g., "wav", "mp3").
*   **Text Processing**:
    *   `chunk_size`: Long input texts are split into smaller chunks to ensure stable generation. This sets the maximum length for these chunks.
*   **Silence Removal (Experimental)**:
    *   `enabled`: Master switch to turn on/off silence removal.
    *   Various parameters (`lt_silence_thresh_dbfs`, `lt_min_silence_duration_ms`, etc.) to fine-tune the detection and removal of leading, trailing, and internal silences. This feature is experimental and may require tuning.

Refer to the provided `conf.yaml` file for an example structure and default values.

### Using the API

See [OpenAI Compatible API Speech endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech). This API takes a json containing an input text and a voice and replies with the TTS audio data. 

Example API call with `curl`:

```sh
curl -X POST http://localhost:5001/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "Hello, this is a test.", "voice": "alloy"}' --output speech.wav
```

### Usage in SillyTavern

See [SillyTavern docs](docs/usage-sillytavern.md)
