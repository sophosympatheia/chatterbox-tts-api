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

```sh
python server.py path_to_voices_dir voices_list
```

Server will run by default on http://127.0.0.1:5001/v1/audio/speech.

Voice are expected to be wav audio files. For example, with a directory `/home/user/voices` containing `alloy.wav`, you would run `python server.py /home/user/voices alloy`

Running `server.py -h` will display the following help message.

```
usage: server.py [-h] [--port PORT] [--host HOST] [--exaggeration EXAGGERATION] [--temperature TEMPERATURE] [--cfg CFG] voices_dir supported_voices

positional arguments:
  voices_dir            Path to the audio prompt files dir.
  supported_voices      Comma-separated list of supported voices. Example: 'alloy,ash,ballad,coral,echo,fable,onyx,nova,sage,shimmer,verse'

options:
  -h, --help            show this help message and exit
  --port PORT           Port to run the server on. Default: 5001
  --host HOST           Host to run the server on. Default: 127.0.0.1
  --exaggeration EXAGGERATION
                        Exaggeration factor for the audio. Default: 0.5
  --temperature TEMPERATURE
                        Temperature for the audio. Default: 0.8
  --cfg CFG             CFG weight for the audio. Default: 0.5
```

### Using the API

See [OpenAI Compatible API Speech endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech). This API takes a json containing an input text and a voice and replies with the TTS audio data. 

Example API call with `curl`:

```sh
curl -X POST http://localhost:5001/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "Hello, this is a test.", "voice": "alloy"}' --output speech.wav
```

### Usage in SillyTavern

See [SillyTavern docs](docs/usage-sillytavern.md)
