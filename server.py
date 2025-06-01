from flask import Flask, request, jsonify, send_file
import io
import numpy as np
import wave
import torch
from chatterbox.tts import ChatterboxTTS
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser("server.py")
parser.add_argument("voices_dir", help="Path to the audio prompt files dir.", type=str)
parser.add_argument(
    "supported_voices",
    help="Comma-separated list of supported voices. Example: 'alloy,ash,ballad,coral,echo,fable,onyx,nova,sage,shimmer,verse'",
    type=str
)
parser.add_argument(
    "--port", help="Port to run the server on. Default: 5001", type=int, default=5001
)
parser.add_argument(
    "--host",
    help="Host to run the server on. Default: 127.0.0.1",
    type=str,
    default="127.0.0.1",
)
parser.add_argument(
    "--exaggeration",
    help="Exaggeration factor for the audio. Default: 0.5",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--temperature",
    help="Temperature for the audio. Default: 0.8",
    type=float,
    default=0.8,
)
parser.add_argument(
    "--cfg",
    help="CFG weight for the audio. Default: 0.5",
    type=float,
    default=0.5,
)
args = parser.parse_args()

AUDIO_PROMPT_PATH = args.voices_dir
if AUDIO_PROMPT_PATH[-1] != "/":
    AUDIO_PROMPT_PATH += "/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_PORT = args.port
API_HOST = args.host
AUDIO_EXAGGERATION = args.exaggeration
AUDIO_TEMPERATURE = args.temperature
AUDIO_CFG_WEIGHT = args.cfg
SUPPORTED_VOICES=args.supported_voices.split(",")
SUPPORTED_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

MAX_CHUNK_LENGTH = 300

print(f"🚀 Running on device: {DEVICE}")

def split_text_into_chunks(text: str, max_length: int = 300) -> list[str]:
    """
    Splits a long text into chunks of max_length, trying to respect sentence boundaries.
    """
    chunks = []
    remaining_text = text.strip()
    sentence_terminators = ".!?"

    while remaining_text:
        if len(remaining_text) <= max_length:
            chunk_to_add = remaining_text.strip()
            if chunk_to_add:  # Ensure non-empty after final strip
                chunks.append(chunk_to_add)
            break

        current_slice_len = min(len(remaining_text), max_length)
        
        split_at = -1
        # Try to find the last sentence terminator in the slice
        # Search from current_slice_len-1 down to 1 (to ensure chunk is not empty if split)
        for i in range(current_slice_len - 1, 0, -1):
            if remaining_text[i] in sentence_terminators:
                split_at = i + 1  # Include the terminator
                break
        
        if split_at == -1:  # No sentence terminator found
            # Try to find the last space in the slice
            for i in range(current_slice_len - 1, 0, -1):
                if remaining_text[i] == ' ':
                    split_at = i + 1  # Split after space (it will be stripped later)
                    break
        
        if split_at == -1:  # Still no suitable split point (no space/terminator or only at index 0)
            # Hard cut at max_length
            split_at = max_length
        
        chunk = remaining_text[:split_at].strip()
        if chunk:  # Add non-empty chunk
            chunks.append(chunk)
        
        remaining_text = remaining_text[split_at:].strip()

    return [c for c in chunks if c] # Filter out any empty strings that might have resulted

app = Flask(__name__)
# Initialize the TTS model
tts_model = ChatterboxTTS.from_pretrained(DEVICE)

# Replace the old generate_audio function with this one:
def generate_audio(text, voice, speed=1.0):
    voice_file = AUDIO_PROMPT_PATH + f"{voice}.wav"

    text_chunks = split_text_into_chunks(text, max_length=MAX_CHUNK_LENGTH)

    if not text_chunks:
        print("No text chunks to process after splitting.")
        # Create an empty WAV file if there are no chunks
        wav_io_empty = io.BytesIO()
        with wave.open(wav_io_empty, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(tts_model.sr) # Use model's sample rate
        wav_io_empty.seek(0)
        return wav_io_empty.getvalue()

    all_wavs_np = []
    print(f"Input text split into {len(text_chunks)} chunk(s) of max size {MAX_CHUNK_LENGTH}.")

    for i, chunk in enumerate(text_chunks):
        # Skip empty chunks (should be filtered by split_text_into_chunks, but as a safeguard)
        if not chunk.strip():
            print(f"Skipping empty chunk {i+1}/{len(text_chunks)}.")
            continue
        
        print(f"Generating audio for chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")
        wav_chunk_tensor = tts_model.generate(
            chunk,
            audio_prompt_path=voice_file,
            exaggeration=AUDIO_EXAGGERATION,
            temperature=AUDIO_TEMPERATURE,
            cfg_weight=AUDIO_CFG_WEIGHT,
        )
        
        # Convert tensor to numpy array, ensure it's on CPU
        wav_chunk_np = wav_chunk_tensor.cpu().numpy()
        
        # Ensure wav_chunk_np is 1D for concatenation
        if wav_chunk_np.ndim > 1 and wav_chunk_np.shape[0] == 1:
            wav_chunk_np = wav_chunk_np.squeeze(0)
        elif wav_chunk_np.ndim > 1:
            # This case is unexpected if model returns (1, N) or (N,)
            # For robustness, try to flatten, but log a warning.
            print(f"Warning: Unexpected shape for wav_chunk_np: {wav_chunk_np.shape}. Flattening.")
            wav_chunk_np = wav_chunk_np.flatten()

        all_wavs_np.append(wav_chunk_np)

    if not all_wavs_np:
        print("No audio generated for any chunk.")
        # Create an empty WAV file if no audio was generated
        wav_io_empty = io.BytesIO()
        with wave.open(wav_io_empty, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_model.sr)
        wav_io_empty.seek(0)
        return wav_io_empty.getvalue()

    final_wav_np = np.concatenate(all_wavs_np)
    print("Audio generation complete for all chunks.")

    # Process the final concatenated audio
    audio_data = np.clip(final_wav_np, -1.0, 1.0)  # Clip to prevent saturation
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create a BytesIO object to write the WAV file
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(int(tts_model.sr * speed))  # Apply speed adjustment to final sample rate
        wf.writeframes(audio_data.tobytes())

    wav_io.seek(0)
    return wav_io.getvalue()  # Return the bytes of the WAV file


def convert_audio_format(audio_data, response_format):
    if response_format == "wav":
        return audio_data

    # Convert the audio data to the desired format using pydub
    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))

    # Convert to the desired format
    output_io = io.BytesIO()
    if response_format == "mp3":
        audio_segment.export(output_io, format="mp3")
    elif response_format == "flac":
        audio_segment.export(output_io, format="flac")
    elif response_format == "opus":
        audio_segment.export(output_io, format="opus")
    elif response_format == "aac":
        audio_segment.export(output_io, format="aac")
    elif response_format == "pcm":
        audio_segment.export(output_io, format="raw")  # PCM is raw audio

    output_io.seek(0)
    return output_io.getvalue()


@app.route("/v1/audio/speech", methods=["POST"])
def speech():
    data = request.get_json()

    # Extract parameters from the request
    text = data.get("input")
    voice = data.get("voice")
    speed = data.get("speed", 1.0)  # Default speed is 1.0
    response_format = data.get(
        "response_format", "wav"
    )  # Default response format is wav

    # Validate parameters
    if not text or len(text) > 4096:
        return (
            jsonify(
                {
                    "error": "Input text is required and must be less than 4096 characters."
                }
            ),
            400,
        )
    if voice not in SUPPORTED_VOICES:
        return jsonify({"error": "Unsupported voice specified."}), 400
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        return (
            jsonify(
                {
                    "error": "Unsupported response format specified. Got: "
                    + response_format
                }
            ),
            400,
        )

    # Generate audio from the text
    audio_data = generate_audio(text, voice, speed)

    # Convert the audio data to the desired format
    converted_audio_data = convert_audio_format(audio_data, response_format)

    # Create a BytesIO object for the response
    audio_io = io.BytesIO(audio_data)
    audio_io.seek(0)

    # Set the appropriate MIME type based on the requested response format
    mime_type = "audio/" + response_format

    return send_file(
        audio_io,
        mimetype=mime_type,
        as_attachment=True,
        download_name=f"speech.{response_format}",
    )


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
