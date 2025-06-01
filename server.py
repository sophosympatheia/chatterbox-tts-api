from flask import Flask, request, jsonify, send_file
import io
import os
import random # Added for set_seed
import numpy as np
import wave
import torch
import yaml # Added import
from chatterbox.tts import ChatterboxTTS
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence

parser = argparse.ArgumentParser("server.py")
parser.add_argument("config_path", help="Path to the YAML configuration file.", type=str)
parser.add_argument("voices_dir", help="Path to the audio prompt files dir.", type=str)
args = parser.parse_args()

# Load configuration from YAML file
try:
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{args.config_path}' not found. Exiting.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration file '{args.config_path}': {e}. Exiting.")
    exit(1)

# Assign settings from config or use defaults
# Server Settings
server_config = config.get('server_settings', {})
API_PORT = server_config.get('port', 5001)
API_HOST = server_config.get('host', "127.0.0.1")

# Audio Generation Settings
audio_gen_config = config.get('audio_generation', {})
AUDIO_EXAGGERATION = audio_gen_config.get('exaggeration', 0.5)
AUDIO_TEMPERATURE = audio_gen_config.get('temperature', 0.8)
AUDIO_CFG_WEIGHT = audio_gen_config.get('cfg_weight', 0.5)
AUDIO_SEED = audio_gen_config.get('seed', 0) # New seed setting
AUDIO_DEFAULT_RESPONSE_FORMAT = audio_gen_config.get('default_response_format', 'wav')
# REMOVE_SILENCE is now part of silence_removal section

# Text Processing Settings
text_proc_config = config.get('text_processing', {})
MAX_CHUNK_LENGTH = text_proc_config.get('chunk_size', 300)

# Silence Removal Settings
silence_config = config.get('silence_removal', {})
REMOVE_SILENCE_ENABLED = silence_config.get('enabled', False)
SR_LT_SILENCE_THRESH_DBFS = silence_config.get('lt_silence_thresh_dbfs', -40)
SR_LT_MIN_SILENCE_DURATION_MS = silence_config.get('lt_min_silence_duration_ms', 500)
SR_INT_MIN_SILENCE_LEN_MS = silence_config.get('int_min_silence_len_ms', 700)
SR_INT_SILENCE_THRESH_DBFS = silence_config.get('int_silence_thresh_dbfs', -35)
SR_INT_KEEP_SILENCE_MS = silence_config.get('int_keep_silence_ms', 300)


AUDIO_PROMPT_PATH = args.voices_dir
if AUDIO_PROMPT_PATH[-1] != "/":
    AUDIO_PROMPT_PATH += "/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dynamically build SUPPORTED_VOICES from voices_dir
try:
    SUPPORTED_VOICES = sorted([
        f.rsplit('.', 1)[0]
        for f in os.listdir(AUDIO_PROMPT_PATH)
        if os.path.isfile(os.path.join(AUDIO_PROMPT_PATH, f)) and f.lower().endswith('.wav')
    ])
    if not SUPPORTED_VOICES:
        print(f"Warning: No .wav files found in {AUDIO_PROMPT_PATH}. No voices will be supported.")
    else:
        print(f"Supported voices found: {', '.join(SUPPORTED_VOICES)}")
except FileNotFoundError:
    print(f"Error: The voices_dir '{AUDIO_PROMPT_PATH}' was not found. Please check the path.")
    SUPPORTED_VOICES = []
except Exception as e:
    print(f"Error scanning voices_dir '{AUDIO_PROMPT_PATH}': {e}")
    SUPPORTED_VOICES = []


SUPPORTED_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

# MAX_CHUNK_LENGTH is now loaded from config

print(f"🚀 Running on device: {DEVICE}")
print(f"📝 Loaded configuration from: {args.config_path}")
print(f"🔊 Voices directory: {AUDIO_PROMPT_PATH}")
base_settings_info = f"Port={API_PORT}, Host={API_HOST}, Exaggeration={AUDIO_EXAGGERATION}, Temp={AUDIO_TEMPERATURE}, CFG={AUDIO_CFG_WEIGHT}, Seed={AUDIO_SEED}, DefaultFormat={AUDIO_DEFAULT_RESPONSE_FORMAT}, ChunkSize={MAX_CHUNK_LENGTH}"
silence_settings_info = f"RemoveSilenceEnabled={REMOVE_SILENCE_ENABLED}"
if REMOVE_SILENCE_ENABLED:
    silence_settings_info += (
        f", LtSilenceThresh={SR_LT_SILENCE_THRESH_DBFS}dBFS, LtMinSilenceDur={SR_LT_MIN_SILENCE_DURATION_MS}ms"
        f", IntMinSilenceLen={SR_INT_MIN_SILENCE_LEN_MS}ms, IntSilenceThresh={SR_INT_SILENCE_THRESH_DBFS}dBFS"
        f", IntKeepSilence={SR_INT_KEEP_SILENCE_MS}ms"
    )
print(f"⚙️ Settings: {base_settings_info}, {silence_settings_info}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_length: int) -> list[str]: # max_length will be passed MAX_CHUNK_LENGTH
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

def remove_silence_from_audio(
    wav_bytes: bytes,
    cfg_lt_silence_thresh_dbfs: int,
    cfg_lt_min_silence_duration_ms: int,
    cfg_int_min_silence_len_ms: int,
    cfg_int_silence_thresh_dbfs: int,
    cfg_int_keep_silence_ms: int
) -> bytes:
    """
    Removes leading, trailing, and internal silences from WAV audio data.
    Most silence parameters are configurable.
    Internal silence parameters are currently hardcoded.
    """
    try:
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    except Exception as e:
        print(f"Error loading audio for silence removal: {e}. Skipping silence removal.")
        return wav_bytes

    # Parameters for silence detection
    # Leading/Trailing from config
    # lt_silence_thresh_dbfs = -40 # Now from arg: cfg_lt_silence_thresh_dbfs
    # lt_min_silence_duration_ms = 500 # Now from arg: cfg_lt_min_silence_duration_ms
    
    # Internal (partially hardcoded, partially from config)
    # int_min_silence_len_ms = 700 # Now from arg: cfg_int_min_silence_len_ms
    # int_silence_thresh_dbfs = -35 # Now from arg: cfg_int_silence_thresh_dbfs
    # int_keep_silence_ms = 300 # Now from arg: cfg_int_keep_silence_ms

    original_duration_ms = len(audio)
    if original_duration_ms == 0:
        print("Input audio for silence removal is empty. Skipping.")
        return wav_bytes
        
    print(f"Original audio duration for silence removal: {original_duration_ms / 1000:.2f}s")

    # 1. Remove leading silence
    start_trim = detect_leading_silence(audio, silence_threshold=cfg_lt_silence_thresh_dbfs, chunk_size=10)
    trimmed_leading = audio[start_trim:]

    if len(trimmed_leading) == 0:
        print("Audio became empty after trimming leading silence. Returning empty audio.")
        empty_audio_io = io.BytesIO()
        AudioSegment.empty().export(empty_audio_io, format="wav")
        empty_audio_io.seek(0)
        return empty_audio_io.getvalue()

    # 2. Remove trailing silence (by reversing, trimming leading, then reversing back)
    end_trim = detect_leading_silence(trimmed_leading.reverse(), silence_threshold=cfg_lt_silence_thresh_dbfs, chunk_size=10)
    # Ensure end_trim does not exceed length of reversed audio
    end_trim = min(end_trim, len(trimmed_leading))
    trimmed_audio = trimmed_leading.reverse()[end_trim:].reverse()
    
    if len(trimmed_audio) == 0:
        print("Audio became empty after trimming trailing silence. Returning empty audio.")
        empty_audio_io = io.BytesIO()
        AudioSegment.empty().export(empty_audio_io, format="wav")
        empty_audio_io.seek(0)
        return empty_audio_io.getvalue()

    duration_after_ends_trimmed_ms = len(trimmed_audio)
    print(f"Duration after trimming leading/trailing silence: {duration_after_ends_trimmed_ms / 1000:.2f}s")

    # 3. Remove awkward pauses (internal silences)
    chunks = split_on_silence(
        trimmed_audio,
        min_silence_len=cfg_int_min_silence_len_ms,
        silence_thresh=cfg_int_silence_thresh_dbfs,
        keep_silence=cfg_int_keep_silence_ms
    )

    if not chunks:
        print("No chunks after splitting on internal silence (audio might be too short or entirely non-silent after end trimming). Using end-trimmed audio.")
        cleaned_audio = trimmed_audio # Use the audio that already had ends trimmed
    else:
        cleaned_audio = AudioSegment.empty()
        for chunk in chunks:
            cleaned_audio += chunk
    
    duration_after_internal_removed_ms = len(cleaned_audio)
    print(f"Duration after removing internal pauses: {duration_after_internal_removed_ms / 1000:.2f}s")

    output_io = io.BytesIO()
    cleaned_audio.export(output_io, format="wav")
    output_io.seek(0)
    return output_io.getvalue()

# Replace the old generate_audio function with this one:
def generate_audio(text, voice, speed=1.0):
    voice_file = AUDIO_PROMPT_PATH + f"{voice}.wav"

    # Set seed if configured to a non-zero value
    if AUDIO_SEED != 0:
        print(f"Setting global seed to: {AUDIO_SEED}")
        set_seed(AUDIO_SEED)
    else:
        print(f"AUDIO_SEED is {AUDIO_SEED}, not setting a global seed. TTS will use its default (likely random) seed behavior.")

    text_chunks = split_text_into_chunks(text, max_length=MAX_CHUNK_LENGTH) # Pass configured MAX_CHUNK_LENGTH

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
    final_wav_bytes = wav_io.getvalue()

    if REMOVE_SILENCE_ENABLED:
        if len(final_wav_bytes) > 0: # Only process if there's audio data
            print("Attempting to remove silence...")
            final_wav_bytes = remove_silence_from_audio(
                final_wav_bytes,
                cfg_lt_silence_thresh_dbfs=SR_LT_SILENCE_THRESH_DBFS,
                cfg_lt_min_silence_duration_ms=SR_LT_MIN_SILENCE_DURATION_MS,
                cfg_int_min_silence_len_ms=SR_INT_MIN_SILENCE_LEN_MS,
                cfg_int_silence_thresh_dbfs=SR_INT_SILENCE_THRESH_DBFS,
                cfg_int_keep_silence_ms=SR_INT_KEEP_SILENCE_MS
            )
            print("Silence removal process completed.")
        else:
            print("Skipping silence removal as generated audio is empty.")
            
    return final_wav_bytes


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
        "response_format", AUDIO_DEFAULT_RESPONSE_FORMAT
    )  # Default response format from config

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

    # Create a BytesIO object for the response using the converted data
    audio_io = io.BytesIO(converted_audio_data)
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
