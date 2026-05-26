# main.py
import os
from preprocess import batch_convert
from pyannote.audio import Pipeline
from diarizer import diarize
from transcriber import transcribe_hf
from output import write_txt, write_json

INPUT_DIR = "input_audio"
TEMP_DIR = "temp_wav"
OUT_TXT = "output_txt"
OUT_JSON = "output_json"
HF_TOKEN = "YOUR_HF_TOKEN"
HF_MODEL = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
DEVICE = "cuda:0"

os.makedirs(OUT_TXT, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

if __name__ == "__main__":
    print("Preprocessing & converting...")
    files = batch_convert(INPUT_DIR, TEMP_DIR)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    for orig_name, wav_path in files:
        print(f"Diarizing: {orig_name}")
        segments = diarize(wav_path, pipeline)
        print(f"Transcribing...")
        transcripts = transcribe_hf(segments, HF_MODEL, DEVICE)
        base = os.path.splitext(orig_name)[0]
        write_txt(transcripts, os.path.join(OUT_TXT, f"{base}.txt"))
        write_json(transcripts, os.path.join(OUT_JSON, f"{base}.json"))
    print("Done.")