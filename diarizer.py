# diarizer.py
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

def diarize(wav_path, pipeline):
    diarization = pipeline(wav_path)
    audio = AudioSegment.from_wav(wav_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segfile = f"{wav_path}_spk{speaker}_{turn.start:.2f}-{turn.end:.2f}.wav"
        chunk = audio[turn.start * 1000: turn.end * 1000]
        chunk.export(segfile, format="wav")
        segments.append((segfile, speaker, turn.start, turn.end))
    return segments

if __name__ == "__main__":
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_HF_TOKEN")
    diarize("temp_wav/example.wav", pipeline)