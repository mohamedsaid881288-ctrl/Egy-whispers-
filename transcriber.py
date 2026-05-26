# transcriber.py
from transformers import pipeline as hf_pipeline

def transcribe_hf(wav_segments, model="AbdelrahmanHassan/whisper-large-v3-egyptian-arabic", device="cuda:0"):
    asr = hf_pipeline("automatic-speech-recognition", model=model, device=device)
    results = []
    for segfile, speaker, start, end in wav_segments:
        result = asr(segfile)
        results.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": result["text"].strip()
        })
    return results

if __name__ == "__main__":
    # You'd call this with segment list produced by diarizer.py
    passy