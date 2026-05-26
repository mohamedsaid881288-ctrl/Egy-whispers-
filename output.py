# output.py
import json

def write_txt(transcripts, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f:
        for entry in transcripts:
            f.write(f"Speaker {entry['speaker']} {entry['start']:.2f}-{entry['end']:.2f}: {entry['text']}\n")

def write_json(transcripts, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)