# preprocess.py
import os
import ffmpeg

def convert_and_denoise(input_path, output_path):
    """Converts audio to 16kHz mono WAV and normalizes/denoises using ffmpeg."""
    (
        ffmpeg.input(input_path)
        .output(output_path, ar='16k', ac=1, format='wav', af='loudnorm,afftdn')
        .overwrite_output()
        .run(quiet=True)
    )

def batch_convert(input_folder, temp_folder):
    os.makedirs(temp_folder, exist_ok=True)
    wav_files = []
    for fname in os.listdir(input_folder):
        in_path = os.path.join(input_folder, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(temp_folder, f"{base}.wav")
        convert_and_denoise(in_path, out_path)
        wav_files.append((fname, out_path))
    return wav_files

if __name__ == "__main__":
    # Example usage
    batch_convert("input_audio", "temp_wav")