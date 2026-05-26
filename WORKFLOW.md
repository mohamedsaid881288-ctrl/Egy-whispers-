# Egy-whispers Workflow Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  INPUT AUDIO FILES                      │
│        (MP3, WAV, M4A, AMR, FLAC, etc.)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   PREPROCESSING MODULE     │
        │   (preprocess.py)          │
        │ • Convert to 16kHz mono    │
        │ • Normalize audio level    │
        │ • Denoise (FFT)            │
        │ • Output: .wav files       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  SPEAKER DIARIZATION       │
        │   (diarizer.py)            │
        │ • Detect speaker changes   │
        │ • Segment by speaker       │
        │ • Create speaker tracks    │
        │ • Output: labeled segments │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  SPEECH-TO-TEXT (ASR)      │
        │  (transcriber.py)          │
        │ • Whisper Large v3 Model   │
        │ • Egyptian Arabic optimized│
        │ • HuggingFace integration  │
        │ • Output: transcriptions   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   OUTPUT FORMATTING        │
        │   (output.py)              │
        │ • TXT: Human readable      │
        │ • JSON: Structured data    │
        │ • Includes speakers & time │
        └────────────┬───────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              OUTPUT FILES                               │
│  • output_txt/  → Plain text transcriptions            │
│  • output_json/ → JSON with metadata                   │
└─────────────────────────────────────────────────────────┘
```

## Step-by-Step Workflow

### Phase 1: Preprocessing
**Module**: `preprocess.py`
**Function**: `batch_convert(input_folder, temp_folder)`

```
Input: Any audio format (mp3, m4a, etc.)
       ↓
   FFmpeg converts:
   • Sample rate: Any → 16,000 Hz
   • Channels: Mono/Stereo → Mono
   • Format: Any → WAV
   • Effects: Loudness normalization + FFT denoising
       ↓
Output: Normalized .wav files in temp_wav/
```

**Code Flow**:
```python
for each audio file:
    convert_and_denoise(input_path, output_path)
    # Saves preprocessed WAV to temp_wav/
```

**Configuration**:
- Sample rate: 16 kHz (Whisper requirement)
- Channels: 1 (mono)
- Normalization: Loudness normalization
- Denoising: FFT-based noise reduction

---

### Phase 2: Speaker Diarization
**Module**: `diarizer.py`
**Function**: `diarize(wav_path, pipeline)`

```
Input: Preprocessed WAV file
       ↓
   pyannote.audio pipeline:
   1. Load speaker-diarization model
   2. Detect speaker turns (timing + identity)
   3. Segment audio by speaker
   4. Export speaker-labeled chunks
       ↓
Output: Speaker segments with timestamps
        Example: "file_spk_Speaker_0-2.45.wav"
```

**Output Structure**:
```python
segments = [
    (filepath, speaker_id, start_time, end_time),
    (filepath, speaker_id, start_time, end_time),
    ...
]
```

**Key Features**:
- Automatic speaker detection (no manual labels needed)
- Timestamp accuracy: ~100ms
- Multi-speaker support
- Works with Egyptian Arabic

---

### Phase 3: Speech-to-Text Transcription
**Module**: `transcriber.py`
**Function**: `transcribe_hf(wav_segments, model, device)`

```
Input: Speaker segments
       ↓
   HuggingFace Transformers:
   1. Load Whisper Large v3 (Egyptian Arabic)
   2. Process each speaker segment
   3. Convert speech to text
   4. Return transcriptions
       ↓
Output: Transcribed text for each segment
```

**Model Used**:
- **Model**: `AbdelrahmanHassan/whisper-large-v3-egyptian-arabic`
- **Language**: Egyptian Arabic (Masri)
- **Base**: OpenAI Whisper Large v3
- **Framework**: HuggingFace Transformers

**Hardware**:
```python
DEVICE = "cuda:0"  # GPU (faster, requires NVIDIA)
DEVICE = "cpu"     # CPU (slower, works everywhere)
```

**Output Format**:
```python
results = [
    {
        "speaker": "Speaker 0",
        "start": 0.0,
        "end": 2.45,
        "text": "السلام عليكم ورحمة الله وبركاته"
    },
    {
        "speaker": "Speaker 1",
        "start": 2.45,
        "end": 5.10,
        "text": "وعليكم السلام ورحمة الله وبركاته"
    },
    ...
]
```

---

### Phase 4: Output Generation
**Module**: `output.py`
**Functions**: `write_txt()`, `write_json()`

#### TXT Output
**File**: `output_txt/{filename}.txt`
```
Speaker 0 0.00-2.45: السلام عليكم ورحمة الله وبركاته
Speaker 1 2.45-5.10: وعليكم السلام ورحمة الله وبركاته
Speaker 0 5.10-8.20: كيف حالك؟
Speaker 1 8.20-10.15: الحمد لله بخير
```

**Format**:
- One line per speaker segment
- Format: `Speaker {ID} {START}-{END}: {TEXT}`
- Timestamps in seconds (2 decimal places)
- UTF-8 encoding (supports Arabic)

#### JSON Output
**File**: `output_json/{filename}.json`
```json
[
  {
    "speaker": "Speaker 0",
    "start": 0.0,
    "end": 2.45,
    "text": "السلام عليكم ورحمة الله وبركاته"
  },
  {
    "speaker": "Speaker 1",
    "start": 2.45,
    "end": 5.10,
    "text": "وعليكم السلام ورحمة الله وبركاته"
  },
  {
    "speaker": "Speaker 0",
    "start": 5.10,
    "end": 8.20,
    "text": "كيف حالك؟"
  },
  {
    "speaker": "Speaker 1",
    "start": 8.20,
    "end": 10.15,
    "text": "الحمد لله بخير"
  }
]
```

**Advantages**:
- Structured data
- Easy to parse programmatically
- Preserves speaker IDs and timing
- Supports downstream processing

---

## Complete Pipeline Execution

### Main Entry Point: `main.py`

```python
# Configuration
INPUT_DIR = "input_audio"          # Where to read audio files
TEMP_DIR = "temp_wav"              # Temporary preprocessed audio
OUT_TXT = "output_txt"             # Text output directory
OUT_JSON = "output_json"           # JSON output directory
HF_TOKEN = "YOUR_HF_TOKEN"         # HuggingFace authentication
HF_MODEL = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
DEVICE = "cuda:0"                  # GPU or "cpu"
```

### Execution Order

```
1. Create output directories
   ├─ output_txt/
   └─ output_json/

2. Preprocess all audio files
   └─ input_audio/* → temp_wav/*.wav

3. Load speaker diarization pipeline
   └─ Download from HuggingFace

4. For each audio file:
   a. Diarize → segments with speaker info
   b. Transcribe → text for each segment
   c. Write TXT output
   d. Write JSON output

5. Cleanup (optional)
   └─ Delete temp_wav files

6. Summary report
   └─ Files processed, errors, timing
```

---

## Running the Complete Workflow

### Quick Start

```bash
# 1. Prepare environment
source venv/bin/activate
export HF_TOKEN="your_token"

# 2. Add audio files
cp /path/to/audio.mp3 input_audio/

# 3. Run pipeline
python main.py

# 4. Check results
cat output_txt/audio.txt
cat output_json/audio.json
```

### With Configuration Options

```bash
# Modify main.py for your needs:
# - Change INPUT_DIR to a custom path
# - Set DEVICE="cpu" for CPU processing
# - Adjust HF_MODEL for different languages
# - Change output directories

python main.py
```

### Processing Multiple Files

```bash
# Add multiple audio files
cp audio1.mp3 audio2.m4a audio3.wav input_audio/

# Run once - all files processed automatically
python main.py

# Results generated for each file
ls output_txt/
# audio1.txt audio2.txt audio3.txt

ls output_json/
# audio1.json audio2.json audio3.json
```

---

## Performance Characteristics

### Processing Time (Approximate)

| Audio Duration | GPU (NVIDIA RTX 3060) | CPU (Intel i7) |
|----------------|----------------------|-----------------|
| 1 minute       | 10-15 seconds       | 1-2 minutes    |
| 10 minutes     | 1.5-2 minutes       | 10-15 minutes  |
| 1 hour         | 15-20 minutes       | 1.5-2 hours    |

*Times depend on:*
- Number of speakers
- Audio quality
- GPU/CPU specifications
- Background noise level

### Resource Requirements

| Component        | Min Requirement    | Recommended         |
|------------------|-------------------|----------------------|
| RAM              | 8 GB              | 16 GB+              |
| GPU VRAM         | 4 GB (optional)   | 8 GB+ (NVIDIA)      |
| Storage          | 2 GB free         | 10 GB+ (temp files) |
| Python           | 3.8               | 3.10+               |

---

## Customization Guide

### 1. Change Audio Preprocessing

**File**: `preprocess.py`
```python
def convert_and_denoise(input_path, output_path):
    # Modify FFmpeg filters:
    # 'loudnorm,afftdn' → other filters
    # Examples:
    # - 'highpass=f=50'        (remove low frequencies)
    # - 'lowpass=f=8000'       (remove high frequencies)
    # - 'aecho=0.8:0.9:1000:0.3' (add echo/reverb)
```

### 2. Use Different Whisper Model

**File**: `main.py`
```python
# For Modern Standard Arabic:
HF_MODEL = "openai/whisper-large-v2"

# For other languages:
HF_MODEL = "openai/whisper-medium"

# Custom fine-tuned models:
HF_MODEL = "your-username/your-model"
```

### 3. Change Output Format

**File**: `output.py`
```python
# Add SRT (subtitle) format:
def write_srt(transcripts, out_srt):
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, entry in enumerate(transcripts):
            f.write(f"{i+1}\n")
            f.write(f"{format_time(entry['start'])} --> {format_time(entry['end'])}\n")
            f.write(f"{entry['text']}\n\n")
```

### 4. Add Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for orig_name, wav_path in files:
        future = executor.submit(process_file, orig_name, wav_path, ...)
        futures.append(future)
    
    for future in futures:
        future.result()
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "CUDA out of memory" | GPU too small | Set `DEVICE="cpu"` |
| "FFmpeg not found" | Not installed | Install FFmpeg |
| "Module not found" | Missing dependencies | `pip install -r requirements.txt` |
| "No speakers detected" | Poor audio quality | Improve audio preprocessing |
| "Very slow processing" | CPU-only | Use GPU or reduce audio length |

---

## Output Examples

### Egyptian Arabic Conversation

**Input**: 30-second meeting recording (2 speakers)

**Output TXT**:
```
Speaker 0 0.00-2.30: السلام عليكم ورحمة الله
Speaker 1 2.50-5.10: وعليكم السلام ورحمة الله وبركاته
Speaker 0 5.30-8.45: إحنا هنتكلم اليوم عن المشروع الجديد
Speaker 1 9.00-12.15: تمام، أنا مستعد. فيه أي تفاصيل قبل ما نبدأ؟
Speaker 0 12.30-15.45: أه، العميل طلب تسليم في آخر الشهر
Speaker 1 16.00-18.20: حسناً، ده ممكن. كم العدد المتوقع؟
```

**Output JSON**:
```json
[
  {
    "speaker": "Speaker 0",
    "start": 0.0,
    "end": 2.3,
    "text": "السلام عليكم ورحمة الله"
  },
  {
    "speaker": "Speaker 1",
    "start": 2.5,
    "end": 5.1,
    "text": "وعليكم السلام ورحمة الله وبركاته"
  },
  ...
]
```

---

## Next Steps

1. ✅ Install dependencies (see SETUP.md)
2. ✅ Configure HF_TOKEN
3. ✅ Run `python main.py`
4. 🎯 Customize preprocessing/model for your needs
5. 🚀 Deploy to production with monitoring

For detailed setup instructions, see **SETUP.md**
