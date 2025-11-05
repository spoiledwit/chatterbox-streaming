import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import login
import os

# Optional: Set your HF token here or use environment variable
HF_TOKEN = os.getenv("HF_TOKEN")  # or paste your token: "hf_xxxxx"
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged in to Hugging Face")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model from Hugging Face...")
print("(First run will download ~2GB of model files, this may take 2-5 minutes)")
model = ChatterboxTTS.from_pretrained(device=device)
print("✓ Model loaded successfully!")
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# STREAMING VERSION - generates and yields audio in real-time
print("\n=== STREAMING GENERATION ===")
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    # audio_prompt_path="YOUR_FILE.wav",  # Uncomment to use custom voice
    chunk_size=25,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    print_metrics=True
):
    audio_chunks.append(audio_chunk)
    print(f"✓ Chunk {metrics.chunk_count} received, duration: {audio_chunk.shape[-1] / model.sr:.2f}s")

# Save streaming result
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("streaming_output.wav", final_audio, model.sr)
print(f"\n✓ Saved streaming audio to streaming_output.wav")
print(f"Total chunks: {len(audio_chunks)}")
print(f"Total duration: {final_audio.shape[-1] / model.sr:.2f}s")
