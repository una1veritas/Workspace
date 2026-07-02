from moviepy.audio.io.AudioFileClip import AudioFileClip

# Define your input and output file paths
input_mp3 = "wdrop.mp3"
output_wav = "wdrop.wav"

# 1. Read the MP3 file
# In MoviePy 2.0+, it is best practice to use a context manager (with statement)
# This ensures that FFmpeg processes and file handles close properly after execution.
with AudioFileClip(input_mp3) as audio_clip:
    
    # 2. Write the audio file as a WAV
    # MoviePy automatically detects the WAV format based on the extension.
    # The default codec for WAV files is 'pcm_s16le' (16-bit PCM).
    audio_clip.write_audiofile(output_wav,
                               fps=44100,
                               nbytes=1,
                               codec="pcm_u8")

print(f"Successfully converted {input_mp3} to {output_wav}")

