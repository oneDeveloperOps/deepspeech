import deepspeech
import numpy as np
import wave

model_path = 'deepspeech-0.9.3-models.pbmm'
scorer_path = 'deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(model_path)
# model.enableExternalScorer(scorer_path)

def read_wav_file(filename):
    with wave.open(filename, 'rb') as wf:
        frames = wf.getnframes()
        buffer = wf.readframes(frames)
        return np.frombuffer(buffer, np.int16), wf.getframerate()

audio_file = 'output.wav'
audio, sample_rate = read_wav_file(audio_file)

if sample_rate != 16000:
    raise ValueError("Sample rate must be 16 kHz")

# Perform speech-to-text
text = model.stt(audio)
print("Recognized text:", text)
