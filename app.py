from flask import Flask, request, jsonify
import os
from moviepy.editor import VideoFileClip
from moviepy.audio.fx.all import audio_normalize
import random
import string
from pydub import AudioSegment

import deepspeech
import numpy as np
import wave

import shutil

import re

import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'deepspeech-0.9.3-models.pbmm'
scorer_path = 'deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(model_path)
# model.enableExternalScorer(scorer_path)

def read_wav_file(filename):
    with wave.open(filename, 'rb') as wf:
        frames = wf.getnframes()
        buffer = wf.readframes(frames)
        return np.frombuffer(buffer, np.int16), wf.getframerate()

def convert_video_to_audio(video_file, audio_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)
    audio_clip.close()
    video_clip.close()

def substring_from_last_dot(s):
    last_dot_index = s.rfind('.')
    if last_dot_index != -1:
        return s[last_dot_index:]
    return ""

def getFileName():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

def split_audio(file_path, folderName, segment_length=30):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    num_segments = len(audio) // (segment_length * 1000) + 1
    output_dir = "uploads/" + folderName
    os.makedirs(output_dir, exist_ok=True)

    filePaths = []

    for i in range(num_segments):
        start_time = i * segment_length * 1000
        end_time = min((i + 1) * segment_length * 1000, len(audio))
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"{folderName}{i + 1}.wav"), format="wav")

        fileName = f"{output_dir}/{folderName}{i + 1}.wav"
        filePaths.append(fileName)
    return filePaths

def processAudioAndExtractTranscription(path):
    audio, sample_rate = read_wav_file(path)
    text = model.stt(audio)
    return path, text

async def transcribe_files(audio_files):
    loop = asyncio.get_event_loop()
    results = {}
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for audio_file in audio_files:
            future = loop.run_in_executor(executor, processAudioAndExtractTranscription, audio_file)
            futures.append(future)

        for future in asyncio.as_completed(futures):
            file_path, transcription = await future
            results[file_path] = transcription
    return results

@app.route('/upload-video', methods=['POST'])
async def upload_file():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        newFileName = getFileName()
        
        folderPath = app.config['UPLOAD_FOLDER']
        
        videoName = newFileName + substring_from_last_dot(file.filename)
        
        audioName =  newFileName + '.wav'
        
        videoPath = os.path.join(folderPath, videoName)
        
        audioPath = os.path.join(folderPath, audioName)
        
        file.save(videoPath)
        
        convert_video_to_audio(videoPath, audioPath)
        
        processedFilePaths = split_audio(audioPath, newFileName)

        results = await transcribe_files(processedFilePaths)

        responseResult = []


        for file_path, transcription in results.items():
            responseResult.append({ "file": file_path, "text": transcription })

        shutil.rmtree(f"{folderPath}/{newFileName}")
        os.remove(audioPath)
        os.remove(videoPath)

        return jsonify({'data': responseResult }), 200
    
    return jsonify({'error': 'Invalid file.' }), 400

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False, threaded=True)