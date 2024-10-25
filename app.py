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
import whisper
import openai
import ast
from dotenv import load_dotenv
from fuzzywuzzy import process

load_dotenv()


modelWisper = whisper.load_model("tiny.en")

openai.api_key = os.getenv("OPENAI_APIKEY")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model_path = 'deepspeech-0.9.3-models.pbmm'
# scorer_path = 'deepspeech-0.9.3-models.scorer'

# model = deepspeech.Model(model_path)
# model.enableExternalScorer(scorer_path)

SAMPLE_FOODS = [
    "Basil",
    "Oregano",
    "Thyme",
    "Rosemary",
    "Cilantro",
    "Parsley",
    "Dill",
    "Sage",
    "Cumin",
    "Paprika",
    "Onion",
    "Garlic",
    "Carrot",
    "Tomato",
    "Bell pepper",
    "Spinach",
    "Broccoli",
    "Potato",
    "Zucchini",
    "Cauliflower",
    "Apple",
    "Banana",
    "Orange",
    "Lemon",
    "Berries",
    "Avocado",
    "Grapes",
    "Mango",
    "Pineapple",
    "Kiwi",
    "Rice",
    "Quinoa",
    "Lentils",
    "Chickpeas",
    "Black beans",
    "Oats",
    "Barley",
    "Pasta",
    "Wheat flour",
    "Cornmeal",
    "Chicken",
    "Beef",
    "Pork",
    "Fish",
    "Tofu",
    "Eggs",
    "Tempeh",
    "Shrimp",
    "Turkey",
    "Lamb",
    "Milk",
    "Yogurt",
    "Cheese",
    "Butter",
    "Cream",
    "Almond milk",
    "Soy milk",
    "Coconut yogurt",
    "Goat cheese",
    "Sour cream",
    "Olive oil",
    "Vegetable oil",
    "Coconut oil",
    "Sesame oil",
    "Balsamic vinegar",
    "Apple cider vinegar",
    "Red wine vinegar",
    "Rice vinegar",
    "Canola oil",
    "Peanut oil",
    "Soy sauce",
    "Ketchup",
    "Mustard",
    "Hot sauce",
    "Barbecue sauce",
    "Salsa",
    "Mayonnaise",
    "Honey",
    "Worcestershire sauce",
    "Tahini",
    "Almonds",
    "Walnuts",
    "Cashews",
    "Chia seeds",
    "Flaxseeds",
    "Pumpkin seeds",
    "Sunflower seeds",
    "Pistachios",
    "Hazelnuts",
    "Pecans",
    "Sugar",
    "Baking powder",
    "Baking soda",
    "Yeast",
    "Cocoa powder",
    "Vanilla extract",
    "Chocolate chips",
    "Cornstarch",
    "Honey",
    "Molasses",
    "Coconut flakes",
    "Panko breadcrumbs",
    "Instant coffee",
    "Matcha powder",
    "Nutritional yeast",
    "Sea salt",
    "Black pepper",
    "Maple syrup",
    "Miso paste",
    "Fermented vegetables"
]

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
    # audio, sample_rate = read_wav_file(path)
    # text = model.stt(audio)
    # return path, text
    pass

def processAudioAndExtractTranscriptionWisper(path):
    try:
        result = modelWisper.transcribe(path)
        return path, result["text"]
    except:
        print("error")

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

def parseIngredients(ingredients):
    ingredient_list = ast.literal_eval(ingredients)
    ingredient_dict = {}
    for item in ingredient_list:
        ingredient, quantity = item.rsplit('-', 1)
        ingredient_dict[ingredient] = quantity
    return ingredient_dict

def fetchIngredientsFromGPT(text):
    prompt = f"Please extract ingredients from the text also if possible extract quantity in array form like ['egg-1', 'tomato-2'], make sure to not add any additional text in response just array, if unable to extract quantity keep it 1, text is: {text}"
    response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    ingredients = response['choices'][0]['message']['content']
    return parseIngredients(ingredients)

@app.route('/upload-video/<mod>', methods=['POST'])
async def upload_file(mod):
    
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

        responseResult = []

        if mod == 1:
            print("mozilla deepspeech")
            results = await transcribe_files(processedFilePaths)
            for file_path, transcription in results.items():
                responseResult.append({ "file": file_path, "text": transcription })
        else:
            finalString = ""
            print("wisper")
            for d in processedFilePaths:
                path, text = processAudioAndExtractTranscriptionWisper(d)
                finalString += text + " "
            responseData = fetchIngredientsFromGPT(finalString)
            responseResult.append(responseData)

        finalIngredients = responseResult[0].keys()


        finalDict = {}
        for k in finalIngredients:
            best_match = process.extractOne(k, SAMPLE_FOODS)
            print(best_match)
            finalDict[best_match[0]] = responseResult[0][k]

        shutil.rmtree(f"{folderPath}/{newFileName}")
        os.remove(audioPath)
        os.remove(videoPath)

        return jsonify({'data': finalDict }), 200
    
    return jsonify({'error': 'Invalid file.' }), 400

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, threaded=True)

    #ingredients
    #quantity if possible
