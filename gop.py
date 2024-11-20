from flask import Flask, request, jsonify
import os
from moviepy.editor import VideoFileClip
from moviepy.audio.fx.all import audio_normalize
import random
import string
from pydub import AudioSegment
#import deepspeech
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
from pytube import YouTube
import subprocess
import shlex
import xmltodict
import json
import requests
from google.cloud import speech_v1p1beta1 as speech
import io
from google.cloud import storage
from google.cloud import translate_v2 as translate

client = speech.SpeechClient()


load_dotenv()


modelWisper = whisper.load_model("base")

openai.api_key = os.getenv("OPENAI_APIKEY")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model_path = 'deepspeech-0.9.3-models.pbmm'
# scorer_path = 'deepspeech-0.9.3-models.scorer'

# model = deepspeech.Model(model_path)
# model.enableExternalScorer(scorer_path)

productNamesArray = []
keyValueDictProducts = {}








def ub(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object for the destination file in the bucket
    blob = bucket.blob(destination_blob_name)
    # Upload the file
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def getSpruceData():
    url = "https://api.cmcffxkc6k-nagarroes2-d1-public.model-t.cc.commerce.ondemand.com/occ/v2/spruce-spa/productDetails?fields=DEFAULT"
    response = requests.get(url)
    data = response.json()
    mainArrayList = data.get("productInfoList")
    for d in mainArrayList:
        productNamesArray.append(d.get("name"))
        keyValueDictProducts[d.get("name")] = d.get("code")
    return 0

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

def split_audio(file_path, folderName, segment_length=60):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    num_segments = len(audio) // (segment_length * 1000) + 1
    if os.path.exists(("uploads/" + folderName)):
        try:
            shutil.rmtree(("uploads/" + folderName))
        except:
            os.remove(("uploads/" + folderName))
    output_dir = "uploads/" + folderName
    os.makedirs(output_dir, exist_ok=True)

    filePaths = []

    for i in range(num_segments):
        start_time = i * segment_length * 1000
        end_time = min((i + 1) * segment_length * 1000, len(audio))
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"{folderName}{i + 1}.wav"), format="wav", bitrate="96k")

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
        result = modelWisper.transcribe(path, task="translate")
        print(result["language"])
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
    ingredient_dict = {}
    try:
        ingredient_list = ast.literal_eval(ingredients)
        for item in ingredient_list:
            ingredient, quantity = item.rsplit('-', 1)
            ingredient_dict[ingredient] = quantity
    except Exception as e:
        print(e)
    return ingredient_dict

def fetchIngredientsFromGPT(text):
    prompt = f"Please extract ingredients from the text, make sure to translate ingredients in english for example bangun should be eggplant, also if possible extract quantity in array form like ['egg-1', 'tomato-2'], in quantity please include only integer like [egg-1] here egg is ingredient and 1 is quantity , make sure to not add any additional text in response just array, if unable to extract quantity keep it 0 and if the quantity is greater than 10 keep it 1, Keep the first letter capital, text is: {text}"
    response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    ingredients = response['choices'][0]['message']['content']
    return parseIngredients(ingredients)

def downloadYoutubeVideo(url, name):
    try:
        print(url)
        command = f"yt-dlp --cookies cookies.txt {url} -o {os.getcwd()}/uploads/{name} -S res:240"
        args = shlex.split(command)
        subprocess.run(args, check=True, text=True, capture_output=True)
        print("downloaded")
        return 1
    except Exception as e:
        print(e)
        return -1


def fetch_recipe_from_transcript(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}]
    )
    recipe = response['choices'][0]['message']['content']
    return json.loads(recipe.replace("```json\n", "").replace("```", "").strip())

def map_ingredients_id(recipe):
    for ingredient in recipe['ingredients']:
        ingredient_name = ingredient['ingredient']
        best_match = process.extractOne(ingredient_name, keyValueDictProducts.keys())
        ingredient['id'] = keyValueDictProducts.get(best_match[0], "") if best_match and best_match[1] > 85 else ""
    return recipe

def getRecipePrompt(transcript):
    prompt = f"""
    Task:
    I have a transcript of a cooking video. Your job is to extract the relevant details and create a step-by-step recipe based on the instructions given in the video. Make sure to include the following sections:

    Recipe Title – A clear, concise title for the recipe.
    Ingredients – List containing all ingredients with their quantities.
    Recipe – List containing the cooking process in easy-to-follow steps.
    Nutritional Information – Provide the nutritional values if mentioned in the video.
    Description – A brief description of the dish.
    Allergens – List any allergens mentioned (e.g., gluten, dairy).
    Bite Type – Specify if the recipe is vegetarian, vegan, non-vegetarian, etc.
    
    Instructions for formatting:
        - The format of the response will be strictly in JSON as shown below:
        {{
            "dish_name": "Recipe title here",
            "cook_time": "Recipe cook time (e.g., 40 minutes)",
            "servings": "Number of people that can be served (e.g., 1)",
            "allergens": "List of allergens if mentioned (e.g., Gluten, Dairy)",
            "description": "Brief description of the dish",
            "nutritional_values": {{
                "energy": "Energy in kcal (e.g., 300 kcal)",
                "protein": "Amount of protein (e.g., 5 g)",
                "fat": "Amount of fat (e.g., 15 g)",
                "carbs": "Amount of carbohydrates (e.g., 40 g)"
            }},
            "bitetype": "Veg, Non-Veg, Vegan, etc.",
            "ingredients": [
                {{
                    "ingredient": "Ingredient name",
                    "quantity": "Quantity of the ingredient"
                }},
                ...  // Additional ingredients as necessary
            ],
            "recipe": [
                {{
                    "step": "Step 1 description"
                }},
                ...  // Additional steps as necessary
            ]
        }}

    - Provide a friendly, easy-to-understand tone, suitable for a general audience.
    - Avoid including unnecessary details that aren't part of the recipe or cooking process (e.g., unrelated discussions).
    - If there are any unclear steps or missing information, make sure to clearly indicate them as "Unclear" or "Missing."

    Transcript:
    Here is the transcript from the video: {transcript}
    """
    return prompt


@app.route('/upload-video/<mod>', methods=['POST'])
async def upload_file(mod):

    is_recipe = request.args.get("recipe")
    url = request.args.get("url")
    videoName = ""

    if 'file' not in request.files and url == None:
        return jsonify({'error': 'No url or file found'}), 400

    file = None

    try:
        file = request.files['file']
    except:
        print("file not available checking with url")
    videoPath = None
    if file or url:
        newFileName = getFileName()

        folderPath = app.config['UPLOAD_FOLDER']
        if file:
            videoName = newFileName + substring_from_last_dot(file.filename)
            videoPath = os.path.join(folderPath, videoName)
            file.save(videoPath)
        else:
            print(url)
            isDownloaded = downloadYoutubeVideo(url, newFileName)
            dirContents = os.listdir("uploads")

            for v in dirContents:
                if newFileName in v:
                    print(v)
                    videoName = v
                    break
            videoPath = os.path.join(folderPath, videoName)
            if (isDownloaded == -1):
                return jsonify({'error': 'Video download failed'}), 400

        audioName =  newFileName + '.wav'
        audioPath = os.path.join(folderPath, audioName)
        convert_video_to_audio(videoPath, audioPath)
        processedFilePaths = split_audio(audioPath, newFileName)
        responseResult = []

        ub("bucket-nagarro", audioPath, audioName)

        audio = speech.RecognitionAudio(uri=f'gs://bucket-nagarro/{audioName}')
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
            audio_channel_count = 2,
            alternative_language_codes=["hi-IN"]
        )
        operation = client.long_running_recognize(config=config, audio=audio)

        response = operation.result(timeout=600)

        full_transcription = ""

        for result in response.results:
            full_transcription += result.alternatives[0].transcript + "\n"

        translate_client = translate.Client()
        translation = translate_client.translate(full_transcription, target_language='en')

        recipe = ''
        if is_recipe == "true" or is_recipe == "True":
            recipe = fetch_recipe_from_transcript(getRecipePrompt(translation['translatedText']))
            recipe = map_ingredients_id(recipe)
        

        ingredientsFromGPT = fetchIngredientsFromGPT(translation['translatedText'])
        finalIngredients = ingredientsFromGPT.keys()

        finalDict = {}
        for k in finalIngredients:
            best_match = process.extractOne(k, productNamesArray)
            print(keyValueDictProducts.get(best_match[0]))
            if best_match and best_match[1] > 85:
                finalDict[keyValueDictProducts.get(best_match[0])] = ingredientsFromGPT[k]

        shutil.rmtree(f"{folderPath}/{newFileName}")
        os.remove(audioPath)
        os.remove(videoPath)

        return jsonify({'data': finalDict, 'recipe': recipe }), 200

    return jsonify({'error': 'Invalid file.' }), 400

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    getSpruceData()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", debug=True, threaded=True)

    #ingredients
    #quantity if possible
