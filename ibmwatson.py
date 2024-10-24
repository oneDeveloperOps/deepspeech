from ibm_watson import SpeechToTextV1

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

apiUrl = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com"
myKey = "nVDzElX"

auth = IAMAuthenticator(myKey)

Speech2Text = SpeechToTextV1(authenticator = auth)

Speech2Text.set_service_url(apiUrl)

with open("audio.mp3", mode="rb") as wav: 
    response = Speech2Text.recognize(audio=wav, content_type="audio/mp3")
    recognized_text = response.result['results'][0]['alternatives'][0]['transcript']
    print(recognized_text)