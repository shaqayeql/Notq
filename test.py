from speechToText import convert_dir_mp3_to_wav, similarity
from speechToText import Google_wav
from speechToText import resample
from speechToText import sentiment
from speechToText import similarity
import os


sampleRate = 4000

singlePath_voice_mp3 = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD\\titleh.mp3"
filename = "titleh.mp3.wav"
directory_voice = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD"
directory_text = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\text"
directory_resample = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD"
singlePath_resample = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD\\titleh.mp3.wav"

#convert_dir_mp3_to_wav(singlePath_voice_mp3 , True)

#resample(singlePath_resample , sampleRate , True)


#TEST Google_wav for a directory
''' arr = os.listdir(directory_voice)
for file in arr:
    if file[-3:] == "wav":
        Google_wav(file , directory_voice , directory_text)
        #VOSK_wav(file)  '''


#TEST Google_wav for a single file
""" if filename[-3:] == "wav":
        Google_wav(filename , directory_voice , directory_text)
        VOSK_wav(file)  """


#similarity()

sentiment()