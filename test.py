from speechToText import convert_dir_mp3_to_wav
from speechToText import Google_wav
from speechToText import resample
from speechToText import sentiment
from speechToText import similarity
from speechToText import predict
from numpy.linalg import norm
import numpy as np
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


#TEST sentiment
def cosine_sim(a, b):
      return np.inner(a, b) / (norm(a) * norm(b))

m_model = similarity()
x = m_model.get_sentence_vector('دو نفر در بیمارستان بستری بودند .یک نفر میتوانست تکون بخوره اما دیگری نمیتونست')
y = m_model.get_sentence_vector('دو نفر تو بیمارستان بستری بودن.یکیشون میتونست تکون بخوره اما دیگری نه')
z = m_model.get_sentence_vector('دو تا مرد بستری بودند که یکیشون میتونست حرکت بکنه  اما اون یکی نه')
w = m_model.get_sentence_vector('دو بیمارستان در کنار هم ساخته شده اند')
xy = cosine_sim(x,y)
yz = cosine_sim(y,z)
zx = cosine_sim(x,z)
xw = cosine_sim(x,w)
wy = cosine_sim(y,w)
wz = cosine_sim(z,w)
print(xy,yz,zx,xw,wy,wz)
###

#TEST sentiment
x_model , tokenizer = sentiment()
pred,prob = predict(x_model ,np.array(['دوست داشتنی بود'
                                        ,'واقعا از قدرت نویسنده لذت بردم داستان کوتاهیه ولی به قول یکی از دوستان داستان تا همیشه گوشه‌ای از ذهن ادم‌میمونه خیلی خوب بود خیلی'
                                        ,'اصلا خوب نبود']) ,tokenizer,max_len=128)
print(pred,prob)
###