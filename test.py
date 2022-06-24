from Notq.Notq import *

# microsoft_from_file
subscription="<paste-your-speech-key-here>"
region="<paste-your-speech-location/region-here>"

sampleRate = 4000

# os.getcwd() returns path of current directory
singlePath_voice_mp3 = os.getcwd()+"\\VOICE_AD"
filename = "titleh.mp3.wav"
directory_voice = os.getcwd()+"\\VOICE_AD"
directory_text = os.getcwd()+"\\text"
directory_resample = os.getcwd()+"\\VOICE_AD"
singlePath_resample = os.getcwd()+"\\VOICE_AD\\titleh.mp3.wav"
singlePath_convert = os.getcwd()+"\\VOICE_AD\\titleh.mp3"


## Test /convert_dir_mp3_to_wav/ (singleFilePath == False)
''' convert_dir_mp3_to_wav(directory_voice , singleFilePath = False) '''

## Test /convert_dir_mp3_to_wav/ (singleFilePath == True)
''' convert_dir_mp3_to_wav(singlePath_convert , singleFilePath = True) '''

## Test /resample/ (singleFilePath == False)
''' resample(directory_resample , sampleRate , False) '''

## Test /resample/ (singleFilePath == True)
''' resample(singlePath_resample , sampleRate , True) '''

## Test /speechToText/ (VOSK_wav)
''' speechToText("VOSK_wav" , filename , directory_voice , directory_text) '''

## Test /speechToText/ (Google_wav)
''' speechToText("Google_wav" , filename , directory_voice , directory_text) '''

## Test /speechToText/ (microsoft_from_file)
''' speechToText("microsoft_from_file" , filename , directory_voice , directory_text, subscription, region) '''

## Test /speechToText/ (default)
''' speechToText("",filename , directory_voice , directory_text) '''










## Test /caclulate_fluency/
''' caclulate_fluency(singlePath_voice_mp3) '''


#TEST similarity
""" def cosine_sim(a, b):
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
print(xy,yz,zx,xw,wy,wz) """
###

#TEST sentiment
""" x_model , tokenizer = sentiment()
pred,prob = predict(x_model ,np.array(['دوست داشتنی بود'
                                        ,'واقعا از قدرت نویسنده لذت بردم داستان کوتاهیه ولی به قول یکی از دوستان داستان تا همیشه گوشه‌ای از ذهن ادم‌میمونه خیلی خوب بود خیلی'
                                        ,'اصلا خوب نبود']) ,tokenizer,max_len=128)
print(pred,prob) """
###


