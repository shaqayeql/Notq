from speechToText import convert_dir_mp3_to_wav
from speechToText import Google_wav
from speechToText import resample
from speechToText import sentiment
from speechToText import similarity
from speechToText import predict
from speechToText import VOSK_wav
import numpy as np
import os
from speechToText import microsoft_from_file


sampleRate = 4000

singlePath_voice_mp3 = "C:\\Users\\Shaghayegh\\Desktop\\new\\SpeechToTextProject\\VOICE_AD"
filename = "titleh.mp3.wav"
directory_voice = "C:\\Users\\Shaghayegh\\Desktop\\new\\SpeechToTextProject\\VOICE_AD"
directory_text = "C:\\Users\\Shaghayegh\\Desktop\\new\\SpeechToTextProject\\text"
directory_resample = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD"
singlePath_resample = "C:\\Users\\Shaghayegh\\Desktop\\My Project\\code1\\VOICE_AD\\titleh.mp3.wav"

#convert_dir_mp3_to_wav(singlePath_voice_mp3 , True)

#resample(singlePath_resample , sampleRate , True)


#TEST Google_wav for a directory
""" arr = os.listdir(directory_voice)
for file in arr:
    if file[-3:] == "wav":
        Google_wav(file , directory_voice , directory_text)
        #VOSK_wav(file) """


#TEST VOSK for a single file
#if filename[-3:] == "wav":
        #Google_wav(filename , directory_voice , directory_text)
''' VOSK_wav(filename , directory_voice ,directory_text ) '''


#TEST sentiment
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

#TEST Microsoft Speech To Text
''' subscription="<paste-your-speech-key-here>"
region="<paste-your-speech-location/region-here>"
filename="your_file_name.wav"

microsoft_from_file(filename , subscription , region) '''
###





## chunk audio file with its words
''' from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_mp3("titleZ.mp3")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=100,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)

for i, chunk in enumerate(audio_chunks):

    out_file = "C:\\Users\\Shaghayegh\\Desktop\\new\\SpeechToTextProject\\splitAudio\\chunk{0}.wav".format(i)
    print("exporting" + out_file) 
    chunk.export(out_file, format="wav") '''


## google text to speech
''' def synthesize_text(text):

    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\Shaghayegh\\Desktop\\new\\SpeechToTextProject\\wired-ripsaw-346010-a26c2441a27d.json"
    #print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))

    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="fa-IR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

synthesize_text("titleh.mp3")'''

