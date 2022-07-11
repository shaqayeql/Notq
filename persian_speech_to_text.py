import subprocess
import wave
import sys
import os

class SpeechToText:        
    def __init__(self):
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            print("module 'vosk' is installed")
        except ModuleNotFoundError:
            print("module 'vosk' is not installed")
            # or
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vosk==0.3.30"])

        SetLogLevel(-1)
        if not os.path.exists("model"):
            print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit (1)


    def get_text_vosk(self, filename):
        from vosk import Model, KaldiRecognizer
        wf = wave.open(filename , "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format mono PCM.")
            exit (1)

        model = Model("model")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        #clean file
        final_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                string = rec.Result()
                text = string[string.find('"text"')+10:-3] + " "
                final_text += text
        string = rec.FinalResult()
        text = string[string.find('"text"')+10:-3]
        final_text += text

        return final_text
