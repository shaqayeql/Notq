import subprocess
import sys
import os
from pathlib import *

# resample
import torchaudio

# convert_dir_mp3_to_wav
import glob
from pydub import AudioSegment

# vosk_wav
import wave
import wget
import zipfile

# Google_wav
import speech_recognition as sr

# Sentiment
import numpy as np
from tqdm.notebook import tqdm
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

labels = ['negative', 'positive']

# Microsoft
subscription="<paste-your-speech-key-here>"
region="<paste-your-speech-location/region-here>"
filename="your_file_name.wav"

#silenceTime
from pydub import AudioSegment, silence

#split_wavfile
from pydub import AudioSegment
import math

#fluency detector
from persian_fluency_detector import *

#emotion Recognition
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
emotion_sampling_rate = feature_extractor.emotion_sampling_rate



def getFluency(audio_file_path, fluency_type="SpeechRate"):
    fluency = Fluency(audio_file_path)

    if fluency_type == "SpeechRate":
        return fluency.get_SpeechRate()
    elif fluency_type == "ArticulationRate":
        return fluency.get_ArticulationRate()
    elif fluency_type == "PhonationTimeRatio":
        return fluency.get_PhonationTimeRatio()
    elif fluency_type == "MeanLengthOfRuns":
        return fluency.get_MeanLengthOfRuns()
    else:
        print("Invalid fluency type...choose either SpeechRate, ArticulationRate, PhonationTimeRatio, or MeanLengthOfRuns")

def speechToText(audio_file_path , function_name="VOSK_wav", output_text_directory = "notq_outputs"+os.sep+"text_files" , subscription = "<paste-your-speech-key-here>" , region = "<paste-your-speech-location/region-here>"):
    if function_name == "VOSK_wav":
        return VOSK_wav(audio_file_path , output_text_directory)
    elif function_name == "Google_wav":
        return Google_wav(audio_file_path , output_text_directory)
    elif function_name == "Microsoft":
        return Microsoft(audio_file_path , subscription , region)
    else:
        print("Invalid speech to text converter tool...choose either VOSK_wav, Google_wav, or Microsoft")

def mp3ToWav(audio_file_path , output_directory_path="notq_outputs"+os.sep+"wav_audio_files", singleFilePath = True):
    """ This function converts mp3 file/files to wav file/files. If singleFilePath sets False,
        that means audio_path should be path of one directory. But if it sets True, that means 
        audio_path should be path of one file """

    #Single file
    if(singleFilePath):
        if audio_file_path[-4:] != ".mp3":
            print("The input audio file format is not mp3")
            return
        filename = audio_file_path.split(os.sep)[-1][:-4]
        src = audio_file_path
        dst = output_directory_path + os.sep + filename + ".wav"
        sound = AudioSegment.from_mp3(src)
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)
        sound.export(dst, format="wav")
        print("Successful .mp3 to .wav conversion for file: " + filename + ".mp3")

    #Multiple files
    else:
        types = (audio_file_path + os.sep + '*.mp3',)
        files_list = []
        for files in types:
            files_list.extend(glob.glob(files))
        for f in files_list:       
            filename = f.split(os.sep)[-1][:-4]
            src = f
            dst = output_directory_path + os.sep + filename + ".wav"
            sound = AudioSegment.from_mp3(src)
            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)
            sound.export(dst, format="wav")
            print("Successful .mp3 to .wav conversion for file: " + filename + ".mp3")
    

def resample(audio_file_path , sampleRate, singleFilePath = False):
    """ This function changes sample rate of file/files to sampleRate. If singleFilePath sets
        False, that means audio_path should be path of one directory. But if it sets True, that
        means audio_path should be path of one file """

    #if have one file
    if(singleFilePath):
        if audio_file_path[-3:] == "wav":
            fullPath = audio_file_path;
            waveform, sample_rate = torchaudio.load(fullPath)           
            print("***************")
            print(fullPath)
            metadata = torchaudio.info(fullPath)
            print(metadata)
            downsample_rate=sampleRate
            downsample_resample = torchaudio.transforms.Resample(
                sample_rate, downsample_rate, resampling_method='sinc_interpolation')
            down_sampled = downsample_resample(waveform)
            torchaudio.save(fullPath, down_sampled, downsample_rate)
            metadata = torchaudio.info(fullPath)
            print(metadata)

    #if have multi files
    else:
        arr = os.listdir(audio_file_path)
        for file in arr:
            if file[-3:] == "wav":
                fullPath = audio_file_path + os.sep + file;
                waveform, sample_rate = torchaudio.load(fullPath)
                print("***************")
                print(fullPath)
                metadata = torchaudio.info(fullPath)
                print(metadata)
                downsample_rate=sampleRate
                downsample_resample = torchaudio.transforms.Resample(
                    sample_rate, downsample_rate, resampling_method='sinc_interpolation')
                down_sampled = downsample_resample(waveform)
                torchaudio.save(fullPath, down_sampled, downsample_rate)
                metadata = torchaudio.info(fullPath)
                print(metadata)




def VOSK_wav(audio_file_path , output_text_directory):
    """ This function convers speech to text.
        filename is the name of file that we want convert it.
        directory_voice is the directory that our file is there.
        directory_text is the directory that output text saves there. """

    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except ModuleNotFoundError:
        print("module 'vosk' is not installed. please install vosk==0.3.30")

    wf = wave.open(audio_file_path , "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print ("Audio file must be WAV format mono PCM.")
        exit (1)

    SetLogLevel(0)
    # file_split = audio_file_path[:-4]
    file_name = audio_file_path.split(os.sep)[-1][:-4]
    
    if not os.path.exists("model"):
        installation = input("The Vosk model is not Installed. If you want install the model, Please enter \"Yes\" otherwise enter \"No\":")
        if (installation=="Yes" or installation=="yes") :
            url = "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.4.zip"
            wget.download(url, os.getcwd())
            with zipfile.ZipFile(os.getcwd()+ os.sep +'vosk-model-small-fa-0.4.zip', 'r') as h:
                h.extractall()
            os.rename("vosk-model-small-fa-0.4", "model")

        else:
            exit (1)

    print("Vosk started...")
    model = Model("model")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    #clean file
    if not os.path.exists(output_text_directory):
        os.makedirs(output_text_directory)
        
    open(output_text_directory + os.sep + file_name + ".txt", 'w').close()
    final_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            string = rec.Result()
            text = string[string.find('"text"')+10:-3] + " "
            f = open(output_text_directory + os.sep + file_name + ".txt", "ab")
            f.write(text.encode("utf-8"))
            f.close()
            final_text += text

    string = rec.FinalResult()
    not_encoded_text = string[string.find('"text"')+10:-3]
    text = not_encoded_text.encode("utf-8")
    f = open(output_text_directory + os.sep + file_name + ".txt", "ab")
    f.write(text)
    f.close()
    final_text += not_encoded_text
    print(file_name + ".wav is done")
    return final_text


def Google_wav(audio_file_path , output_text_directory):
    """ This function convers speech to text with Google.
        filename is the name of file that we want convert it.
        directory_voice is the directory that our file is there.
        directory_text is the directory that output text saves there."""

    #!/usr/bin/env python3
    print("Google started...")

    # obtain path to "english.wav" in the same folder as this script

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`

        wf = wave.open(audio_file_path , "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format mono PCM.")
            exit (1)

        file_name = audio_file_path.split(os.sep)[-1][:-4]

        #clean file
        if not os.path.exists(output_text_directory):
            os.makedirs(output_text_directory)
            
        open(output_text_directory + os.sep + file_name + ".txt", 'w').close()
        
        final_text = r.recognize_google(audio,language ='fa-IR')
        f = open(output_text_directory + os.sep + file_name + ".txt", "ab")
        f.write(final_text.encode("utf-8"))
        f.close()

        print(file_name + " is done")
        return final_text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


#Microsoft Speech To Text
def Microsoft(audio_file_path = "your_file_name.wav" , subscription = "<paste-your-speech-key-here>" , region = "<paste-your-speech-location/region-here>"):
    """ This function converts speech to text using microsoft azure. """
    
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ModuleNotFoundError:
        print("module 'azure-cognitiveservices-speech' is not installed. please install azure-cognitiveservices-speech==1.20.0")

    speech_config = speechsdk.SpeechConfig(subscription , region)
    audio_input = speechsdk.AudioConfig(audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="fa" , audio_config=audio_input)
    
    result = speech_recognizer.recognize_once_async().get()
    return result.text


from numpy.linalg import norm

# Similarity
def loadSimilarityModel(similarityModelPath):
    if not os.path.exists("cc.fa.300.bin"):
        print("Please download model from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz and unzip that near your files")
        exit(1)

    try:
        import fasttext
    except ModuleNotFoundError:
        print("module 'fasttext' is not installed")

    m_model = fasttext.load_model(similarityModelPath)
    return m_model


def cosineSimilarity(sentence1, sentence2, similarityModel):
    x = similarityModel.get_sentence_vector(sentence1)
    y = similarityModel.get_sentence_vector(sentence2)
    return np.inner(x, y) / (norm(x) * norm(y))


# Sentiment
MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-base-uncased'

def sentiment(listOfSentence):
    
    device = setup_device()

    # general config
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16

    EPOCHS = 3
    EEVERY_EPOCH = 1000
    LEARNING_RATE = 2e-5
    CLIP = 0.0
   
    

    # create a key finder based on label 2 id and id to label
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}

    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')

    # setup the tokenizer and configuration
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    config = BertConfig.from_pretrained(
        MODEL_NAME_OR_PATH, **{
            'label2id': label2id,
            'id2label': id2label,
        })

    x_model = SentimentModel(config=config)
    x_model = x_model.to(device)
    x_model.load_state_dict(torch.load(MODEL_NAME_OR_PATH , map_location=torch.device('cpu')))#if gpu is ready delete map location arg

    pred,prob = predict(x_model ,np.array(listOfSentence) ,tokenizer,max_len=128)

    return pred,prob
 
class SentimentModel(nn.Module):
    
    def __init__(self, config):
        super(SentimentModel, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, return_dict=False)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 

class TaaghcheDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Taaghche. """

    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len

        
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
    
    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        if self.has_target:
            target = self.label_map.get(str(self.targets[item]), str(self.targets[item]))

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')
        
        inputs = {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long)
        
        return inputs

def setup_device():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    return device

def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = TaaghcheDataset(
        comments=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len, 
        label_list=label_list)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def predict(model, comments, tokenizer, max_len=128, batch_size=32):
    device = setup_device()
    data_loader = create_data_loader(comments, None, tokenizer, max_len, batch_size, None)
    
    predictions = []
    prediction_probs = []

    
    model.eval()
    with torch.no_grad():
        for dl in tqdm(data_loader, position=0):
            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']

            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            
            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs, dim=1))

    predictions = torch.stack(predictions).cpu().detach().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()

    return predictions, prediction_probs
###

def silenceTime(audio_file_path, min_silence_time=100, silence_threshhold = None):
    myaudio = AudioSegment.from_wav(audio_file_path)
    if silence_threshhold == None:
        silence_threshhold=myaudio.dBFS - 16
    
    Silence = silence.detect_silence(myaudio, min_silence_len=min_silence_time, silence_thresh=silence_threshhold)
    Silence = [((start/1000),(stop/1000)) for start,stop in Silence] #convert to sec

    return Silence

def splitAudiofile(audio_file_path, output_directory_path=None, dividing_len=60):

    audio_file_format = audio_file_path[-3:]
    if audio_file_format != "mp3" and audio_file_format != "wav":
        print("audio file format must be .mp3 or .wav")
        return
    audio = AudioSegment.from_file(audio_file_path, audio_file_format)
    file_name = audio_file_path.split(os.sep)[-1][:-4]

    total_seconds = math.ceil(audio.duration_seconds)
    
    if output_directory_path == None:
        output_directory_path = "notq_outputs" + os.sep +file_name + "_splitted"
    
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    
    num = 0
    for i in range(0, total_seconds, dividing_len):
        t1 = i
        t2 = i+dividing_len
        t1 = t1 * 1000 #Works in (milliseconds * 60)=seconds
        t2 = t2 * 1000 
        newAudio = audio[t1:t2]
        dest = output_directory_path + os.sep + f'_part{num}.' + audio_file_format
        newAudio.export(dest, format=audio_file_format)
        num+=1
    print('All splited successfully')



# emotion Recognition
def emotionRecognition(path_of_your_wav):
    return emotionPredict(path_of_your_wav,emotion_sampling_rate)

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

def speech_file_to_array_fn(path, emotion_sampling_rate):
    speech_array, _emotion_sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_emotion_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def emotionPredict(path, emotion_sampling_rate):
    speech = speech_file_to_array_fn(path, emotion_sampling_rate)
    inputs = feature_extractor(speech, emotion_sampling_rate=emotion_sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs
##