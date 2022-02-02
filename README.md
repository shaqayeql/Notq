# Nodq: A Library For Processing Speech In Persian
Notq is a Python base tool collected and developed for speech and language processing in Persian. Speech processing is increasingly playing an important role in data analysis in various health research such as diagnose mental disorders. Early diagnosis of diseases is one of the most important concerns of the health system and most psychiatric disorders cause changes in the semantic network of words. Knowing and extracting the features of this network can help diagnose these disorders. The purpose of this project is to collect and develop tools for speech processing in Persian and semantic load analysis of their words to be integrated in a library or tool and the user can easily access all available high quality tools. In this library, to achieve this goal, modules such as converting speech to text, audio and vice manipulation tools, processing and analyzing text have been provided.

## Prerequisites


## Install
```python
pip install SpeechToText.py
```

## Documentation
Before you get started, here's a list of functions you can use:
### convert_dir_mp3_to_wav
This function converts mp3 file/files to wav file/files. To work with other functions, their format should be **.wav** . So you can use thie function.
If singleFilePath sets False, that means audio_path should be path of one directory(include many audio files). But if it sets True, that means audio_path should be path of single audio file.
```python
convert_dir_mp3_to_wav(audio_path , singleFilePath = False)
```
### resample
This function changes sample rate of file/files to the desired rate. If singleFilePath sets False, that means audio_path should be path of one directory(include many audio files). But if it sets True, that means audio_path should be path of single audio file.
```python
resample(directory_resample , sampleRate, singleFilePath = False)
```
### VOSK_wav
[Vosk](https://alphacephei.com/vosk/) is an offline speech recognition toolkit and this function convers speech to text using Vosk toolkit. filename is the name of file that we want convert it. directory_voice is the directory that our file is there. directory_text is the directory that output text saves there.
```python
VOSK_wav(filename , directory_voice , directory_text)
```
### Google_wav
This function convers speech to text with Google Speech Recognition. filename is the name of file that we want convert it. directory_voice is the directory that our file is there. directory_text is the directory that output text saves there.
```python
Google_wav(filename , directory_voice , directory_text)
```

### 

