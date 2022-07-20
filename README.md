# Notq: A Library For Processing Speech In Persian
Notq is a Python base tool collected and developed for speech and language processing in Persian. Speech processing is increasingly playing an important role in data analysis in various health research such as diagnose mental disorders. Early diagnosis of diseases is one of the most important concerns of the health system and most psychiatric disorders cause changes in the semantic network of words. Knowing and extracting the features of this network can help diagnose these disorders. The purpose of this project is to collect and develop tools for speech processing in Persian and semantic load analysis of their words to be integrated in a library or tool and the user can easily access all available high quality tools. In this library, to achieve this goal, modules such as converting speech to text, audio and vice manipulation tools, processing and analyzing text have been provided.

## Prerequisites
First, install [fluency package](https://github.com/salsina/persian-fluency-detector) and [syllable counter package](https://github.com/salsina/Persian-syllable-counter):
```python
pip install git+https://github.com/salsina/persian-fluency-detector#egg=persian_fluency_detector
```
```python
pip install git+https://github.com/salsina/Persian-syllable-counter#egg=persian_syllable_counter
```

## Install
```python
pip install git+https://github.com/shaqayeql/Notq#egg=notq
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
[Vosk](https://alphacephei.com/vosk/) is an offline speech recognition toolkit and this function converts speech to text using Vosk toolkit. filename is the name of file that we want convert it. directory_voice is the directory that our file is there. directory_text is the directory that output text saves there.
```python
VOSK_wav(filename , directory_voice , directory_text)
```
### Google_wav
This function converts speech to text using Google Speech Recognition. filename is the name of file that we want convert it. directory_voice is the directory that our file is there. directory_text is the directory that output text saves there.
```python
Google_wav(filename , directory_voice , directory_text)
```
### microsoft_from_file
This function converts speech to text using [microsoft azure](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/#overview).
```python
microsoft_from_file(filename , subscription , region)
```
### similarity
This function finds similarities between sentences using [model](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz) .
```python
similarity(similarityModelPath)
```
### sentiment
This function finds semantic similarity between sentences.
```python
sentiment(sentimentFilename , sentimentModelPath)
```
### fluency
This function calculates fluency factors in a .wav speech audio file, which are "SpeechRate", "ArticulationRate", "PhonationTimeRatio", "MeanLengthOfRuns".
The default value for fluency factor type is "SpeechRate".
```python
caclulate_fluency(filename, fluencyType)
```
## Test
You can run test.py for testing functions that used in Notq library.