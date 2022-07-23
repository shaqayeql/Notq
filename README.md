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

## speechToText
This function converts audio files to text files.
### Arguments
- audio_file_path: The path of the **.wav** audio file 
- function_name: The tool of speech-to-text converter user wants to use. it can have the values of “VOSK_wav”, “Google_wav” or “Microsoft”. If no argument is given, the default value would be “VOSK_wav”.
- output_text_directory: The directory in which the output text file would be saved. Default value is "notq_outputs"+os.sep+"text_files".
- subscription: The subscription for microsoft azure.
- region: The region for microsoft azure.

### Example
```python
speechToText("VOICE_AD\\titleA.mp3", function_name="Google_wav", output_text_directory="myDirectory\\myTextFiles")

```

## mp3ToWav
This function converts mp3 file/files to wav file/files. 
### Arguments
- audio_file_path: The path of the **.mp3** audio file/files 
- output_directory_path: The directory in which the output **.wav** file/files would be saved. Default value is "notq_outputs"+os.sep+"wav_audio_files".
- singleFilePath: A boolean which indicates whether there are multiple MP3 files the user wants to convert or there is only one file. If sets to **False**, the "audio_file_path" argument must be the path of a directory; otherwise, the audio_file_path" argument must be the path of a single MP3 file. The default value is True.

### Example
```python
mp3ToWav("VOICE_AD\\titleA.mp3", output_directory_path="myDirectory")
```
## resample(needs to be fixed)
This function changes sample rate of file/files to the desired rate.
### Arguments
- audio_file_path: The path of the **.wav** audio file/files 
- sampleRate: The desired sample rate of the output file. Default value is 
- output_directory_path: The directory in which the output **.wav** file/files would be saved. Default value is "notq_outputs"+os.sep+"wav_audio_files".
- singleFilePath: A boolean which indicates whether there are multiple MP3 files the user wants to convert or there is only one file. If sets to **False**, the "audio_file_path" argument must be the path of a directory; otherwise, the audio_file_path" argument must be the path of a single **.wav** file. The default value is True.

### Example
```python
resample("VOICE_AD\\titleA.mp3" , sampleRate)
```

<!-- ### VOSK_wav
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
``` -->

## loadSimilarityModel
This function returns a similarity model by getting a [similarity model](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz) path as an input.
### Arguments
- similarityModelPath: The path of similarity model. The model has a size of about 4GBs. Please be be careful to address the **.bin** file to the input.

### Example
```python
similarityModel = loadSimilarityModel("cc.fa.300.bin")
```

## cosineSimilarity
This function finds cosine similarity between sentences.
### Arguments
- sentence1: The first sentence in string format.
- sentence2: The second sentence in string format.
- similarityModel: The model object got from the [loadSimilarityModel](https://github.com/shaqayeql/Notq#loadsimilaritymodel) function (mentioned above).

### Example
```python
similarity = cosineSimilarity("من امروز به باشگاه رفتم", "امروز بود که بنده به باشگاه ورزش رجوع کردم", similarityModel)
```
## sentiment(needs to be changed)

## splitAudiofile
This function splits **.wav** and **.mp3** audio files into smaller parts.
### Arguments
- audio_file_path: The path of the **.wav** or **.mp3** audio file
- output_directory_path: The directory in which the splitted files would be saved. Default value is "notq_outputs" + os.sep +file_name + "_splitted".
- dividing_len: The length of splitted audio files in seconds. The default value is 60.

### Example
```python
splitAudiofile("VOICE_AD\\titleA.mp3", output_directory_path="myDirectory", dividing_len = 120)
```

# silenceTime
This function returns a list of the beginings and the ends of silence times in a **.wav** audio file.
### Arguments
- audio_file_path: The path of the **.wav** audio file
- min_silence_time: The minimum silence time that counts as silence in miliseconds. The default value is 100.
- silence_threshhold: The minimum threshhold for frequency of silence times. The default value is inputAudio.dBFS - 16.

### Example
```python
silenceTime("VOICE_AD\\titleA.mp3", min_silence_time=200)
```

### fluency
This function calculates fluency factors in a .wav speech audio file, which are "SpeechRate", "ArticulationRate", "PhonationTimeRatio", "MeanLengthOfRuns".
The default value for fluency factor type is "SpeechRate".
```python
caclulate_fluency(filename, fluencyType)
```
## Test
You can run test.py for testing functions that used in Notq library.