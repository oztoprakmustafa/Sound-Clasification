## Introduction
Gaining information from sounds is a fundamental human ability: We can detect and identify objects just from our hearing as well as estimate the direction and distance of that object. 
Writing software with the same abilities is a difficult task due to the enormous complexity of audio signals. Applying machine learning, in particular neural networks, is the most promising approach to meet this challenge. 
In this project, we are researching common methods to deploy neural networks for prediction from sound and are creating our own neural network that is able to extract certain information from audio samples. In particular, we are trying to predict the distance between source and microphone, and predict the sound itself
## Audio theory
In this section, essential theoretical elements of audio analysis are introduced.

### Pulse-code modulation (PCM)
In order to store an analog audio signal in memory, it has to be digitized by applying sampling and quantization. Sampling refers to measuring the signal values at specific timesteps, which transforms the original time-continuous signal into a time-discrete one. Quantization implies mapping the continuous signal values to discrete values in a specific range, e.g. 16 bits.
![](https://upload.wikimedia.org/wikipedia/commons/b/bf/Pcm.svg "Sampling and quantization of an analog signal (red) with 4-bit PCM, resulting in a time-discrete and value-discrete signal (blue).")

PCM is a format for storing uncompressed audio signals. It simply contains an array of values that have been produced by sampling and quantizing an analog signal. It has two basic properties:  The sampling rate (how many samples per second were taken) and the bit depth (the number of bits per sample value), which determines the resolution. A typical sampling rate is 44.1 kHz (e.g. CDs), and 16 bits is a common choice for the bit depth.
### Spectograms

A spectrogram is a visualiziaton of the frequency spectrum of a signal over time. The frequency spectrum represents the signal strength of the various frequencies present in the signal. It can be calculated by applying a fourier transform to the signal.
The spectogram is depicted as a heat map, which means the intensity at a specific frequency and time is expressed through the color.


![spectogram](https://user-images.githubusercontent.com/34604921/152851220-10f18d09-4c90-4a9f-b125-82f6b19d7647.png "Spectrogram of a recording of a clarinet playing a note. The bottom line is at the frequency of the keynote, the higher lines are the harmonics. The clarinet starts playing at 0.4 seconds")
### MFCC (Mel-Frequency Cepstral Coefficients)
For audio analysis, it often makes sense to extract certain features from the raw audio signal, like the signal energy or the spectogram. As a feature, the MFC coefficients represent the entire frequency spectrum compactly with few values (e.g. 40), which approximates the human auditory system more closely. This has proven useful for applications like speech or song recognition.
## The Dataset
### The dataset
For our project, we used an [FSD50k Zenodo](https://zenodo.org/record/4060432) audio dataset, commonly used in kaggle competitions. It has over 50k audio samples with 200 classes, one audio sample having multiple classes. The FSD50k dataset would be used for the classification part of this project.

### Creating our own dataset
While the FSD50K dataset has plenty of sound samples, it cannot be employed for distance prediction on its own. Hence using the FSD50k dataset, we recorded approximately 3000 audio samples from different distances in a room. Note that the room wasn't soundproof, therefore some background noise was inevitably included.

Figure 3 demonstrates the recording process. We placed the microphone in a distance of either 1 meter or 2 meters away from the speaker. To properly record the audio samples, we used a distributed client-server architecture that would notify the microphone client when the speaker would start playing a sample.

![distance](https://user-images.githubusercontent.com/34604921/152852520-7f0a2ece-4b9c-43a8-b2c9-f369775e68e8.png " Draft of the recording process. A PC connected to a speaker plays the samples, while a laptop records it with a microphone from a certain distance d, in our case 1 and 2 meters. The PC signals the laptop when it starts and stops playing each sample over a socket connection, so the laptop can start and stop recording its samples accordingly.")
