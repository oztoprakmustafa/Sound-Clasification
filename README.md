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

