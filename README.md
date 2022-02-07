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

It could be argued that it would have been easier to play and record the sound from the same computer. However this was infeasible, as we didn't have the proper equipment necessary to accomplish this.

After the recording work was done, our final distance dataset contained approximately 3000 audio samples from over 100 different classes.
### Data Preprocessing
#### Polishing the dataset
Polishing the dataset
During the creation of the dataset, we noticed that the distributed client-server architecture would sometimes start recording too late, as some audio files were so short that the laptop didn't receive the signal to start recording in time. Because recording again was deemed wasteful of our precious time, we decided to simply filter out the non-existent audio samples. Our dataset was still over 2500 audio samples large even after eliminating the faulty samples.

As mentioned before, a single audio sample could have multiple classes. While this was not an issue with distance prediction, it provided an extra challenge in the classification part of the project. A first step to manage this was to make all audio samples have the same amount of classes. Figure 4 shows an example of the process we used to accomplish this.
![figure prep](https://user-images.githubusercontent.com/34604921/152852972-0c77811e-a0cc-4258-a983-f30962c30519.png "Figure 4: Adapting the dataset so that the amount of classes is the same for each audio sample")
#### Normalization
Normalizing the dataset is a frequently used technique to achieve better training results. In order to normalize raw audio signals, one way is to set the RMS (root mean square) of all signals to a fixed value 2. This can be done by calculating the RMS for each audio sample and then dividing all the sample values by the RMS. Normalizing the RMS can be interpreted as ensuring that each audio sample has the same average power output.

Another way is to use the common min/max normalization, which we ended up using in the project due to better results.
#### Removing Background Noise
In order to provide as much meaningful information as possible to the network, any unintended parts of the audio signal should be filtered out. In our case, the goal was to reduce the background noise injected into the samples during the recording process as much as we could. For this task, we tested several common background noise removal approaches for their effectiveness.

The method of short-term energy for noise cancellation is often used for voice detection tasks 3. It detects the noisy parts of a signal because they have less energy than the voice parts. The identified noise can then be removed from the signal.

For noise cancellation in vibration signals, autocorrelation has been shown to be a useful tool 4. It works because the vibration is correlated to itself, whereas the noise is neither correlated to itself nor the vibration. Therefore repeatedly applying the autocorrelation operation to the signal reduces the noise more and more while the vibration itself is not decreased.

These two concepts however rely on a priori estimates of the signal, e.g. the signal being a voice or a vibration. As our dataset contains lots of very different signals, a priori estimates are impractical for our purposes.

Another method, Adaptive Noise Cancellation (ANC), deals with noise without any of these assumptions. ANC is achieved by introducing a cancelling anti-noise wave through secondary sources, which is generated by a neural network. These secondary sources are interconnected through an electronic system using a specific signal processing algorithm for the particular cancellation scheme. To implement ANC, we wrote an algorithm in Matlab for a single channel feed-forward active noise control system. We then used a sequence of training data to estimate the noise before this noise cancellation setup, and trained the ANC filter with white gaussian noise.

For testing our ANC, we took one of the longer noisy signals from our dataset and tried to eliminate the noise with the ANC feedforward method. Figure 5 shows the difference between the original signal and the noise eliminated signal.
![ANC_difference](https://user-images.githubusercontent.com/34604921/152853840-c16173cd-6298-473b-8499-a641a34f5cf2.png "igure 5: Original vs. denoised signal with use of ANC")

Applying this denoising technique indeed resulted in less interference for some samples. However due to the big diversity in our dataset the findings were inconsistent. What worked for one kind of signal effected another kind of signal negatively. Hence we ultimately decided not to deploy any denoising techniques for now and instead let our network deal with the background noise. In future work, handling denoising properly could lead to accuracy improvements.




