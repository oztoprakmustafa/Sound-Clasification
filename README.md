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
## Creating the Network
### Model Implementation
Due to having two different problem statements, we opted on using two different models, one for classification and the other for distance prediction. In this section, we will discuss the two models that we implemented as well as their performance.
#### Python framework
We are using Keras as the deep learning library to construct our network.

For audio processing, Librosa turned out to be a suitable library. It provides several functions to extract features from audio data, e.g. for creating spectograms, calculating the MFCCs or performing a fourier transform.

Other libraries, such as SKlearn and Pandas, helped with the data processing and K-Fold model fitting.
#### Network Input
The audio source prediction part of this project could have easily boiled down to image classification by using some form of image representation of the audio samples, for example a picture of the raw audio signal or the corresponding spectrogram. Figure 6 shows some of these raw audio inputs.


![figure example](https://user-images.githubusercontent.com/34604921/152854490-52331ad8-653d-48fd-bae9-ac0c79c91ec8.png "Figure 6: Some samples of the dataset depicted as pictures of their raw audio signals")
Instead, however, for our first try we opted to extract the MFCC features using Librosa. Librosa returns the MFCC features from a signal over time in an array, which we can then use for classification. As MFCC is a quite compact representation of an audio signal, training is a lot faster due to the network having to process a smaller amount of data.

For the distance part of our project, we decided to try a different approach. We created a spectogram of each audio sample and indeed turned the distance prediction into an image classification problem. There are primarly two reasons for adopting this approach:

The previous approach yielded petty results in terms of accuracy.
We wanted to compare feature extraction and image classification approaches.
#### Classification
For the classification, we used a classic Convolution Neural Network (CNN). This decision was inspired by a popular blog from a kaggle competition called ["Beginners guide to audio data"](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data/).

The convolutional network model can be seen in Figure 7.
![class1](https://user-images.githubusercontent.com/34604921/152854736-00f0e57c-3276-4b78-ad82-96b160f71b41.png "Figure 7: The initial CNN model for the classification task")
This model, however, proved to perform very poorly, barely hitting 1% validation accuracy, even though the training accuracy proved to be at 28%. We believe the cause for this was the size of the dataset (2500) being far too small compared to the abundance of classes, leaving little room for error.
##### Tackling multi-label classification
In addition to the problems mentioned above, this model wasn't suited to predict multiple labels. Jason Brownlee and his guide "Multi-Label Classification of Satellite Photos of the Amazon Rainforest" 5 gave us an idea: what we can do is map all n labels to integers and store an n-element vector for each audio file. This vector contains 0 for labels that don't apply to the audio file, and 1 for labels that do. This corresponds to one hot encoding. After utilizing this technique, the predicting process was quite straightforward, resembling an Image Classification task.

##### Evaluating the performance
As we are dealing with a multi-class classification task, commonly used performance metrics for binary classification tasks aren't suitable. As an example for binary classifier metrics, the classic F1 score calculates the mean of precision and recall. Here, the precision describes how good the model is at predicting the positive outcome, and the recall quantifies the model's ability to predict positive samples as positive. For predicting multiple classes however, this is inapplicable as there is no clear positive or negative class.

The so-called F-Beta metric overcomes this by first calculating prediction and recall for each class in a one vs rest manner and then averaging them over all classes. A constant Beta is utilized in order to weigh precision and recall differently. We are using the common choice of two for Beta, which makes the recall valued twice as highly as the precision. The F-Beta metric is calculated as follows:

F-Beta = (1 + Beta^2) x (precision x recall) / (Beta^2 x precision + recall)

Figure 8 shows the results as two graphs, the cross entropy loss and F-Beta. This figure explains that the network is quite heavily overfitting the data, however the F-Beta value is close to the ideal, signifying the incredible boost in performance compared to the previous model.

![second_model](https://user-images.githubusercontent.com/34604921/152855106-b2b9a2a6-0526-4cf1-bf63-13e66b68a208.png "Figure 8: Evaluating the training performance of the improved classification network")
#### Distance
Regarding the distance prediction model, we were fitting it with the images of all the audio spectograms. As for the model design itself, we used a new concept previously unknown to us: a CRNN (convolutional recurrent neural network).

The idea to use a CRNN for distance classification was proposed by Mariam Yiwere et al. in ["Sound Source Distance Estimation Using Deep Learning: An Image Classification Approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6982911/). A CRNN enables the model to learn both the spectral and temporal features and relationships effectively.

The CRNN model we used can be see in Figure 9. As a loss function, we used cross entropy, and the Adam optimizer was used for training.

![CRNN](https://user-images.githubusercontent.com/34604921/152855814-bee08404-6166-4ebe-853a-ebefb2010f71.png "Figure 9: The CRNN used for the distance prediction task")
As we only had the time to record two different distances (1 and 2 meters), the hope for this project was that the network at least would predict more accurately than a coin flip. Our hope was fulfilled, since the validation accuracy for this model reached 53%.
### Conclusion and future work
We have created 2 models for two different problem statements - that being audio classification and sound distance prediction. For these tasks, we researched existing models that solve them and adapted them for our purposes. While the results are not great, we believe that the methods we used are intuitive and correct.

For future work, we would continue by attempting to tackle the different kinds of noise cancellation techniques that were researched throughout this project, such as the short-term autocorrellation method.




