# Speech Emotion Analyzer

* The idea behind creating this project was to build a machine learning model that could detect emotions from the speech we have with each other all the time. Nowadays personalization is something that is needed in all the things we experience everyday. 

* So why not have a emotion detector that will guage your emotions and in the future recommend you different things based on your mood. 
This can be used by multiple industries to offer different services like marketing company suggesting you to buy products based on your emotions, automotive industry can detect the persons emotions and adjust the speed of autonomous cars as required to avoid any collisions etc.

## Analyzing audio signals
![](images/joomla_speech_prosody.png?raw=true)

[©Fabien_Ringeval_PhD_Thesis](https://drive.google.com/file/d/0B2V_I9XKBODhcEtZV1lRWW1fYTg/view).
<br>

### Datasets:
Made use of two different datasets:
1. RAVDESS.
This dataset includes around 1500 audio file input from 24 different actors. 12 male and 12 female where these actors record short audios in 8 different emotions i.e 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised.<br>
Each audio file is named in such a way that the 7th character is consistent with the different emotions that they represent.

2. SAVEE.
This dataset contains around 500 audio files recorded by 4 different male actors. The first two characters of the file name correspond to the different emotions that the potray. 

3. TESS.
There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.

## Audio files:
Tested out the audio files by plotting out the waveform and a spectrogram to see the sample audio files.<br>
**Waveform**

![](images/wave.png?raw=true)
<br>
<br>
**Spectrogram**<br>
![](images/spec.png?raw=true)
<br>

## data augmentation?
The data provided in model is clear which is recorded in well setup studio which doesn’t got any disturbances in it, so the model will overfit by training on same data, and to overcome this we are using data augmentation

Data augmentation is a process of increasing the amount of data by adding small disturbances and generating new data points from existing data.
The benefit of data augmentation is to reduce overfitting.To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed.
The objective is to make our model unchanged to those disturbances and enhace its ability to generalize.
<br>
![](images/data_augmentation.png?raw=true)
<br>


## Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files.
For feature extraction we make use of the [**LibROSA**](https://librosa.github.io/librosa/) library in python which is one of the libraries used for audio analysis. 
<br>
![](images/features_extraction.png?raw=true)
<br>
* Here there are some things to note. While extracting the features, all the audio files have been timed for 3 seconds to get equal number of features. 
* The sampling rate of each file is doubled keeping sampling frequency constant to get more features which will help classify the audio file when the size of dataset is small.
<br>

**The extracted features looks as follows**

<br>

![](images/feature2.png?raw=true)

<br>

These are array of values with lables appended to them. 

## Building Models

Since the project is a classification problem, we choose **Convolution Neural Network** as a model. We also built  **Long Short Term Memory** model and they performed with better accuracies.

Building and tuning a model is a very time consuming process. The idea is to always start small without adding too many layers just for the sake of making it complex. After testing out with layers, the model which gave the max validation accuracy against test data was more than 85%
<br>
<br>
![](images/cnn_acc_and_.png?raw=true)
<br>

## Predictions

After tuning the model, tested it out by predicting the emotions for the test data. For a model with the given accuracy these are a sample of the actual vs predicted values.
<br>
<br>
![](images/predictions.png?raw=true)
<br>

## Testing out with live voices.
In order to test out our model on voices that were completely different than what we have in our training and test data, we recorded our own voices with dfferent emotions and predicted the outcomes. 
<br>
![](images/predicting_emotion.png?raw=true)
<br>
This is the function which extracts features from audio and take it as numpy array and predicts the emotion.

## Conclusion
Building the model was a challenging task as it involved lot of trial and error methods, tuning etc. The model is very well trained to distinguish between emotions and it distinguishes with 100% accuracy. The model was tuned to detect emotions with more than 85% accuracy. Accuracy can be increased by including more audio files for training.
