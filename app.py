from asyncio.format_helpers import extract_stack
from contextlib import redirect_stderr
from fileinput import filename
from importlib.metadata import files
from wsgiref.util import request_uri
# from practice import get_features
import tensorflow as tf
from flask import Flask,request,render_template,redirect
import speech_recognition as sr
import librosa
import librosa.display
import pickle
import numpy as np
import os
import pandas
import soundfile
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

loaded_model_cnn = tf.keras.models.load_model('C:\course\Project_SR\Project webpage\model (2).h5')


def extract_features(file,ZCR,stft,mfcc,rms,mel):            
    with soundfile.SoundFile(file) as sound_file:
        data = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # ZCR
        result = np.array([])
        if ZCR:
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        if stft:
            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        if rms:
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
    return result

emotions={
  1:'neutral',
  2:'calm',
  3:'happy',
  4:'sad',
  5:'angry',
  6:'fearful',
  7:'disgust',
  8:'surprised'
}

@app.route('/predict',methods=["GET","POST"])
def predict(file):
        ans =[]
        new_feature  = extract_features(file,ZCR=True,stft=True,mfcc=True,rms=True,mel=True)
        # np.reshape(new_feature,newshape=(1,180,1))
        ans.append(new_feature)
        ans = np.array(ans)
        prediction = loaded_model_cnn.predict([ans])
        return emotions[np.argmax(prediction[0])+1]


@app.route("/",methods=["GET","POST"])
def index():
    prediction = ""
    if request.method == "POST":
        print("Data received.")

        file = request.files["file"]

        if "file" not in request.files:
            return redirect(request.url)


        if file.filename =="":
            return redirect(request.url)

        if file:
            if request.method == 'POST':
                # app.config['UPLOAD_FOLDER']
                file = request.files['file']
                
                file_path = "static/" + file.filename

                # global file_path
                file.save(file_path)
                
                # os.rename(file_path,'123.wav')
                # file_path = "static/123.wav"
                # file.save(file_path)
            
                
                # recognizer = sr.Recognizer()
                # audiofile = sr.AudioFile('static/123.wav')
                # with audiofile as source:
                    # Path = source
                    # data = recognizer.record(source)
                    # transcript = recognizer.recognize_google(data) 
                    # path = file_path()
                    # prediction = loaded_model.predict(path)
                # Path1 = file_path
                prediction = predict(file_path)

    return render_template('index.html',prediction_html=prediction)
                    
            

    # return render_template('index.html', transcript = transcript)

# prediction(index.file)

        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(file_path)
        #     output = predict(file_path)
        

           
# @app.route("/Choose_file_prediction")
# def Choose_file_prediction():
#     Path1 = 'static/'
#     prediction_html = pr.predict(Path1)
#     return render_template('Choose_file_prediction.html',prediction_html=prediction_html)

# @app.route("/Record")
# def Recorde():
#     return render_template('Recorde.html')


if __name__ == "__main__":

    app.run(debug=True,port=9999,threaded=True)