import wave
import matplotlib.pyplot as plt
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import Dense


def create_datasets():
    wavs=[]
    labels=[] # labels testlabels 
    testwavs=[]
    testlabels=[]

    labsInd=[]      ##  0：seven   1：stop
    testlabsInd=[]  ##  0：seven   1：stop

    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\one\\"
    files = os.listdir(path)
    for i in files:
        
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("one" in labsInd)==False:
            labsInd.append("one")
        labels.append(labsInd.index("one"))

    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\two\\"
    files = os.listdir(path)
    for i in files:
       
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("two" in labsInd)==False:
            labsInd.append("two")
        labels.append(labsInd.index("two"))


    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\three\\"
    files = os.listdir(path)
    for i in files:
      
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("three" in labsInd)==False:
            labsInd.append("three")
        labels.append(labsInd.index("three"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\four\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("four" in labsInd)==False:
            labsInd.append("four")
        labels.append(labsInd.index("four"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\five\\"
    files = os.listdir(path)
    for i in files:

        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("five" in labsInd)==False:
            labsInd.append("five")
        labels.append(labsInd.index("five"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\six\\"
    files = os.listdir(path)
    for i in files:

        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("six" in labsInd)==False:
            labsInd.append("six")
        labels.append(labsInd.index("six"))    
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\seven\\"
    files = os.listdir(path)
    for i in files:
      
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("seven" in labsInd)==False:
            labsInd.append("seven")
        labels.append(labsInd.index("seven"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\eight\\"
    files = os.listdir(path)
    for i in files:
     
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("eight" in labsInd)==False:
            labsInd.append("eight")
        labels.append(labsInd.index("eight"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\entrenamiento\\nine\\"
    files = os.listdir(path)
    for i in files:
     
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("nine" in labsInd)==False:
            labsInd.append("nine")
        labels.append(labsInd.index("nine"))  
        
        
        
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\one\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("one" in testlabsInd)==False:
            testlabsInd.append("one")
        testlabels.append(testlabsInd.index("one"))


    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\two\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("two" in testlabsInd)==False:
            testlabsInd.append("two")
        testlabels.append(testlabsInd.index("two"))
        
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\three\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("three" in testlabsInd)==False:
            testlabsInd.append("three")
        testlabels.append(testlabsInd.index("three"))
        
  
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\four\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("four" in testlabsInd)==False:
            testlabsInd.append("four")
        testlabels.append(testlabsInd.index("four"))
 
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\five\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("five" in testlabsInd)==False:
            testlabsInd.append("five")
        testlabels.append(testlabsInd.index("five"))
 
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\six\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("six" in testlabsInd)==False:
            testlabsInd.append("six")
        testlabels.append(testlabsInd.index("six"))
 
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\seven\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("seven" in testlabsInd)==False:
            testlabsInd.append("seven")
        testlabels.append(testlabsInd.index("seven"))
    
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\eight\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("eight" in testlabsInd)==False:
            testlabsInd.append("eight")
        testlabels.append(testlabsInd.index("eight"))
 
    path="D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\nine\\"
    files = os.listdir(path)
    for i in files:
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("nine" in testlabsInd)==False:
            testlabsInd.append("nine")
        testlabels.append(testlabsInd.index("nine"))


    wavs=np.array(wavs)
    labels=np.array(labels)
    testwavs=np.array(testwavs)
    testlabels=np.array(testlabels)
    return (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd)


def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.frombuffer(strData,dtype=np.int16)
    waveData = waveData*1.0/(max(abs(waveData)))#wave
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

  
    data = list(np.array(waveData[0]))

    while len(data)>20000:
        del data[len(data)-1]
        del data[0]
    
    while len(data)<20000:
        data.append(0)
   

    data=np.array(data)
    data = data ** 2
    data = data ** 0.5

    return data


if __name__ == '__main__':
    (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd) = create_datasets()
    print(wavs.shape,"   ",labels.shape)
    print(testwavs.shape,"   ",testlabels.shape)
    print(labsInd,"  ",testlabsInd)

  
    labels = keras.utils.to_categorical(labels, 9)
    testlabels = keras.utils.to_categorical(testlabels, 9)
    print(labels[0]) 
    print(testlabels[0])

    print(wavs.shape,"   ",labels.shape)
    print(testwavs.shape,"   ",testlabels.shape)

    model = Sequential()
    model.add(Dense(1024, activation='relu',input_shape=(20000,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(216, activation='relu'))
    model.add(Dense(9, activation='softmax'))
  
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #  validation_data
    model.fit(wavs, labels, batch_size=124, epochs=20, verbose=1, validation_data=(testwavs, testlabels))


    score = model.evaluate(testwavs, testlabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model_weights.h5') 
    
    
    

from keras.models import load_model    
model = load_model ('model_weights.h5') # cargar modelo de entrenamiento
wavs=[]
wavs.append (get_wav_mfcc ("D:\\UMSS\\8 Semestre\\Reconocimiento de voz\\proyecto\\test\\four\\4_05_38.wav")) # usa un determinado archivo
X=np.array(wavs)
print(X.shape)

result = model.predict (X)
print ("Resultado de reconocimiento", result)

lista = []
name = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"] # Cree un conjunto de etiquetas que sean las mismas que durante el entrenamiento

rows, columns = result.shape    
i=0
while i < columns:
    a = result[0][i] 
    lista.append(a)
    i = i+1

indice=lista.index(max(lista))
    
print ("El resultado de voz reconocido es:", name [indice])
