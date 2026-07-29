

import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os
import time

#%%
Train_dir = 'resized.1'
Test_dir='testdata'
Img_size = 64 #256

LR = 0.000001
def label_img(img):
    word_label = img.split('.')[0]
    if word_label ==  'resized_Jacques_Chirac':                                            #'resized_Ariel_Sharon':
        return [1,0,0,0,0]
    elif word_label == 'resized_Colin_Powell':
        return [0,1,0,0,0]
    elif word_label =='resized_George_W_Bush':
        return [0,0,1,0,0]
    elif word_label =='resized_Gerhard_Schroeder':
        return [0,0,0,1,0]
    elif word_label =='resized_Tony_Blair':
        return [0,0,0,0,1]
#    elif word_label =='resized_Ariel_Sharon':
#        return [0,0,0,0,0,1]
#    elif word_label =='Jean_Chretien':
#        return [0,0,0,0,0,0,1,0,0,0,0]
#    elif word_label =='John_Aschcroft':
#        return [0,0,0,0,0,0,0,1,0,0,0]
#    elif word_label =='Junichiro_Koizumi':
#        return [0,0,0,0,0,0,0,0,1,0,0]
#    elif word_label =='Serena_Williams':
#        return [0,0,0,0,0,0,0,0,0,1,0]
#    elif word_label =='Tony_Blair':
#        return [0,0,0,0,0,0,0,0,0,0,1]
#%%
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        if label==None:
            continue
        path = os.path.join(Train_dir,img)
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #np.save('sc_train_data.npy',training_data)
    return training_data
    
#%%
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        label = label_img(img)
        if label==None:
            continue
        path = os.path.join(Test_dir,img)
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        testing_data.append([np.array(img),np.array(label)])
    shuffle(testing_data)
    #np.save('sc_train_data.npy',training_data)
    return testing_data


#%%Load Traing and Testing Data:
train_data = create_training_data()
test_data = create_test_data()

train = train_data[:-20] 
test = test_data[-20:]
#%%
X_train= np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
Y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
Y_test= np.array([i[1] for i in test])
#%%


#%%
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils
#from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from keras import UpSampling2D, ZeroPadding2D, Input
batch_size = 1     
hidden_neurons = 100
classes = 5   
epochs = 20

#%%
model = Sequential() 
model.add(Convolution2D(4, (3, 3), input_shape=(Img_size, Img_size, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(Convolution2D(4, (3, 3)))  
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))   
#
#model.add(Convolution2D(4, (3, 3))) 
#model.add(Activation('relu'))     
#model.add(Convolution2D(4, (3, 3)))     
#model.add(Activation('relu'))  
#model.add(Convolution2D(4, (3, 3)))     
#model.add(Activation('relu'))        
#model.add(MaxPooling2D(pool_size=(2, 2)))     
#model.add(Dropout(0.25))
               
model.add(Flatten())
 
model.add(Dense(hidden_neurons)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))
     

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1]*100) 

#%%
import matplotlib.pyplot as plt
fig =plt.figure(figsize=(10,10))
out=[]
for num,data in enumerate(test[:16]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4,4,num+1)
    og = img_data
    img_data = img_data.reshape(1,Img_size,Img_size,1)
    model_output = model.predict([img_data])
    #print(model_output)
    if np.argmax(model_output) == 0: str_label =  'resized_Jacques_Chirac'                      #'resized_Ariel_Sharon'
    elif np.argmax(model_output) == 1: str_label =  'resized_Colin_Powell'
    elif np.argmax(model_output) == 2: str_label =  'resized_George_W_Bush'
    elif np.argmax(model_output) == 3: str_label =  'resized_Gerhard_Schroeder'
    elif np.argmax(model_output) == 4: str_label = 'resized_Tony_Blair'
#    elif np.argmax(model_output) == 5: str_label = 'resized_Ariel_Sharon'
#    elif np.argmax(model_output) == 6: str_label = 'Jean'
#    elif np.argmax(model_output) == 7: str_label = 'John_Ashcroft'
#    elif np.argmax(model_output) == 8: str_label = 'Junichiro_Koimuzi'
#    elif np.argmax(model_output) == 9: str_label = 'Serena_Williams'
#    elif np.argmax(model_output) == 10: str_label = 'Tony_Blair'
#    
    else: str_label = 'cat'
    
    if np.argmax(img_num) == 0: act_label =  'resized_Jacques_Chirac'                       #'resized_Ariel_Sharon'
    elif np.argmax(img_num) == 1: act_label = 'resized_Colin_Powell'
    elif np.argmax(img_num) == 2: act_label = 'resized_George_W_Bush'
    elif np.argmax(img_num) == 3: act_label ='resized_Gerhard_Schroeder'
    elif np.argmax(img_num) == 4: act_label = 'resized_Tony_Blair'
#    elif np.argmax(img_num) == 5: act_label = 'resized_Ariel_Sharon'
#    elif np.argmax(img_num) == 6: act_label = 'Jean'
#    elif np.argmax(img_num) == 7: act_label = 'John_Ashcroft'
#    elif np.argmax(img_num) == 8: act_label = 'Junichiro_Koimuzi'
#    elif np.argmax(img_num) == 9: act_label = 'Serena_Williams'
#    elif np.argmax(img_num) == 10: act_label = 'Tony_Blair'
    else: str_label = 'cat'
    out.append(act_label)
    
    y = plt.imshow(og,cmap='gray')
    plt.title(str_label)
    plt.xlabel(act_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
print(out)
#new_FR.py
#Displaying new_FR.py.
#%%
predictions=model.predict(X_test)
y_pred= (predictions>0.80)
matrix= confusion_matrix(Y_test.argmax(axis=1),y_pred.argmax(axis=1))
print(classification_report(Y_test.argmax(axis=1),y_pred.argmax(axis=1)))
print(matrix)
