
import numpy as np
import scipy.io

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras import backend as K


print('loading data ...')
data = scipy.io.loadmat('data/PAMAP2_f')
print('including MLP/CNN/LSTM/ConvLSTM')
X_train = data['X_train']
X_valid = data['X_valid']
X_test = data['X_test']
y_train = data['y_train'].reshape(-1).astype(np.uint8)
y_valid = data['y_valid'].reshape(-1).astype(np.uint8)
y_test = data['y_test'].reshape(-1).astype(np.uint8)

#print('merging validation set and test set into one')
X_valid = np.concatenate((X_valid, X_test), axis=0)
y_valid = np.concatenate((y_valid, y_test), axis=0)

num_classes = 12 # 12 classes for PAMAP2
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)



#specifying hyper-parameters
batch_size = 128
feat_map_num = 16
_, win_len, dim = X_train.shape
#network_type = 'CNN'
network_type = 'ConvLSTM'
#network_type = 'LSTM'
#network_type = 'MLP'


# useful functions
def _data_reshaping(X_train, X_valid, network_type):
    _, win_len, dim = X_train.shape
    print(network_type)
    if network_type=='CNN' or network_type=='ConvLSTM':
        
        # make it into (frame_number, dimension, window_size, channel=1) for convNet
        X_train = np.swapaxes(X_train,1,2)
        X_valid = np.swapaxes(X_valid,1,2)

        X_train = np.reshape(X_train, (-1, dim, win_len, 1))
        X_valid = np.reshape(X_valid, (-1, dim, win_len, 1))
    if network_type=='MLP':
        X_train = np.reshape(X_train, (-1, dim*win_len))
        X_valid = np.reshape(X_valid, (-1, dim*win_len))
    
    return X_train, X_valid

def model_variant(model, network_type):
    print(network_type)
    if network_type == 'ConvLSTM':
        
        
        model.add(Permute((2, 1, 3))) # for swap-dimension
        model.add(Reshape((-1,feat_map_num*dim)))
        model.add(LSTM(32, return_sequences=False, stateful=False))
        model.add(Dropout(0.5))
    if network_type == 'CNN':
        
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))

        
def model_conv(model):
    model.add(Conv2D(feat_map_num, kernel_size=(1, 5),
                 activation='relu',
                 input_shape=(dim, win_len, 1),
                 padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(feat_map_num, kernel_size=(1, 5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    
def model_LSTM(model):
    model.add(LSTM(64, 
               input_shape=(win_len,dim), 
               return_sequences=True, 
               stateful=False))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, stateful=False))
    model.add(Dropout(0.5))

def model_MLP(model):
    model.add(Dense(128, activation='relu', input_shape=(dim*win_len,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    
print('reshaping data for different models ...')
X_train, X_valid = _data_reshaping(X_train, X_valid, network_type)




    

        

print('building the model ...')
model = Sequential()
if network_type=='CNN' or network_type=='ConvLSTM':
    model_conv(model)
    model_variant(model, network_type)
if network_type=='LSTM':
    model_LSTM(model)
if network_type=='MLP': 
    model_MLP(model)
model.add(Dense(num_classes, activation='softmax'))
model.summary()


print('model training ...')
epochs = 3
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid))



from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

y_pred = np.argmax(model.predict(X_valid), axis=1)
y_true = np.argmax(y_valid, axis=1)
print('calculating confusion matrix ... ')
cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)
class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None)*100)*0.01
print('the mean f1 score:{:.2f}'.format(np.mean(class_wise_f1)))



