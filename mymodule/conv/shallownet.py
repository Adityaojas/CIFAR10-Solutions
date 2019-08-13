from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense, Activation
from keras import backend as K

class ShallowNet:
    def build(width, height, depth, classes):
        
        model = Sequential()
        input_shape = (height, width, depth)
        
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            
        model.add(Conv2D(32, (3,3), input_shape = input_shape, padding = 'same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
    
        
        
        
        

