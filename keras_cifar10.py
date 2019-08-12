import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle("cifar10/data_batch_1")
batch2 = unpickle("cifar10/data_batch_2")
batch3 = unpickle("cifar10/data_batch_3")
batch4 = unpickle("cifar10/data_batch_4")
batch5 = unpickle("cifar10/data_batch_5")
test_batch = unpickle("cifar10/test_batch")

def load_data0(btch):
    labels = btch[b'labels']
    imgs = btch[b'data'].reshape((-1, 32, 32, 3))
    
    res = []
    for ii in range(imgs.shape[0]):
        img = imgs[ii].copy()
        #img = np.transpose(img.flatten().reshape(3,32,32))
        img = np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3,32,32)), k=-1))
        res.append(img)
    imgs = np.stack(res)
    return labels, imgs

labels, imgs = load_data0(batch1)
imgs.shape

def load_data():
    x_train_l = []
    y_train_l = []
    for ibatch in [batch1, batch2, batch3, batch4, batch5]:
        labels, imgs = load_data0(ibatch)
        x_train_l.append(imgs)
        y_train_l.extend(labels)
    x_train = np.vstack(x_train_l)
    y_train = np.vstack(y_train_l)
    
    x_test_l = []
    y_test_l = []
    labels, imgs = load_data0(test_batch)
    x_test_l.append(imgs)
    y_test_l.extend(labels)
    x_test = np.vstack(x_test_l)
    y_test = np.vstack(y_test_l)
    
    
    return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()
del batch1, batch2, batch3, batch4, batch5, test_batch

X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0
X_train = X_train.reshape(X_train.shape[0], 3072)
X_test = X_test.reshape(X_test.shape[0], 3072)
    

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

model = Sequential()
model.add(Dense(1024, input_shape = (3072,), activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(0.01)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

H = model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_test, y_test))

y_pred = model.predict(X_test, batch_size = 32)
print(classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))

report = classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('cifar10/keras_FC/classification_report.csv', index = False)


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0,100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0,100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0,100), H.history['val_acc'], label = 'val_acc')
plt.title('Accuracy/ Loss')
plt.xlabel('acc/loss')
plt.ylabel('# of epochs')
plt.legend()
plt.savefig('cifar10/keras_FC/Acc-Loss_Plot.png')
plt.show()

















