import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, RMSprop
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from mymodule.conv.minivggnet import MiniVGGNet

if not os.path.exists('cifar10/minivggnet_aug'):
    os.mkdir('cifar10/minivggnet_aug')

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

aug = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1,
                         horizontal_flip = True, fill_mode = 'nearest')

X_train = X_train.astype('float')/255.0
X_test = X_test.astype('float')/255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])

fname = 'cifar10/minivggnet_aug/weight_minivggnet_aug.hdf5'
checkpoint = ModelCheckpoint(fname, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks = [checkpoint]

H = model.fit_generator(aug.flow(X_train, y_train, batch_size = 64), validation_data = [X_test, y_test],
                        epochs = 40, steps_per_epoch=len(X_train) // 64, callbacks = callbacks, verbose = 1)

y_pred = model.predict(X_test, batch_size=64)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('cifar10/minivggnet_aug/classification_report.csv', index = False)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label = 'validation_loss')
plt.plot(np.arange(0, 40), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 40), H.history['val_acc'], label = 'validation_acc')
plt.title('Loss and Accuracy')
plt.xlabel('Loss/Acc')
plt.ylabel('# epochs')
plt.legend()
plt.show
plt.savefig('cifar10/minivggnet_aug/plot.jpg')

# continuing training

opt_2 = RMSprop(lr = 0.001)

model_2 = load_model('cifar10/minivggnet_aug/weight_minivggnet_aug.hdf5')
model_2.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

fname_2 = 'cifar10/minivggnet_aug/weight_minivggnet_aug_2.hdf5'
checkpoint_2 = ModelCheckpoint(fname_2, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks_2 = [checkpoint_2]

H_2 = model_2.fit_generator(aug.flow(X_train, y_train, batch_size = 64), validation_data = [X_test, y_test],
                        epochs = 15, steps_per_epoch=len(X_train) // 64, callbacks = callbacks_2, verbose = 1)

y_pred_2 = model_2.predict(X_test, batch_size=64)

print(classification_report(y_test.argmax(axis=1), y_pred_2.argmax(axis=1)))

report_2 = classification_report(y_test.argmax(axis = 1), y_pred_2.argmax(axis = 1), output_dict = True)
df_2 = pd.DataFrame(report_2).transpose()
df_2.to_csv('cifar10/minivggnet_aug/classification_report_2.csv', index = False)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), H_2.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 15), H_2.history['val_loss'], label = 'validation_loss')
plt.plot(np.arange(0, 15), H_2.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 15), H_2.history['val_acc'], label = 'validation_acc')
plt.title('Loss and Accuracy')
plt.xlabel('Loss/Acc')
plt.ylabel('# epochs')
plt.legend()
plt.show
plt.savefig('cifar10/minivggnet_aug/plot_2.jpg')






















