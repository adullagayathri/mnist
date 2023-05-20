#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy
import tensorflow as tf   
from tensorflow import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
import pandas as pd
from keras.layers.core import Activation
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


# In[6]:


K.set_image_data_format('channels_last')
numpy.random.seed(0)


# In[7]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[9]:


print('The shape of the training inputs:', X_train.shape)
print('The shape of the training labels:',y_train.shape)
print('The shape of the testing inputs:',X_test.shape)
print('The shape of the testing labels:',y_test.shape)


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2 , random_state=42)


# In[12]:


X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')


# In[13]:


import matplotlib.pyplot as plt
print("the number of training examples = %i" % X_train.shape[0])
print("the number of classes = %i" % len(numpy.unique(y_train)))
print("Dimention of images = {:d} x {:d}  ".format(X_train[1].shape[0],X_train[1].shape[1])  )

#This line will allow us to know the number of occurrences of each specific class in the data
unique, count= numpy.unique(y_train, return_counts=True)
print("The number of occuranc of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
 
images_and_labels = list(zip(X_train,  y_train))
for index, (image, label) in enumerate(images_and_labels[:12]):
    plt.subplot(5, 4, index + 1)
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('label: %i' % label )


# In[14]:


from keras.layers import Dropout

model = Sequential()

model.add(Conv2D(100, kernel_size=3, padding="valid", input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(units= 500, activation='relu'  ))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[15]:


y_train = np_utils.to_categorical(y_train).astype('int32')
y_test = np_utils.to_categorical(y_test)


# In[16]:


from tensorflow import keras

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=25,
        verbose=1)
]


# In[17]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )

datagen.fit(X_train)

train_generator = datagen.flow(X_train, y_train, batch_size=60, subset='training')

validation_generator = datagen.flow(X_train, y_train, batch_size=60, subset='validation')



history = model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 60,
                    validation_steps = len(validation_generator) / 60,
                    epochs = 300,
                    workers=-1)


# In[18]:


loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)


# In[19]:


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[21]:


prediction = model.predict([X_test])
prediction


# In[23]:


import numpy as np
print('Probabilities: ', prediction[10])
print('\n')
print('Prediction: ', np.argmax(prediction[10]))


# In[24]:


plt.imshow(X_test[100])
plt.show()


# In[25]:


import pandas as pd
history_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
})


history_df.to_excel('accuracy_Mnist.xlsx', index=False)


# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
from IPython.display import SVG


# In[27]:


plot_model(model, to_file='model_mnist.png', show_shapes=True, show_layer_names=True)



# In[ ]:




