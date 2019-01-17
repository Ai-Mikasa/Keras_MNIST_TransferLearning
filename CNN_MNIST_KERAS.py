import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Size of batch (Number of images to train network at a single time)
batch_size = 128

# Training runs
epochs = 5

# Image dimensions
img_rows, img_cols = 28, 28

# Number of convulution filters
filters = 32

# Max pooling size
pooling_size = 2

# convolution kernel size
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

# Function to train the CNN Model 
def train_cnn_model(model, train_data, test_data, num_classes):
    x_train = train_data[0].reshape((train_data[0].shape[0],) + input_shape)
    x_test = test_data[0].reshape((test_data[0].shape[0],) + input_shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert classes into binary class matrices 
    y_train = keras.utils.to_categorical(train_data[1], num_classes)
    y_test = keras.utils.to_categorical(test_data[1], num_classes)

    # Compile the CNN
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])
    print('\n\n')


# Split data between train and test sets
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Create two datasets for seperating digits below 7 (0-6), and the rest(7-9).
x_train_lt7 = x_train[y_train < 7]
y_train_lt7 = y_train[y_train < 7]
x_test_lt7 = x_test[y_test < 7]
y_test_lt7 = y_test[y_test < 7]

x_train_gte7 = x_train[y_train >= 7]
y_train_gte7 = y_train[y_train >= 7] - 7
x_test_gte7 = x_test[y_test >= 7]
y_test_gte7 = y_test[y_test >= 7] - 7

# Defining two groups of layers within out CNN, feature extraction (convolution) and classification (dense).
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pooling_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(7),
    Activation('softmax')
]

# Merge the CNN layers into a Keras Sequencial Model
cnn_model = Sequential(feature_layers + classification_layers)

# Train model for classifiying the first dataset of digits (0-6)

print('Training the CNN on digits 0-6...\n')
train_cnn_model(cnn_model, # Model to train
                (x_train_lt7, y_train_lt7), # Training set of digits (0-6) 
                (x_test_lt7, y_test_lt7), # Testing set of digits (0-6)
                7) # Output classes

# Freeze feature layers to prevent further training
for layer in feature_layers:
    layer.trainable = False

# Change the output size of the last dense layer in the network to 3 (Digits 7-9)
cnn_model.layers.pop()
cnn_model.layers.pop()
cnn_model.add(Dense(3))
cnn_model.add(Activation('softmax'))


# Use dense layers for classification of remaining digits (7-9)
# using the same network weights obtained from previous training 
print('Classifying remaining digits 7-9 by transfer learning')
train_cnn_model(cnn_model, # Model to train
                (x_train_gte7, y_train_gte7), # Training set of digits (7-9) 
                (x_test_gte7, y_test_gte7), # Testing set of digits (7-9)
                3) # Output classes

