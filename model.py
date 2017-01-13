import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

from setuptools.sandbox import save_argv

from utils import gen_batches


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'data/IMG/', 'The directory of the image data.')
#flags.DEFINE_string('imgs_dir', 'temp/IMG/', 'The directory of the image data.')
flags.DEFINE_string('data_path', 'data/driving_log.csv', 'The path to the csv of training data.')
#flags.DEFINE_string('data_path', 'temp/driving_log.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')


def main(_):
    # Print current FLAGS
    print("Running with the following FLAGs:")
    print("Mini-batch size: {}".format(FLAGS.batch_size))
    print("Number of epochs: {}".format(FLAGS.num_epochs))

    ##
    # Load Data
    ##

    with open(FLAGS.data_path, 'r') as f:
        reader = csv.reader(f)
        # skip header
        next(reader)
        # data is a list of tuples (img path, steering angle)
        data = np.array([row for row in reader])

    # Split train and validation data
    # Note that the data file is 7 columns wide though we are only using 2 columns
    # here. We could instead pre-process the data into a clean file?!?
    np.random.shuffle(data)
    split_i = int(len(data) * 0.9)
    X_train, _, _, y_train, _, _, _ = list(zip(*data[:split_i]))
    X_val, _, _, y_val, _, _, _ = list(zip(*data[split_i:]))

    # Prepend absolute path to image filenames
    base_path = os.getcwd() + "/data/"
    #base_path = os.getcwd() + "/temp/"
    X_train = [base_path + s for s in X_train]
    X_val = [base_path + s for s in X_val]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)


    ##
    # Define Model
    ##

    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(32, 16, 3), border_mode="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, 3, 3, border_mode="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        Conv2D(128, 3, 3, border_mode="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        #Conv2D(256, 3, 3, border_mode='same', activation='relu'),
        #MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        Flatten(),
        Dense(1024, activation="relu"),
        Dense(512, activation="relu"),
        #Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, name="output", activation="linear"),  #tanh
    ])

    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss="mse")

    # Setup callbacks
    # Run tensorboard with: tensorboard --logdir=./logs (it is installed as part of TF)
    # then navigate to localhost:6006
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    save_weights = ModelCheckpoint("save/model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False)
    model.summary()

    ##
    # Train
    ##

    print("X_train size:", len(X_train))

    samples_per_epoch = (len(X_train) // FLAGS.batch_size) * FLAGS.batch_size
    print("samples_per_epoch:", samples_per_epoch)
    print("\n")

    history = model.fit_generator(gen_batches(X_train, y_train, FLAGS.batch_size),
                                  samples_per_epoch,
                                  FLAGS.num_epochs,
                                  callbacks=[early_stopping, save_weights, tensorboard],
                                  validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),
                                  nb_val_samples=len(X_val))

    ##
    # Save model - note that the best weights are already saved above with the save_weights callback
    ##

    json = model.to_json()
    #model.save_weights("save/model.h5")
    with open("save/model.json", "w") as f:
        f.write(json)


if __name__ == '__main__':
    tf.app.run()
