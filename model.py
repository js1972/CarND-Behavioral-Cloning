import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

from utils import batch_generator


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("image_dir", "data/IMG/", "The directory of the image data.")
flags.DEFINE_string('data_path', "data/driving_log.csv", "The path to the csv of training data.")
#flags.DEFINE_string('data_path', 'temp/driving_log.csv', 'The path to the csv of training data.')
flags.DEFINE_integer("batch_size", 128, 'The minibatch size.')
flags.DEFINE_integer("num_epochs", 20, 'The number of epochs to train for.')
flags.DEFINE_float("lrate", 0.0001, "The learning rate for training.")
flags.DEFINE_boolean("alldata", True, "Run with ALL cameras data.")
flags.DEFINE_boolean("dropzeros", False, "Randomly drop zero steering angles data.")


def main(_):
    # Print current FLAGS
    print("Running with the following FLAGs:")
    print("Mini-batch size: {}".format(FLAGS.batch_size))
    print("Number of epochs: {}".format(FLAGS.num_epochs))
    print("Running with ALL camera angles: {}".format(FLAGS.alldata))
    print("Dropping random zero steering angles: {}".format(FLAGS.dropzeros))

    ##
    # Load Data
    ##

    file_to_process = FLAGS.data_path

    if FLAGS.alldata == True:
        # Use a file that includes center, left and right cameras - this was created with Jupyter
        # Notebook pre_process_data.ipynb.
        # With the all_data file only the center camera column is useful as its had the left/right cameras
        # added to that column. The left/right columns are now meaningless.
        file_to_process = "data/driving_log_all.csv"

    with open(file_to_process, 'r') as f:
        reader = csv.reader(f)
        # skip header
        if FLAGS.alldata == False:
            next(reader)

        # data is a list of tuples (img path, steering angle, etc.)
        data = np.array([row for row in reader])

    if FLAGS.dropzeros == True:
        # Drop rows with zero steering angle with 50% chance! This is to try and balance the fact that there are far
        # more zero entries than turning angles.
        data_with_some_zeros_removed = []
        for d in data:
            prob = np.random.random()

            if d[3].astype("float") != 0.:
                data_with_some_zeros_removed.append(d)
            elif prob < 0.5:
                data_with_some_zeros_removed.append(d)

        data = np.array(data_with_some_zeros_removed)


    ##
    # Split train and validation data
    ##

    # Note that the data file is 7 columns wide though we are only using 2 columns
    # here. We could instead pre-process the data into a clean file or use a pandas
    # dataframe.
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

        Conv2D(128, 3, 3, border_mode="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(1024, activation="relu"),
        Dropout(0.5),

        Dense(512, activation="relu"),
        Dropout(0.5),

        Dense(1, name="output", activation="linear"),
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
    # Training
    ##

    print("X_train size:", len(X_train))

    samples_per_epoch = (len(X_train) // FLAGS.batch_size) * FLAGS.batch_size
    print("samples_per_epoch:", samples_per_epoch)
    print("\n")

    history = model.fit_generator(batch_generator(X_train, y_train, FLAGS.batch_size),
                                  samples_per_epoch,
                                  FLAGS.num_epochs,
                                  callbacks=[early_stopping, save_weights, tensorboard],
                                  validation_data=batch_generator(X_val, y_val, FLAGS.batch_size),
                                  nb_val_samples=len(X_val))

    ##
    # Save model - note that the best weights are already saved above with the save_weights callback
    ##

    json = model.to_json()
    #model.save_weights("save/model.h5")
    with open("save/model.json", "w") as f:
        f.write(json)


if __name__ == "__main__":
    tf.app.run()
