""""
With admiration for and inspiration from:
    https://github.com/dolaameng/Udacity-SDC_Behavior-Cloning/
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
Accompanies the blog post at medium.com/@harvitronix
"""
import csv, random, numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift


def model(load, shape):
    """
    Return a model from file or to train on.
    """

    #if load: return load_model('checkpoints/short.h5')

    conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=shape))
    model.add(MaxPooling2D())
    for cl in conv_layers:
        model.add(Convolution2D(cl, 3, 3, activation='elu'))
        model.add(MaxPooling2D())
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model


def get_X_y(data_dir, X=[], y=[]):
    """
    Read the log file and turn it into X/y pairs using only the left and
    right cameras and ignoring low speed entries.
    """

    with open(data_dir + "driving_log.csv") as f:
        log = list(csv.reader(f))

    for row in log:
        if float(row[6]) < 20: continue  # throw away low-speed samples
        X += [row[1].strip(), row[2].strip()]
        y += [float(row[3]) + 0.4, float(row[3]) - 0.4]

    return X, y


def process_image(path, sa, augment=False, shape=(100, 100, 3)):
    """
    Process the image.
    """

    image = load_img("session_data/" + path, target_size=(shape[0], shape[1]))

    if augment and random.random() < 0.5:
        image = random_darken(image)  # before numpy'd

    image = img_to_array(image)

    if augment:
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            sa = -sa

    image = (image / 255. - .5).astype(np.float32)
    return image, sa


def random_darken(image):
    """
    Given an image (from Image.open), randomly darken a part of it.
    """

    w, h = image.size

    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    # Loop through every pixel of our box (*GASP*) and darken.
    for i in range(x1, x2):
        for j in range(y1, y2):
            new_value = tuple([int(x * 0.5) for x in image.getpixel((i, j))])
            image.putpixel((i, j), new_value)
    return image


def _generator(batch_size, X, y):
    """
    Generate batches of training data forever.
    """

    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)
            sa = y[sample_index]
            image, sa = process_image(X[sample_index], sa, augment=True)
            batch_X.append(image)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)


def train():
    """
    Load our network and our data, fit the model, save it.
    """

    net = model(load=False, shape=(100, 100, 3))
    X, y = get_X_y("session_data/")
    net.fit_generator(_generator(256, X, y), samples_per_epoch=20224, nb_epoch=2)
    #net.save("save/model.h5")

    net.save_weights("save/model.h5")
    json = net.to_json()
    with open("save/model.json", "w") as f:
        f.write(json)


if __name__ == '__main__':
    train()