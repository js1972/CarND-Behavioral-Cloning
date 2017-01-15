import numpy as np
from scipy.misc import imread, imresize, toimage
import cv2
from keras.preprocessing.image import img_to_array, load_img


def read_images(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)
        #image = load_img(path, target_size=(160, 320))
        #imgs[i] = img_to_array(image)

    return imgs


def resize(imgs, shape=(32, 16, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)
        #imgs_resized[i] = cv2.resize(img, (16, 32))

    return imgs_resized


def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)


def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1


def preprocess(imgs):
    imgs_processed = resize(imgs)
    #imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed


def random_flip(imgs, angles):
    """
    Augment the data by randomly flipping some angles / images horizontally.
    """
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles


def augment_brightness(images):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    new_imgs = np.empty_like(images)

    for i, image in enumerate(images):
        # convert to HSV so that its easy to adjust brightness
        hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2HSV)

        # randomly generate the brightness reduction factor
        # Add a constant so that it prevents the image from being completely dark
        random_bright = .25+np.random.uniform()

        # Apply the brightness reduction to the V channel
        hsv[:,:,2] = hsv[:,:,2]*random_bright

        # convert to RBG again
        new_imgs[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return new_imgs


def augment(imgs, angles):
    augmented_brightness_imgs = augment_brightness(imgs)
    imgs_augmented, angles_augmented = random_flip(augmented_brightness_imgs, angles)

    return imgs_augmented, angles_augmented


def gen_batches(imgs, angles, batch_size):
    """
    Generates random batches of the input data.

    :param imgs: The input images.
    :param angles: The steering angles associated with each image.
    :param batch_size: The size of each minibatch.

    :yield: A tuple (images, angles), where both images and angles have batch_size elements.
    """

    while True:
        indices = np.random.choice(len(imgs), batch_size)
        batch_imgs_raw, angles_raw = read_images(imgs[indices]), angles[indices].astype(float)

        #batch_imgs, batch_angles = augment(preprocess(batch_imgs_raw), angles_raw)
        batch_imgs, batch_angles = augment(batch_imgs_raw, angles_raw)
        batch_imgs = preprocess(batch_imgs)

        yield batch_imgs, batch_angles