import numpy as np
from scipy.misc import imread, imresize, toimage
import cv2
from keras.preprocessing.image import img_to_array, load_img


def read_images(img_paths):
    """
    Use the scipy imread function to read each image into a nunmpy array

    :param img_paths: Numpy array of image paths to read
    :return: 4d Numpy array containing all the images from image_paths.
    """
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)
        #image = load_img(path, target_size=(160, 320))
        #imgs[i] = img_to_array(image)

    return imgs


def crop_and_resize(imgs, shape=(32, 16, 3)):
    """
    Crop and Resize images to given shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        cropped = img[55:135, :, :]
        imgs_resized[i] = imresize(cropped, shape)
        #imgs_resized[i] = cv2.resize(img, (16, 32))

    return imgs_resized


def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)


def rgb2hsv(imgs):
    """
    Convert RGB images array into HSV and zero-out all but the V dimension!
    """
    hsv_imgs = np.empty_like(imgs)
    for i, image in enumerate(imgs):
        hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] * 0
        hsv[:, :, 1] = hsv[:, :, 1] * 0
        hsv_imgs[i] = hsv

    return hsv_imgs


def normalize(imgs):
    """
    Normalize images between [-1, 1]. Why not [0, 1] ?
    """
    return imgs / (255.0 / 2) - 1


def preprocess(imgs):
    """
    Pre-process the images. Note that pre-processing must be applied
    for training and predictions.

    :param imgs: Numpy array of images
    :return:  Numpy array of pre-processed images
    """
    imgs_processed = crop_and_resize(imgs)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed


def random_flip(imgs, angles):
    """
    Augment the data by randomly flipping some images/angles horizontally.
    """
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.rand() > 0.5:  # 50 percent chance to see the right angle
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles


def add_random_shadow(image):
    """
    Add a random shadow to an image
    :param image:
    :return:
    """
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()

    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def add_random_shadows(imgs):
    """
    Add random shadows across the image.

    :param imgs: Numpy array of images
    :return: Numpy array of images with a random shadow
    """
    shadow_imgs = np.empty_like(imgs)

    for i, image in enumerate(imgs):
        shadow_imgs[i] = add_random_shadow(image)

    return shadow_imgs


def augment_brightness(images):
    """
    Randomly adjust brightness of provided images.

    :param images: Numpy array of images
    :return: Numpy array of brightness adjusted imahes
    """

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
    """
    Perform dynamic image augmentation by randomly adjust the provdied images
    for brightness and flipping horizontally.

    :param imgs: Numpy array of images
    :param angles: Numpy array of angles
    :return: The augmented images and angles as a tuple
    """
    augmented_shadow_imgs = add_random_shadows(imgs)
    augmented_brightness_imgs = augment_brightness(augmented_shadow_imgs)
    imgs_augmented, angles_augmented = random_flip(augmented_brightness_imgs, angles)

    return imgs_augmented, angles_augmented


def batch_generator(image_filenames, angles, batch_size):
    """
    Generates random batches of the input data by randomly selecting indices into
    the provided data; reading the raw images files, augmenting and pre-processing.

    :param imgs: Numpy array of image filenames
    :param angles: Numpy array of the steering values associated with each image
    :param batch_size: The size of each minibatch

    :yield: A tuple (images, angles), where both images and angles have batch_size elements.
    """

    while True:
        indices = np.random.choice(len(image_filenames), batch_size)
        batch_imgs_raw, angles_raw = read_images(image_filenames[indices]), angles[indices].astype(float)

        batch_imgs, batch_angles = augment(batch_imgs_raw, angles_raw)
        batch_imgs = preprocess(batch_imgs)

        yield batch_imgs, batch_angles