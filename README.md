# CarND-Behavioral-Cloning
Behavioural learning project for Udacity's self-driving car nanodegree program.

## Python environment
`conda env create -f environment.yml`
For GPU additionally we need to install `pip install tensorflow-gpu`

`conda install -c https://conda.anaconda.org/menpo opencv3`


## Data
Initially I am using the provided dataset as I do not have a joystick to generate reliable data with the driving simulator.
Ensure you have a sub-directory called data to contain the data and logs for tensorflow event logs to be written to (for use by tensorboard).

To get the data directory just unzip the dataset which already has the contents in folders data/IMG.
Dataset: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

First working model was based on images resized down to 32 rows x 16 columns. Even though the model was flawless on track 1 the driving behaviour was quite wobbly and it failed part way around track 2. I believe I was losing too much information at this image size so moved up to 32 x 32.


## Model Architecture
The base of this model is from the Keras lab for traffic sign classification (See the Jupyter notebook traffic-sign-classification-with-keras.ipynb here https://github.com/js1972/CarND-Keras-Traffic-Sign-Classifier). I added to my final model in this lab one more convolutional layer and one more fully connected layer.

The final architecture is:
![Model layers](save/model.png "Keras Model Architecture")


## Training
Run the model with default parameters: `python model.py`
The default params are:
-

While training you can check the logs with tensorbard `tensorboard --logdir=./logs` (but clear out the logdir first).

Current trained model https://www.dropbox.com/s/xecw5h6zq94zo9v/model.h5?dl=0

If the model has been trained on AWS you can copy the model to your local PC with: `scp carnd@<aws ip>:/home/carnd/CarND-Behavioral-Cloning/save/model.h5 save/model.h5` and likewise with the model definition which is saved to model.json.

## Prediction
To run the trained model on the simulator, first start up the Udacity driving simulator app in autonomous mode.
Then run the following at the terminal:
```
python drive.py save/model.json
```

## Notes
Initially I used a model from the Keras Traffic Signs lab as a base. With pre-processing I started out switching the images to grayscale thinking it would be easier for the model to train on - but in hindsight I believe this removed valuable information for the convolutional layers to use so I switched back to colour images.
