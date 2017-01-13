# CarND-Behavioral-Cloning
Behavioural learning project for Udacity's self-driving car nanodegree program.

## Data
Initially I am using the provided dataset as I do not have a joystick to generate reliable data with the driving simulator.
Ensure you have a sub-directory called data to contain the data and logs for tensorflow event logs to be written to (for use by tensorboard).
I've tracked the save directory in github so we immediately have working weights on cloning the repo.
## Model
<describe model here>


## Notes
Initially I used a model from the Keras Traffic Signs lab as a base. With pre-processing I started out switching the images to grayscale thinking it would be easier for the model to train on - but in hindsight I believe this removed valuable information for the convolutional layers to use so I switched back to colour images.
