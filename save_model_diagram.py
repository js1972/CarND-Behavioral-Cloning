import argparse
from keras.models import model_from_json
from keras.utils.visualize_util import plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Keras model from json")
    parser.add_argument("--model", type=str, help="Path to model definition json.")
    args = parser.parse_args()

    with open(args.model, "r") as jfile:
        model = model_from_json(jfile.read())
        model.compile("adam", "mse")
        plot(model, to_file="model.png")
