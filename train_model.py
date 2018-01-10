# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import RBM_lib as RBM_lib

# Read mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale the data between 0 and 1
x_train, x_test = np.divide(x_train, 255.0), np.divide(x_test, 255.0)

# Flatten data: reshape 28*28 matrices in flat arrays of 784 elements
x_train= x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test= x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
# Binarize the data
x_train = 1*(x_train >= 0.5)
x_test = 1*(x_test >= 0.5)
print(x_train.shape, x_test.shape)

# Switches for train/retrain of pretrained model
train = True
retrain = False
# Name for model file
model_name = "models/new_model.pkl"

if train:
    if retrain:
        # Load model from file if continuing the training
        model = RBM_lib.load_model(model_name)
    else:
        # Initialize RBM model if not continuing training from previously
        # saved model
        visible_units = x_train.shape[1]
        hidden_units = 10
        model = RBM_lib.RBM(nvisible = visible_units, nhidden = hidden_units)

    # Print initial weights stats
    model.w_histogram(fname="init_hist.png")
    model.w_map(1,10, fname="init_weights.png")
    # Check initial free energy of two datasets
    # Free energy ratio should stay more or less constant and close to one
    # If the ratio is larger than one, it could mean we are overfitting our model
    Ftrain = RBM_lib.free_energy(model, x_train)
    Fval = RBM_lib.free_energy(model, x_test)
    print("Free energy ratio before training: ", Ftrain/Fval)

    # Train model

    # First round
    model.train(x_train[:, :], gibbs_sampling_steps=1,
                lrate=0.04, epochs=5, decay_rate=0.8,
                verbose=True, binary_v=True, binary_h=True)
    # Check free energy
    Ftrain = RBM_lib.free_energy(model, x_train)
    Fval = RBM_lib.free_energy(model, x_test)
    print("Free energy ratio after training: ", Ftrain / Fval)
    # Second round
    model.train(x_train[:, :], gibbs_sampling_steps=4,
                lrate=0.04, epochs=15, decay_rate=0.95,
                verbose=True, binary_v=True, binary_h=True)

    # Save model to file
    RBM_lib.save_model(model, model_name)

    # Print final weights stats
    model.w_histogram(fname="final_hist.png")
    model.w_map(1,10, fname="final_weights.png")

    # Check free energy for the final model
    Ftrain = RBM_lib.free_energy(model, x_train)
    Fval = RBM_lib.free_energy(model, x_test)
    print("Free energy ratio after training: ", Ftrain / Fval)