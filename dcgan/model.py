from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Activation, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU, ZeroPadding2D, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import argparse

img_shape = (28, 28, 1) # shape of the image
EPOCHS = 150 # number of epochs for which the model is trained
BATCH_SIZE = 128 # number of samples to process per minibatch during training
LATENT_DIM = 100 # dimension of the latent space from which images are genrated
LEARNING_RATE = 0.0001 # learning rate for the optimizer
adam = Adam(LEARNING_RATE, 0.5) # adam optimizer
sgd = SGD(LEARNING_RATE, momentum = 0.9) # sgd optimizer
OPTIMIZER = adam # optimizer for training the model

# function to get the arguments
def get_arguments():

    parser = argparse.ArgumentParser("DCGAN model for generating handwritten digits using MNIST dataset.")
    parser.add_argument("--e", type = int, metavar = "Epochs", default = EPOCHS, help = "Number of epochs for which the model is trained")
    parser.add_argument("--bs", type = int, metavar = "Batch Size", default = BATCH_SIZE, help = "Number of samples to process per minibatch during training")
    parser.add_argument("--lr", type = float, metavar = "Learning Rate", default = LEARNING_RATE, help = "Learning Rate for optimizer")
    parser.add_argument("--ld", type = int, metavar = "Latent Dimension", default = LATENT_DIM, help = "Dimension of the latent space from which images are generated")
    parser.add_argument("--opt", type = str, metavar = "Optimizer", default = OPTIMIZER, help = "Optimizer for training the model. Either 'adam' or 'sgd'")
    return parser.parse_args()

# function to assign the arguments to the variables
def get_hyperparameters():

    args = get_arguments()
    global LATENT_DIM, SAVE_INTERVAL, EPOCHS, LEARNING_RATE, BATCH_SIZE, adam, sgd
    LATENT_DIM = args.ld
    EPOCHS = args.e
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.bs

    if args.opt == "adam":
        OPTIMIZER = adam
    elif args.opt == "sgd":
        OPTIMIZER = sgd
    else:
        print("Invalid input for Optimizer, using default optimizer.")

# function to define the generator model
def get_generator():

    model = Sequential()

    # output dimension: (7, 7, 256)
    model.add(Dense(256*7*7, input_dim=LATENT_DIM))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Reshape((7, 7, 256)))

    # output dimension: (7, 7, 128)
    model.add(Conv2D(128, (5, 5), padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # output dimension: (14, 14, 64)
    model.add(UpSampling2D())
    model.add(Conv2D(64, (5, 5), padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # output dimension: (28, 28, 1)
    model.add(UpSampling2D())
    model.add(Conv2D(1, (5, 5), padding = "same"))
    model.add(Activation("tanh"))

    return model

# function to define the discriminator model
def get_discriminator():

    model = Sequential()

    # output dimension: (14, 14, 64)
    model.add(Conv2D(64, (5, 5), padding = "same", strides = (2, 2), input_shape = img_shape)
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    # output dimension: (7, 7, 128)
    model.add(Conv2D(128, (5, 5), padding = "same", strides = (1, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    # output dimension: (7*7*128)
    model.add(Flatten())

    # output size: (1)
    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = BinaryCrossentropy(), optimizer = OPTIMIZER, metrics = ["accuracy"])
    return model

# function to define the gan model
def get_gan(gen, disc):

    disc.trainable = False

    model = Sequential()
    model.add(gen)
    model.add(disc)

    model.compile(loss = BinaryCrossentropy(), optimizer = OPTIMIZER)
    return model

# function to load the data
def get_data():

    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 255.0
    X_train = np.expand_dims(X_train, axis = -1)
    np.random.shuffle(X_train)

    return X_train

# function to get the models from their defining functions
def get_models():

    disc = get_discriminator()
    gen = get_generator()
    gan = get_gan(gen, disc)

    return disc, gen, gan

# function to save the generator model and 25 samples generated using it for visualization purposes
def save_image_generator(gen, epoch):

    row, col = 5, 5
    noise = np.random.normal(0, 1, (row * col, LATENT_DIM)) # noise matrix
    gen_imgs = gen.predict(noise) # generating images using the generator

    fig, axs = plt.subplots(row, col)
    ctr = 0
    for i in range(row):
        for j in range(col):
            axs[i,j].imshow(gen_imgs[ctr,:,:,0], cmap = "gray")
            axs[i,j].axis("off")
            ctr += 1
        fig.savefig("images/mnist_%d.png"%epoch) # save the image
        plt.close()

    gen.save("generators/generator_%d.h5"%epoch) # save the generator model

# function to train a single minibatch
def train_batch(disc, gen, gan, imgs):

    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM)) # noise matrix
    gen_imgs = gen.predict(noise) # generating images from the generator

    # evaluating the discriminator model
    disc_loss_real = disc.train_on_batch(imgs, valid)
    disc_loss_fake = disc.train_on_batch(gen_imgs, fake)
    disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

    # evaluating the generator model
    gen_loss = gan.train_on_batch(noise, valid)

    dict = {
    "disc" : disc,
    "gen" : gen,
    "gan" : gan,
    "disc_loss" : disc_loss,
    "gen_loss" : gen_loss
    }

    return dict

# function to train the model
def train():

    get_hyperparameters()
    X = get_data()
    disc, gen, gan = get_models()

    disc.summary()
    gen.summary()
    gan.summary()

    for epoch in range(1, EPOCHS+1):

        # mean generator and mean discriminator loss for one epoch
        mean_gen_loss = 0
        mean_disc_loss = 0

        for i in range(60000 // BATCH_SIZE):

            # taking batches of images from X
            idxs = np.arange(i*BATCH_SIZE, (i+1)*BATCH_SIZE)
            img_batch = X[idxs]

            dict = train_batch(disc, gen, gan, img_batch)

            disc = dict["disc"]
            gen = dict["gen"]
            gan = dict["gan"]

            mean_gen_loss += dict["gen_loss"]
            mean_disc_loss += dict["disc_loss"]

        mean_gen_loss /= (60000 // BATCH_SIZE)
        mean_disc_loss /= (60000 // BATCH_SIZE)

        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]"%(epoch, mean_disc_loss[0], mean_disc_loss[1]*100, mean_gen_loss))

        save_image_generator(gen, epoch)

train()
