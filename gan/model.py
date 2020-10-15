import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, LeakyReLU, BatchNormalization, Reshape
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.losses import BinaryCrossentropy
from keras.regularizers import l2
import argparse

img_shape = (28, 28, 1) # shape of image
LEARNING_RATE = 0.0002 # learning rate of the model
adam = Adam(LEARNING_RATE, 0.5) # adam optimizer
sgd = SGD(LEARNING_RATE, momentum = 0.9) # sgd optimizer
LATENT_DIM = 128 # dimension of the latent space
EPOCHS = 100 # number of epochs for which the model is trained
BATCH_SIZE = 128 # number of samples to process per minibatch during training
OPTIMIZER = adam # optimizer for the model
WEIGHT_DECAY = 0 # magnitude of weight decay

# function to get the arguments
def get_arguments():

    parser = argparse.ArgumentParser("Generative Adverserial Network for generating handwritten digits using MNIST dataset")
    parser.add_argument("--e", type = int, metavar = "Epochs", default = EPOCHS, help = "Number of epochs for which the model is trained")
    parser.add_argument("--bs", type =  int, metavar = "Batch Size", default = BATCH_SIZE, help = "Number of samples to process per minibatch during training")
    parser.add_argument("--lr", type = float, metavar = "Learning Rate", default = LEARNING_RATE, help = "Learning Rate for Adam optimizer")
    parser.add_argument("--ld", type = int, metavar = "Latent Dimension", default = LATENT_DIM, help = "Dimension of the latent space from which images are generated")
    parser.add_argument("--opt", type = str, metavar = "Optimizer", default = OPTIMIZER, help = "Optimizer for training the model. Either 'adam' or 'sgd'")
    parser.add_argument("--wd", type = float, metavar = "Weight Decay", default = WEIGHT_DECAY, help = "Weight decay using l2 regularizer")
    return parser.parse_args()

# function to assign the arguments to the variables
def get_hyperparameters():
    args = get_arguments()
    global LATENT_DIM, SAVE_INTERVAL, EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, adam, sgd
    LATENT_DIM = args.ld
    EPOCHS = args.e
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.bs
    WEIGHT_DECAY = args.wd

    # checking if the optimizer input is valid
    if args.opt == "adam":
        OPTIMIZER = adam
    elif args.opt == "sgd":
        OPTIMIZER = sgd
    else:
        print("Invalid input for Optimizer, using default optimizer.")

# function to define the generator
def get_generator():

    model = Sequential()

    # first block. input dimension: LATENT_DIM      output dimension: 128
    model.add(Dense(128, input_dim = LATENT_DIM, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))

    # second block. input dimension: 128    output dimension: 256
    model.add(Dense(256, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))

    # third block. input dimension: 256     output dimension: 512
    model.add(Dense(512, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))

    # fourth block. input dimension: 512    output dimension: 1024
    model.add(Dense(1024, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))

    # final block. input dimension: 1024    output dimension: 784
    model.add(Dense(np.prod(img_shape), activation = "sigmoid", kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(Reshape(img_shape))

    return model

# funciton to define the discriminator
def get_discriminator():

    model = Sequential()

    # converts from 3-dimension to 1-dimension
    model.add(Flatten(input_shape = img_shape))

    # first block. input dimension: 784     output dimension: 512
    model.add(Dense(512, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))

    # second block. input dimension: 512    output dimension: 256
    model.add(Dense(256, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))

    # third block. input dimension: 256     output dimension: 128
    model.add(Dense(128, kernel_regularizer = l2(WEIGHT_DECAY)))
    model.add(LeakyReLU(alpha = 0.2))

    # final block. input dimension: 128     output dimension: 1
    model.add(Dense(1, activation = "sigmoid", kernel_regularizer = l2(WEIGHT_DECAY)))

    model.compile(loss = BinaryCrossentropy(), optimizer = OPTIMIZER, metrics = ["accuracy"])
    return model

# function to define the gan model
def get_gan(gen, disc):

    # disabling the training of the model
    disc.trainable = False

    # combining the generator and discriminator model
    model = Sequential()
    model.add(gen)
    model.add(disc)

    model.compile(loss = BinaryCrossentropy(), optimizer = OPTIMIZER)
    return model

# function to load data
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

# function to save 25 samples generated using the model for visualization purposes
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
