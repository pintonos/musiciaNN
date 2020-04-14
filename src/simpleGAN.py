import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from keras import Sequential
from keras.layers import Conv1D, Dense
from tensorflow.keras import layers
import time

from IPython import display

print(tf.__version__)

TENSOR_SIZE = 16384
directory = "../data/"
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(sounds):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_sounds = generator(noise, training=True)

        real_output = discriminator(sounds, training=True)
        fake_output = discriminator(generated_sounds, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for sound_batch in dataset:
            train_step(sound_batch)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_sounds(generator,
                             epochs,
                             seed)


def generate_and_save_sounds(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    prediction = model(test_input, training=False)
    pred_sound = tf.reshape(prediction[0], [TENSOR_SIZE, 1])

    raw = tf.audio.encode_wav(pred_sound, tf.constant(44000))
    tf.io.write_file("./test.wav", raw)

# define the standalone generator model
def define_generator(n_outputs=TENSOR_SIZE):
    model = tf.keras.Sequential()
    model.add(layers.Dense(15, use_bias=False, input_dim=100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(n_outputs, activation='linear'))
    return model


# define the standalone discriminator model
def define_discriminator(n_inputs=TENSOR_SIZE):
    model = tf.keras.Sequential()
    model.add(layers.Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


data = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        raw_audio = tf.io.read_file(os.path.join(directory, filename))
        waveform = tf.audio.decode_wav(raw_audio)
        data.append(tf.reshape(waveform[0], [1, TENSOR_SIZE]))

BUFFER_SIZE = 60000
BATCH_SIZE = 128

print(len(data))

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_dataset)

generator = define_generator()
noise = tf.random.normal([1, 100])
generated_sound = generator(noise)
print(generated_sound)
discriminator = define_discriminator()
decision = discriminator(generated_sound)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
print("train...")
train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
