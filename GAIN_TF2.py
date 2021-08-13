import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

data_name = 'pm'
missing_col = 'ele'
missing_rate = 0.2
hint_rate = 0.9
alpha = 100
batch_size = 58
epochs = 100

# 1. data preparation
## 1-1) load dataset, data
data = pd.read_csv('data/{}.csv'.format(data_name))
rows, cols = data.shape

## 1-2) binary data, data_m
def missing_sampler(rate, rows, cols):
  unif_matrix = np.full((rows, cols), 1)

  num_rows = int(rows*rate)

  col_dic = {'ocec': ['OC', 'EC'],
             'ele': ['Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb'],
             'ion': ['SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.'],
             'ocec+ele': ['OC', 'EC', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb'],
             'ocec+ion': ['OC', 'EC', 'SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.'],
             'ele+ion': ['SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb'],
             'all': ['OC', 'EC', 'SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb']
             }

  random_row = random.sample(range(rows), num_rows)
  data_col = data.columns.tolist()

  col_index = []
  for k in col_dic[missing_col]:
    index = data_col.index(k)
    col_index.append(index)

  unif_matrix[np.ix_(random_row, col_index)] = 0

  return unif_matrix

data_m = missing_sampler(missing_rate, rows, cols)

## 1-3) data_x and miss_data_x
data_x = data.to_numpy()
miss_data_x = data_x.copy()
miss_data_x[data_m==0] = np.nan

## 1-4) normalize miss_data_x
min_vector = np.nanmin(miss_data_x, axis=0)
max_vector = np.nanmax(miss_data_x, axis=0)

miss_norm_x = (miss_data_x - min_vector)/(max_vector - min_vector)
miss_norm_x = np.nan_to_num(miss_norm_x, 0)


# 2. GAIN
## 2-1) architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(cols, activation='relu', input_shape=(batch_size, cols*2)))
    model.add(layers.Dense(cols, activation='relu'))
    model.add(layers.Dense(cols, activation='sigmoid'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(cols, activation='relu', input_shape=(batch_size, cols*2)))
    model.add(layers.Dense(cols, activation='relu'))
    model.add(layers.Dense(cols, activation='sigmoid'))

    return model

## 2-2) loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(batch_m, D_output):
    loss = cross_entropy(batch_m, D_output)
    return loss

def G_loss(batch_x, batch_m, G_output, D_output):
    binary_loss = cross_entropy(tf.ones_like(D_output[batch_m==0]), D_output[batch_m==0])
    mse_loss = tf.keras.losses.MSE(batch_x[batch_m==1], G_output[batch_m==1])

    total_loss = binary_loss + alpha * mse_loss
    return total_loss

## 2-3) optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

## 2-4) train
generator = make_generator_model()
discriminator = make_discriminator_model()

def train_step(batch_x, batch_m):
    Z_mb = np.random.uniform(0, 0.01, size = [batch_size, cols])
    H_mb = batch_m * missing_sampler(1-hint_rate, batch_size, cols)
    X_mb = batch_m * batch_x + (1-batch_m) * Z_mb

    G_input = tf.concat(values = [X_mb, batch_m], axis = 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        G_output = generator(G_input, training=True)

        D_input = batch_m * batch_x + (1-batch_m) * G_output
        D_input = tf.concat(values = [D_input, H_mb], axis = 1)
        D_output = discriminator(D_input, training=True)

        gen_loss = G_loss(batch_x, batch_m, G_output, D_output)
        disc_loss = D_loss(batch_m, D_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train_GAN(epochs):
    for epoch in range(epochs):
        start = time.time()

        for batch_idx in tqdm(range(int(rows/batch_size))):
            batch_x = miss_norm_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch_m = data_m[batch_idx*batch_size:(batch_idx+1)*batch_size]

            gen_loss, disc_loss = train_step(batch_x, batch_m)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Generator loss is {} and discriminator loss is {}'.format(gen_loss, disc_loss))

train_GAN(epochs)

# 3. Output
## 3-1) imputed data
test_z = np.random.uniform(0, 0.01, size = [rows, cols])
test_x = data_m * miss_norm_x + (1-data_m) * test_z
input_x =