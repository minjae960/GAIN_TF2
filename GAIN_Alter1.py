import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.impute import KNNImputer
import time
import random

data_name = '1_Basic_1_Seoul'
missing_col = 'ion'
missing_rate = 0.2
hint_rate = 0.9
alpha = 10
epochs = 1500
length = 720
lr = 0.001

# 1. data preparation
# 1-1) load dataset, data
if data_name == 'pm' or data_name == 'pm_wthr':
    data = pd.read_csv('data/{}.csv'.format(data_name))
else:
    data = pd.read_csv('data/{}_raw.csv'.format(data_name)).drop(columns='date')[:length]

rows, cols = data.shape

# 1-2) binary data, data_m
def missing_sampler(rate, rows, cols, type):
    unif_matrix = np.full((rows, cols), 1)

    if data_name == 'pm' or data_name == 'pm_wthr':
        col_dic = {'ocec': ['OC', 'EC'],
                   'ele': ['Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb'],
                   'ion': ['SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.'],
                   'ocec+ele': ['OC', 'EC', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb'],
                   'ocec+ion': ['OC', 'EC', 'SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.'],
                   'ele+ion': ['Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb', 'SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.'],
                   'all': ['OC', 'EC', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb', 'SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.']
                   }
    else:
        col_dic = {'ocec': ['OC', 'EC'],
                   'ele': ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb'],
                   'ion': ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
                   'ocec+ele': ['OC', 'EC', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb'],
                   'ocec+ion': ['OC', 'EC', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
                   'ele+ion': ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
                   'all': ['OC', 'EC', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']
                   }

    if type == 'data_m':
        random_row = data.sample(int(len(data) * rate), random_state=777).index.tolist()
    elif type == 'hint':
        missing_num = int(rows * rate)
        random_row = random.sample(range(rows), missing_num)

    data_col = data.columns.tolist()

    col_index = []
    for k in col_dic[missing_col]:
        index = data_col.index(k)
        col_index.append(index)

    unif_matrix[np.ix_(random_row, col_index)] = 0

    return unif_matrix

data_m = missing_sampler(missing_rate, rows, cols, 'data_m')

# 1-3) data_x and miss_data_x
data_ori_x = data.to_numpy()
data_miss_x = data_ori_x.copy()
data_miss_x[data_m==0] = np.nan

# 1-4) normalize data_miss_x
min_vector = np.nanmin(data_miss_x, axis=0)
max_vector = np.nanmax(data_miss_x, axis=0)

data_norm_x = (data_miss_x - min_vector)/(max_vector - min_vector)
train_x = data_norm_x.copy()
train_x = np.nan_to_num(train_x, 0)


# 2. GAIN
# 2-1) architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(cols, kernel_initializer='glorot_normal', input_shape=(None, cols*2), return_sequences=True))
    model.add(layers.LSTM(cols, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(layers.Dense(cols, activation='sigmoid', kernel_initializer = 'glorot_normal'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(cols, kernel_initializer='glorot_normal', input_shape=(None, cols*2), return_sequences=True))
    model.add(layers.LSTM(cols, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(layers.Dense(cols, activation='sigmoid', kernel_initializer='glorot_normal'))

    return model

# 2-2) loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(batch_m, D_output):
    loss = cross_entropy(batch_m, D_output)
    return loss

def G_loss(batch_x, batch_m, G_output, D_output):
    binary_loss = cross_entropy(tf.ones_like(D_output[batch_m==0]), D_output[batch_m==0])
    mse_loss = tf.keras.losses.MSE(batch_x[batch_m==1], G_output[batch_m==1]) / (1-missing_rate)

    total_loss = binary_loss + alpha * mse_loss
    return total_loss

# 2-3) optimizer
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

# 2-4) train function
generator = make_generator_model()
discriminator = make_discriminator_model()

def train_step(batch_x, batch_m):
    Z_mb = np.random.uniform(0, 0.01, size = [1, rows, cols])
    H_mb = batch_m * missing_sampler(1-hint_rate, rows, cols, 'hint')
    X_mb = batch_m * batch_x + (1-batch_m) * Z_mb

    G_input = tf.concat(values = [X_mb, batch_m], axis = 2)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        G_output = generator(G_input, training=True)

        D_input = batch_m * batch_x + (1-batch_m) * G_output
        D_input = tf.concat(values = [D_input, H_mb], axis = 2)
        D_output = discriminator(D_input, training=True)

        gen_loss = G_loss(batch_x, batch_m, G_output, D_output)
        disc_loss = D_loss(batch_m, D_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# 2-5) batch function
def train_lstm_gain(epochs): # inputting (1, sq, cols) size data with random order
    batch_x = train_x.reshape(-1, rows, cols)
    batch_m = data_m.reshape(-1, rows, cols)

    for epoch in tqdm(range(epochs)):
        start = time.time()

        gen_loss, disc_loss = train_step(batch_x, batch_m)

        if (epoch+1)%20 == 0:
            test_z = np.random.uniform(0, 0.01, size=[rows, cols])
            test_x = data_m * train_x + (1 - data_m) * test_z
            test_x = np.concatenate((test_x, data_m), axis=1)
            test_x_sq = test_x.reshape(-1, rows, cols * 2)

            GAIN_imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1 - data_m)
            GAIN_imputed = GAIN_imputed * (max_vector - min_vector) + min_vector

            REAL = data_ori_x[data_m == 0]
            GAIN = GAIN_imputed[data_m == 0]

            slope_GAIN, intercept_GAIN, r_value_GAIN, p_value_GAIN, std_err_GAIN = stats.linregress(REAL, GAIN)

            print()
            print('Time for {} epochs is {} sec'.format(epoch + 1, time.time() - start))
            print('G_loss is {} and D_loss is {}'.format(gen_loss, disc_loss))
            print('GAIN r-square value is', round(r_value_GAIN ** 2, 4))
            print()


# 2-6) train model
train_lstm_gain(epochs)


# 3. Output (GAIN, KNN)
# 3-1) GAIN imputed data
test_z = np.random.uniform(0, 0.01, size = [rows, cols])

test_x = data_m * train_x + (1-data_m) * test_z
test_x = np.concatenate((test_x, data_m), axis=1)
test_x_sq = test_x.reshape(-1, rows, cols*2)

GAIN_imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1-data_m)
GAIN_imputed = np.array(GAIN_imputed)

# 3-2) KNN imputed data
imputer = KNNImputer(n_neighbors = 3)
KNN_imputed = imputer.fit_transform(data_norm_x)

# 3-3) denormalization
GAIN_imputed = GAIN_imputed * (max_vector - min_vector) + min_vector
KNN_imputed = KNN_imputed * (max_vector - min_vector) + min_vector

# 3-4) r-square value
def r_square():
    REAL = data_ori_x[data_m == 0]
    GAIN = GAIN_imputed[data_m == 0]
    KNN = KNN_imputed[data_m == 0]

    slope_GAIN, intercept_GAIN, r_value_GAIN, p_value_GAIN, std_err_GAIN = stats.linregress(REAL, GAIN)
    slope_KNN, intercept_KNN, r_value_KNN, p_value_KNN, std_err_KNN = stats.linregress(REAL, KNN)

    print()
    print('GAIN r-square value is', round(r_value_GAIN ** 2, 4))
    print('KNN r-square value is', round(r_value_KNN ** 2, 4))

r_square()