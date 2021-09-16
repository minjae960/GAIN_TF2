import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.impute import KNNImputer
import random
from sklearn.metrics import r2_score

load_dir = 'data'
save_dir = 'GAIN_HP_Tuning'

data_name = '4_AP+Meteo_1_Seoul'
missing_col = 'elementals'
seed = 777

missing_rate = 0.2
hint_rate = 0.8
alpha = 10
epochs = int(2000/(missing_rate*10))
L = 240
lr = 0.0005

col_dic = {'ocec': ['OC', 'EC'],
           'elementals': ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb'],
           'ions': ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
           'ocec-elementals': ['OC', 'EC', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb'],
           'ion-ocec': ['OC', 'EC', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
           'ion-elementals': ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se',
                              'Br', 'Pb', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+'],
           'ions-ocec-elementals': ['OC', 'EC', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+', 'S',
                                    'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']
           }

g_loss = []
d_loss = []
r_sqr = []

# 1. data preparation
# 1-1) load dataset, data
D = pd.read_csv('{}/{}_raw.csv'.format(load_dir, data_name)).drop(columns='date')

R, C = D.shape

# 1-2) binary data, data_m
# 1-2-1) whole data
def missing_sampler(rate, rows, cols, type):

    unif_matrix = np.full((rows, cols), 1)

    if type == 'data_m':
        random_row = D.sample(int(len(D) * rate), random_state=seed).index.tolist()
    elif type == 'hint':
        missing_num = int(rows * rate)
        random_row = random.sample(range(rows), missing_num)

    data_col = D.columns.tolist()

    col_index = []
    for k in col_dic[missing_col]:
        index = data_col.index(k)
        col_index.append(index)

    unif_matrix[np.ix_(random_row, col_index)] = 0

    return unif_matrix

M = missing_sampler(missing_rate, R, C, 'data_m')

# 1-2-2) most recent data
data = D.to_numpy()[-1*L:]
data_m = M[-1*L:]
rows, cols = data.shape

# 1-3) data_x and miss_data_x
DATA_ori_x = data
DATA_miss_x = DATA_ori_x.copy()
DATA_miss_x[data_m==0] = np.nan

# 1-4) normalize DATA_miss_x
min_vector = np.nanmin(DATA_miss_x, axis=0)
max_vector = np.nanmax(DATA_miss_x, axis=0)

data_norm_x = (DATA_miss_x - min_vector)/(max_vector - min_vector + 1e-10)
train_x = data_norm_x.copy()
train_x = np.nan_to_num(train_x, 0)


# 2. GAIN
# 2-1) architecture
structure = 'G:LSTM_50_50, D:LSTM_50_50'

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(cols*2, kernel_initializer='glorot_normal', input_shape=(None, cols*2), return_sequences=True))
    model.add(layers.LSTM(cols*2, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(layers.LSTM(cols*4, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(layers.Dense(cols, activation='sigmoid', kernel_initializer = 'glorot_normal'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(cols*2, kernel_initializer='glorot_normal', input_shape=(None, cols*2), return_sequences=True))
    model.add(layers.LSTM(cols*2, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(layers.LSTM(cols*4, kernel_initializer='glorot_normal', return_sequences=True))
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
    Z_mb = np.random.uniform(0, 0.1, size = [1, rows, cols])
    H_mb = batch_m * missing_sampler(1-hint_rate, rows, cols, 'hint')
    X_mb = batch_m * batch_x + (1-batch_m) * Z_mb

    G_input = np.concatenate((X_mb, batch_m), axis = 2)

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

        gen_loss, disc_loss = train_step(batch_x, batch_m)

        test_z = np.random.uniform(0, 0.1, size=[rows, cols])
        test_x = data_m * train_x + (1 - data_m) * test_z
        test_x = np.concatenate((test_x, data_m), axis=1)
        test_x_sq = test_x.reshape(-1, rows, cols * 2)

        GAIN_imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1 - data_m)
        GAIN_imputed = GAIN_imputed * (max_vector - min_vector) + min_vector

        REAL = DATA_ori_x[data_m == 0].reshape(-1, len(col_dic[missing_col]))
        GAIN = np.array(GAIN_imputed[data_m == 0]).reshape(-1, len(col_dic[missing_col]))

        results_r2_GAIN = r2_score(REAL, GAIN)

        g_loss.append(gen_loss)
        d_loss.append(disc_loss)
        r_sqr.append(results_r2_GAIN)

        if (epoch+1)%50 == 0:

            print()
            print('G_loss is {} and D_loss is {}'.format(gen_loss, disc_loss))
            print('GAIN r-square value is', round(results_r2_GAIN, 4))
            print()

# 2-6) train model
train_lstm_gain(epochs)


# 3. Output (GAIN, KNN)

# 3-1) GAIN imputed data
test_z = np.random.uniform(0, 0.1, size = [rows, cols])
test_x = data_m * train_x + (1-data_m) * test_z
test_x = np.concatenate((test_x, data_m), axis=1)
test_x_sq = test_x.reshape(-1, rows, cols*2)

GAIN_imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1-data_m)
GAIN_imputed = np.array(GAIN_imputed)
GAIN_imputed = GAIN_imputed * (max_vector - min_vector) + min_vector

# 3-2) KNN imputed data
def knn(n):
    imputer = KNNImputer(n_neighbors=n)

    knn_imputed = imputer.fit_transform(data_norm_x)
    knn_imputed = knn_imputed * (max_vector - min_vector) + min_vector

    return knn_imputed

KNN_3_imputed = knn(3)
KNN_5_imputed = knn(5)
KNN_7_imputed = knn(7)

# 3-3) r-square
def r_square():
    REAL = DATA_ori_x[data_m == 0].reshape(-1, len(col_dic[missing_col]))
    GAIN = GAIN_imputed[data_m == 0].reshape(-1, len(col_dic[missing_col]))
    KNN3 = KNN_3_imputed[data_m == 0].reshape(-1, len(col_dic[missing_col]))
    KNN5 = KNN_5_imputed[data_m == 0].reshape(-1, len(col_dic[missing_col]))
    KNN7 = KNN_7_imputed[data_m == 0].reshape(-1, len(col_dic[missing_col]))

    results_r2_GAIN = r2_score(REAL, GAIN)
    results_r2_KNN3 = r2_score(REAL, KNN3)
    results_r2_KNN5 = r2_score(REAL, KNN5)
    results_r2_KNN7 = r2_score(REAL, KNN7)

    print()
    print('GAIN r-square value is', round(results_r2_GAIN, 4))
    print('KNN 3 r-square value is', round(results_r2_KNN3, 4))
    print('KNN 5 r-square value is', round(results_r2_KNN5, 4))
    print('KNN 7 r-square value is', round(results_r2_KNN7, 4))
    print()

    return results_r2_GAIN, results_r2_KNN3, results_r2_KNN5, results_r2_KNN7

gain, knn3, knn5, knn7 = r_square()

# 3-4) loss r2 plot
plt.plot(range(epochs), g_loss, label='G loss')
plt.plot(range(epochs), d_loss, label='D loss')
plt.plot(range(epochs), r_sqr, label='GAIN r-square')
plt.plot([0, epochs], [knn3, knn3], 'k--', label='kNN (k=3), R2={}'.format(round(knn3, 2)))
plt.plot([0, epochs], [knn5, knn5], 'r--', label='kNN (k=5), R2={}'.format(round(knn5, 2)))
plt.plot([0, epochs], [knn7, knn7], 'b--', label='kNN (k=7), R2={}'.format(round(knn7, 2)))
plt.legend()
plt.title('{}_{}_{}\n'
          '{}_{}_{}_{}_{}_{}\n'
          '{}'.format(data_name, missing_col, seed, missing_rate, hint_rate, alpha, epochs, L, lr, structure),
          size=10)
plt.xlabel('Iteration')
plt.ylabel('Loss and r-square')
plt.ylim(0,1.0)

plt.savefig('{}/{}_{}_{}_{}_({}_{}_{}_{}_{}_{}).png'.format(save_dir, data_name, missing_col, seed, round(gain, 6),
                                                                 missing_rate, hint_rate, alpha, epochs, L, lr))

plt.show()