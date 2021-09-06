import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer
import time
import random
from sklearn.metrics import r2_score

# Data_Name = ['1_Basic_1_Seoul', '1_Basic_2_BR', '1_Basic_3_Ulsan', 민재 파이참
#              '2_Informed_1_Seoul', '2_Informed_2_BR', '2_Informed_3_Ulsan', 민재 파이참
#              '3_AP_1_Seoul', '3_AP_2_BR', '3_AP_3_Ulsan', 서울, 백령: 혜리 / 울산: 민재 코랩, 끝
#              '4_AP+Meteo_1_Seoul', '4_AP+Meteo_2_BR', '4_AP+Meteo_3_Ulsan'] 서울, 백령: 민재 파이참 / 울산: 영수 코랩

Data_Name = ['1_Basic_1_Seoul', '1_Basic_2_BR', '1_Basic_3_Ulsan', '2_Informed_1_Seoul', '2_Informed_2_BR', '2_Informed_3_Ulsan']

Missing_Col = ['ocec', 'elementals', 'ions', 'ocec-elementals', 'ion-ocec', 'ion-elementals', 'ions-ocec-elementals']

Random_State = [1004, 322]


missing_rate = 0.2
hint_rate = 0.9
alpha = 10
epochs = 1500
L = 720
lr = 0.001

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

for data_name in Data_Name:
    for missing_col in Missing_Col:
        for seed in Random_State:

            print(seed, missing_col, data_name)

            # 1. data preparation
            # 1-1) load dataset, data
            if data_name == 'pm' or data_name == 'pm_wthr':
                D = pd.read_csv('data/{}.csv'.format(data_name))
            else:
                D = pd.read_csv('data/{}_raw.csv'.format(data_name)).drop(columns='date')

            R, C = D.shape

            # 1-2) binary data, data_m
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

            concat = []

            for i in range(int(R/L)):

                if i == int(R/L)-1:
                    data = D.to_numpy()[i*L:]
                    data_m = M[i*L:]
                    rows, cols = data.shape
                else:
                    data = D.to_numpy()[i * L:(i + 1) * L]
                    data_m = M[i * L:(i + 1) * L]
                    rows, cols = data.shape

                # 1-3) data_x and miss_data_x
                data_ori_x = data
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

                        if (epoch+1)%50 == 0:
                            test_z = np.random.uniform(0, 0.01, size=[rows, cols])
                            test_x = data_m * train_x + (1 - data_m) * test_z
                            test_x = np.concatenate((test_x, data_m), axis=1)
                            test_x_sq = test_x.reshape(-1, rows, cols * 2)

                            GAIN_imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1 - data_m)
                            GAIN_imputed = GAIN_imputed * (max_vector - min_vector) + min_vector

                            REAL = data_ori_x[data_m == 0].reshape(-1, len(col_dic[missing_col]))
                            GAIN = np.array(GAIN_imputed[data_m == 0]).reshape(-1, len(col_dic[missing_col]))

                            results_r2_GAIN = r2_score(REAL, GAIN)

                            print()
                            print('Data name: {}, Column: {}, Seed: {}'.format(data_name, missing_col, seed))
                            print('{}/{}th sample, time for {} epochs is {} sec'.format(i+1, int(R/L), epoch + 1, time.time() - start))
                            print('G_loss is {} and D_loss is {}'.format(gen_loss, disc_loss))
                            print('GAIN r-square value is', round(results_r2_GAIN, 4))
                            print()

                # 2-6) train model
                train_lstm_gain(epochs)


                # 3. Output (GAIN, KNN)
                # 3-1) GAIN imputed data
                test_z = np.random.uniform(0, 0.01, size = [rows, cols])

                test_x = data_m * train_x + (1-data_m) * test_z
                test_x = np.concatenate((test_x, data_m), axis=1)
                test_x_sq = test_x.reshape(-1, rows, cols*2)

                imputed = train_x * data_m + generator(test_x_sq, training=False)[0] * (1-data_m)
                imputed = np.array(imputed)
                imputed = imputed * (max_vector - min_vector) + min_vector

                concat.append(imputed)

            GAIN_imputed = np.array(concat[:-1]).reshape(-1, cols)
            GAIN_imputed = np.append(GAIN_imputed, concat[-1], axis=0)

            def knn():
                data_KNN = D.copy().to_numpy()
                data_KNN[M==0] = np.nan
                MIN = np.nanmin(data_KNN, axis=0)
                MAX = np.nanmax(data_KNN, axis=0)

                data_norm_KNN = (data_KNN- MIN)/(MAX - MIN)

                imputer = KNNImputer(n_neighbors = 3)
                KNN_imputed = imputer.fit_transform(data_norm_KNN)

                KNN_imputed = KNN_imputed * (MAX - MIN) + MIN

                return KNN_imputed

            KNN_imputed = knn()

            ## 3-4) r-square value
            def r_square():
                D = pd.read_csv('data/{}_raw.csv'.format(data_name)).drop(columns='date')
                REAL = D.to_numpy()[M == 0].reshape(int(R*missing_rate), -1)
                GAIN = GAIN_imputed[M == 0].reshape(int(R*missing_rate), -1)
                KNN = KNN_imputed[M == 0].reshape(int(R*missing_rate), -1)

                results_r2_GAIN = r2_score(REAL, GAIN)
                results_r2_KNN = r2_score(REAL, KNN)

                print()
                print('GAIN r-square value is', round(results_r2_GAIN, 4))
                print('KNN r-square value is', round(results_r2_KNN, 4))
                print()

            r_square()

            ## 3-5) save output
            def save_csv():
                GAIN = pd.DataFrame(GAIN_imputed, columns=D.columns.tolist())
                KNN = pd.DataFrame(KNN_imputed, columns=D.columns.tolist())

                GAIN.to_csv('result3/{}_result_{}_GAIN_{}_2.csv'.format(data_name, seed, missing_col), index=False)
                KNN.to_csv('result3/{}_result_{}_KNN_{}_2.csv'.format(data_name, seed, missing_col), index=False)

                print()
                print('{}_result_{}_GAIN|KNN_{}_1.csv. Saved GAIN and KNN results'.format(data_name, seed, missing_col))
                print()

            save_csv()