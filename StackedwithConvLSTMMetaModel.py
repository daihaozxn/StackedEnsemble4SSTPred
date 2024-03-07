
from numpy import argwhere, array, meshgrid, dstack, sum, mean
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, ConvLSTM2D, Conv2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
# from math import sqrt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time, strftime, localtime
from tensorflow.keras.models import load_model
import ipykernel, h5py
from os import makedirs, path
from pandas import DataFrame
from shutil import rmtree

# define the meta model ConvLSTM
def TaiwanStraitSSTPredictionwithConvLSTM(n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs):
    model = Sequential()
    if n_layers == 1:
        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       input_shape=(n_steps, n_rows, n_cols, 1)))  # 一层隐藏层

        model.add(Dropout(dropout_rate))
    else:
        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       input_shape=(n_steps, n_rows, n_cols, 1), return_sequences=True))  # 一层隐藏层
        model.add(Dropout(dropout_rate))
    if n_layers > 1:
        for i in range(n_layers - 2):
            model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu',
                                 padding='same',
                                 data_format='channels_last', kernel_initializer='glorot_uniform',
                                 return_sequences=True))
            model.add(Dropout(dropout_rate))

        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_outputs, kernel_initializer='glorot_uniform'))
    model.summary()
    model.compile(optimizer='Adam', loss='mse')
    return model

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(stackedX, inputy, epochs, batch_size, n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs, filename, valid_ratio):
    # create dataset using ensemble
    stackedX = stackedX.reshape(stackedX.shape[0], stackedX.shape[2], n_rows, n_cols, 1)
    # fit stacked model
    model = TaiwanStraitSSTPredictionwithConvLSTM(n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=40)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.5, patience=20)
    mc = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=1, save_best_only=True)

    model.fit(stackedX, inputy, epochs=epochs, batch_size=batch_size, validation_split=valid_ratio,
              verbose=1, callbacks=[mc, es, reduce_lr], shuffle=False)

    return model

# make a prediction with the stacked model
def stacked_prediction(stackedX, model, n_rows, n_cols):
    # create dataset using ensemble
    stackedX = stackedX.reshape(stackedX.shape[0], stackedX.shape[2], n_rows, n_cols, 1)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

def create_data(dataset, n_steps, lead_time):
    dataset_len = len(dataset)
    dataX, dataY = [], []
    for i in range(dataset_len - n_steps - lead_time + 1):
        tempX, tempY = dataset[i: (i + n_steps), :], dataset[i + n_steps - 1 + lead_time, :]
        dataX.append(tempX)
        dataY.append(tempY)
    return array(dataX), array(dataY)

def StackedEnsembleConvLSTM(is_training, study_area, lead_time, batch_size, filters, dropout_rate, n_layers, epochs, kernel_size):
    n_steps = 4
    n_outputs = 1

    if study_area == 'tw':
        f = h5py.File('data/raw_train.h5', 'r')
        raw_train = f.get('raw_train')[:]
        f = h5py.File('data/raw_test.h5', 'r')
        raw_test = f.get('raw_test')[:]
    elif study_area == 'dh':
        f = h5py.File('data/donghai_raw_train.h5', 'r')
        raw_train = f.get('donghai_raw_train')[:]
        f = h5py.File('data/donghai_raw_test.h5', 'r')
        raw_test = f.get('donghai_raw_test')[:]

    scaler = StandardScaler()
    raw_train_scaled = scaler.fit_transform(raw_train.reshape(-1, raw_train.shape[1] * raw_train.shape[2]))
    raw_test_scaled = scaler.transform(raw_test.reshape(-1, raw_test.shape[1] * raw_test.shape[2]))
    n_rows = raw_train.shape[1]
    n_cols = raw_train.shape[2]

    # convert SST spatio-temporal sequence data into a supervised learning problem
    X_train_valid, y_train_valid = create_data(raw_train_scaled, n_steps, lead_time)
    X_test, y_test = create_data(raw_test_scaled, n_steps, lead_time)

    X_train_valid = X_train_valid.reshape(X_train_valid.shape[0], n_steps, n_rows, n_cols, 1)
    y_train_valid = y_train_valid.reshape(y_train_valid.shape[0], n_rows, n_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], n_steps, n_rows, n_cols, 1)
    y_test = y_test.reshape(y_test.shape[0], n_rows, n_cols, 1)

    ratio = 0.5
    len_train_valid = X_train_valid.shape[0]
    X_train4stacked, y_train4stacked = X_train_valid[int(len_train_valid * ratio):, :], y_train_valid[int(len_train_valid * ratio):, :]
    valid_ratio = 0.2
    len_train = X_train4stacked.shape[0]
    X_valid, y_valid = X_train4stacked[int(len_train*valid_ratio):, :], y_train4stacked[int(len_train*valid_ratio):, :]

    if study_area == 'tw':
        if lead_time == 1:
            model_MLP = load_model('Submodels/lead_time_1/model_MLP_16.h5')
            model_LSTM = load_model('Submodels/lead_time_1/model_LSTM_68.h5')
            model_CNN = load_model('Submodels/lead_time_1/model_CNN_47.h5')
            model_CNNLSTM = load_model('Submodels/lead_time_1/model_CNNLSTM_94.h5')
        elif lead_time == 3:
            model_MLP = load_model('Submodels/lead_time_3/model_MLP_17.h5')
            model_LSTM = load_model('Submodels/lead_time_3/model_LSTM_31.h5')
            model_CNN = load_model('Submodels/lead_time_3/model_CNN_94.h5')
            model_CNNLSTM = load_model('Submodels/lead_time_3/model_CNNLSTM_56.h5')
        elif lead_time == 5:
            model_MLP = load_model('Submodels/lead_time_5/model_MLP_3.h5')
            model_LSTM = load_model('Submodels/lead_time_5/model_LSTM_46.h5')
            model_CNN = load_model('Submodels/lead_time_5/model_CNN_75.h5')
            model_CNNLSTM = load_model('Submodels/lead_time_5/model_CNNLSTM_1.h5')
        elif lead_time == 7:
            model_MLP = load_model('Submodels/lead_time_7/model_MLP_36.h5')
            model_LSTM = load_model('Submodels/lead_time_7/model_LSTM_5.h5')
            model_CNN = load_model('Submodels/lead_time_7/model_CNN_87.h5')
            model_CNNLSTM = load_model('Submodels/lead_time_7/model_CNNLSTM_12.h5')

    elif study_area == 'dh':
        if lead_time == 1:
            model_MLP = load_model('Submodels/DH_lead_time_1/model_MLP_88.h5')
            model_LSTM = load_model('Submodels/DH_lead_time_1/model_LSTM_52.h5')
            model_CNN = load_model('Submodels/DH_lead_time_1/model_CNN_69.h5')
            model_CNNLSTM = load_model('Submodels/DH_lead_time_1/model_CNNLSTM_20.h5')
        elif lead_time == 3:
            model_MLP = load_model('Submodels/DH_lead_time_3/model_MLP_22.h5')
            model_LSTM = load_model('Submodels/DH_lead_time_3/model_LSTM_47.h5')
            model_CNN = load_model('Submodels/DH_lead_time_3/model_CNN_35.h5')
            model_CNNLSTM = load_model('Submodels/DH_lead_time_3/model_CNNLSTM_5.h5')
        elif lead_time == 5:
            model_MLP = load_model('Submodels/DH_lead_time_5/model_MLP_36.h5')
            model_LSTM = load_model('Submodels/DH_lead_time_5/model_LSTM_78.h5')
            model_CNN = load_model('Submodels/DH_lead_time_5/model_CNN_27.h5')
            model_CNNLSTM = load_model('Submodels/DH_lead_time_5/model_CNNLSTM_76.h5')
        elif lead_time == 7:
            model_MLP = load_model('Submodels/DH_lead_time_7/model_MLP_32.h5')
            model_LSTM = load_model('Submodels/DH_lead_time_7/model_LSTM_2.h5')
            model_CNN = load_model('Submodels/DH_lead_time_7/model_CNN_78.h5')
            model_CNNLSTM = load_model('Submodels/DH_lead_time_7/model_CNNLSTM_59.h5')

    if is_training == 0:    # evaluate the individual models and the meta model on test dataset
        inv_y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[1] * y_test.shape[2] * y_test.shape[3]))

        X_test_MLP = X_test.reshape(-1, n_steps*n_rows*n_cols)
        X_test_LSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)
        X_test_CNN = X_test.reshape(-1, n_steps, n_rows*n_cols)
        X_test_CNNLSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)

        yhat_test_MLP = model_MLP.predict(X_test_MLP)
        inv_yhat_test_MLP = scaler.inverse_transform(yhat_test_MLP)
        rmse_test_MLP = mean_squared_error(inv_y_test, inv_yhat_test_MLP, squared=False)
        ce_test_MLP = 1 - (sum((inv_y_test - inv_yhat_test_MLP) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
        print('rmse_test_MLP: {:.3f}℃, ce_test_MLP: {:.3f}'.format(rmse_test_MLP, ce_test_MLP))

        # LSTM
        yhat_test_LSTM = model_LSTM.predict(X_test_LSTM)
        inv_yhat_test_LSTM = scaler.inverse_transform(yhat_test_LSTM)
        rmse_test_LSTM = mean_squared_error(inv_y_test, inv_yhat_test_LSTM, squared=False)
        ce_test_LSTM = 1 - (sum((inv_y_test - inv_yhat_test_LSTM) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
        print('rmse_test_LSTM: {:.3f}℃, ce_test_LSTM: {:.3f}'.format(rmse_test_LSTM, ce_test_LSTM))

        # CNN
        yhat_test_CNN = model_CNN.predict(X_test_CNN)
        inv_yhat_test_CNN = scaler.inverse_transform(yhat_test_CNN)
        rmse_test_CNN = mean_squared_error(inv_y_test, inv_yhat_test_CNN, squared=False)
        ce_test_CNN = 1 - (sum((inv_y_test - inv_yhat_test_CNN) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
        print('rmse_test_CNN: {:.3f}℃, ce_test_CNN: {:.3f}'.format(rmse_test_CNN, ce_test_CNN))

        # CNNLSTM
        yhat_test_CNNLSTM = model_CNNLSTM.predict(X_test_CNNLSTM)
        inv_yhat_test_CNNLSTM = scaler.inverse_transform(yhat_test_CNNLSTM)
        rmse_test_CNNLSTM = mean_squared_error(inv_y_test, inv_yhat_test_CNNLSTM, squared=False)
        ce_test_CNNLSTM = 1 - (sum((inv_y_test - inv_yhat_test_CNNLSTM) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
        print('rmse_test_CNNLSTM: {:.3f}℃, ce_test_CNNLSTM: {:.3f}'.format(rmse_test_CNNLSTM, ce_test_CNNLSTM))

        rmse_test_sum = rmse_test_MLP + rmse_test_LSTM + rmse_test_CNN + rmse_test_CNNLSTM
        inv_rmse_test_sum = rmse_test_sum/rmse_test_MLP + rmse_test_sum/rmse_test_LSTM + rmse_test_sum/rmse_test_CNN + rmse_test_sum/rmse_test_CNNLSTM
        weights_test_MLP = (rmse_test_sum/rmse_test_MLP)/inv_rmse_test_sum
        weights_test_LSTM = (rmse_test_sum/rmse_test_LSTM)/inv_rmse_test_sum
        weights_test_CNN = (rmse_test_sum/rmse_test_CNN)/inv_rmse_test_sum
        weights_test_CNNLSTM = (rmse_test_sum/rmse_test_CNNLSTM)/inv_rmse_test_sum
        yhat_test_ModelsWeightAve = weights_test_MLP*yhat_test_MLP + weights_test_LSTM*yhat_test_LSTM + weights_test_CNN*yhat_test_CNN + weights_test_CNNLSTM*yhat_test_CNNLSTM
        inv_yhat_test_ModelsWeightAve = scaler.inverse_transform(yhat_test_ModelsWeightAve)
        rmse_test_ModelsWeightAve = mean_squared_error(inv_y_test, inv_yhat_test_ModelsWeightAve, squared=False)
        ce_test_ModelsWeightAve = 1 - (sum((inv_y_test - inv_yhat_test_ModelsWeightAve) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
        print('rmse_test_ModelsWeightAve: {:.3f}℃, ce_test_ModelsWeightAve: {:.3f}'.format(rmse_test_ModelsWeightAve, ce_test_ModelsWeightAve))

        # create stacked model input dataset as outputs from the ensemble
        X_train4stacked_MLP = X_train4stacked.reshape(-1, n_steps*n_rows*n_cols)
        X_train4stacked_LSTM = X_train4stacked.reshape(-1, n_steps, n_rows*n_cols)
        X_train4stacked_CNN = X_train4stacked.reshape(-1, n_steps, n_rows*n_cols)
        X_train4stacked_CNNLSTM = X_train4stacked.reshape(-1, n_steps, n_rows*n_cols)
        # MLP
        yhat_train4stacked_MLP = model_MLP.predict(X_train4stacked_MLP)
        # LSTM
        yhat_train4stacked_LSTM = model_LSTM.predict(X_train4stacked_LSTM)
        # CNN
        yhat_train4stacked_CNN = model_CNN.predict(X_train4stacked_CNN)
        # CNNLSTM
        yhat_train4stacked_CNNLSTM = model_CNNLSTM.predict(X_train4stacked_CNNLSTM)

        stackX_train4stacked = dstack((yhat_train4stacked_MLP, yhat_train4stacked_LSTM, yhat_train4stacked_CNN, yhat_train4stacked_CNNLSTM))

        stackX_test = dstack((yhat_test_MLP, yhat_test_LSTM, yhat_test_CNN, yhat_test_CNNLSTM))

        folder = study_area + '_lt' + str(lead_time) + '_ConvLSTMStacked'
        if path.exists(folder):
            rmtree(folder)
        makedirs(folder)

        rmse_test_Stacked = []
        ce_test_Stacked = []
        n_members = 100
        for i in range(n_members):
            # fit stacked model using the ensemble
            filename = folder + '/model_ConvLSTMStacked_' + str(i + 1) + '.h5'
            model = fit_stacked_model(stackX_train4stacked, y_train4stacked, epochs, batch_size, n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs, filename, valid_ratio)

            # evaluate model on test set
            yhat_test_Stacked = stacked_prediction(stackX_test, model, n_rows, n_cols)

            inv_yhat_test_Stacked = scaler.inverse_transform(yhat_test_Stacked.reshape(-1, yhat_test_Stacked.shape[1] * yhat_test_Stacked.shape[2] * yhat_test_Stacked.shape[3]))

            rmse_test_temp = mean_squared_error(inv_y_test, inv_yhat_test_Stacked, squared=False)
            rmse_test_Stacked.append(rmse_test_temp)
            ce_test_temp = 1 - (sum((inv_y_test - inv_yhat_test_Stacked) ** 2)) / (sum((inv_y_test - mean(inv_y_test)) ** 2))
            ce_test_Stacked.append(ce_test_temp)

        rmse_test_MLP_new = []
        rmse_test_LSTM_new = []
        rmse_test_CNN_new = []
        rmse_test_CNNLSTM_new = []
        rmse_test_ModelsWeightAve_new = []

        ce_test_MLP_new = []
        ce_test_LSTM_new = []
        ce_test_CNN_new = []
        ce_test_CNNLSTM_new = []
        ce_test_ModelsWeightAve_new = []
        for i in range(n_members):
            rmse_test_MLP_new.append(rmse_test_MLP)
            rmse_test_LSTM_new.append(rmse_test_LSTM)
            rmse_test_CNN_new.append(rmse_test_CNN)
            rmse_test_CNNLSTM_new.append(rmse_test_CNNLSTM)
            rmse_test_ModelsWeightAve_new.append(rmse_test_ModelsWeightAve)

            ce_test_MLP_new.append(ce_test_MLP)
            ce_test_LSTM_new.append(ce_test_LSTM)
            ce_test_CNN_new.append(ce_test_CNN)
            ce_test_CNNLSTM_new.append(ce_test_CNNLSTM)
            ce_test_ModelsWeightAve_new.append(ce_test_ModelsWeightAve)

        metrics = {'MLP_RMSE': rmse_test_MLP_new,
                'MLP_CE': ce_test_MLP_new,
                'LSTM_RMSE': rmse_test_LSTM_new,
                'LSTM_CE': ce_test_LSTM_new,
                'CNN_RMSE': rmse_test_CNN_new,
                'CNN_CE': ce_test_CNN_new,
                'CNNLSTM_RMSE': rmse_test_CNNLSTM_new,
                'CNNLSTM_CE': ce_test_CNNLSTM_new,
                'ModelsWeightAve_RMSE': rmse_test_ModelsWeightAve_new,
                'ModelsWeightAve_CE': ce_test_ModelsWeightAve_new,
                'ConvLSTMStacked_RMSE': rmse_test_Stacked,
                'ConvLSTMStacked_CE': ce_test_Stacked,}
        dt_metrics = DataFrame(metrics)
        dt_metrics.to_csv(study_area+'_lt'+str(lead_time)+'_ConvLSTMStacked_test_metrics.csv', index=False, header=True)

    elif is_training == 1:   # evaluate metamodel performance on the validation set for hyper-parameters tuning
        inv_y_valid = scaler.inverse_transform(y_valid.reshape(-1, y_valid.shape[1] * y_valid.shape[2] * y_valid.shape[3]))

        X_valid_MLP = X_valid.reshape(-1, n_steps * n_rows * n_cols)
        X_valid_LSTM = X_valid.reshape(-1, n_steps, n_rows * n_cols)
        X_valid_CNN = X_valid.reshape(-1, n_steps, n_rows * n_cols)
        X_valid_CNNLSTM = X_valid.reshape(-1, n_steps, n_rows * n_cols)

        yhat_valid_MLP = model_MLP.predict(X_valid_MLP)
        inv_yhat_valid_MLP = scaler.inverse_transform(yhat_valid_MLP)
        rmse_valid_MLP = mean_squared_error(inv_y_valid, inv_yhat_valid_MLP, squared=False)
        ce_valid_MLP = 1 - (sum((inv_y_valid - inv_yhat_valid_MLP) ** 2)) / (sum((inv_y_valid - mean(inv_y_valid)) ** 2))
        print('rmse_valid_MLP: {:.3f}℃, ce_test_MLP: {:.3f}'.format(rmse_valid_MLP, ce_valid_MLP))

        # LSTM
        yhat_valid_LSTM = model_LSTM.predict(X_valid_LSTM)
        inv_yhat_valid_LSTM = scaler.inverse_transform(yhat_valid_LSTM)
        rmse_valid_LSTM = mean_squared_error(inv_y_valid, inv_yhat_valid_LSTM, squared=False)
        ce_valid_LSTM = 1 - (sum((inv_y_valid - inv_yhat_valid_LSTM) ** 2)) / (sum((inv_y_valid - mean(inv_y_valid)) ** 2))
        print('rmse_valid_LSTM: {:.3f}℃, ce_valid_LSTM: {:.3f}'.format(rmse_valid_LSTM, ce_valid_LSTM))

        # CNN
        yhat_valid_CNN = model_CNN.predict(X_valid_CNN)
        inv_yhat_valid_CNN = scaler.inverse_transform(yhat_valid_CNN)
        rmse_valid_CNN = mean_squared_error(inv_y_valid, inv_yhat_valid_CNN, squared=False)
        ce_valid_CNN = 1 - (sum((inv_y_valid - inv_yhat_valid_CNN) ** 2)) / (sum((inv_y_valid - mean(inv_y_valid)) ** 2))
        print('rmse_valid_CNN: {:.3f}℃, ce_valid_CNN: {:.3f}'.format(rmse_valid_CNN, ce_valid_CNN))

        # CNNLSTM
        yhat_valid_CNNLSTM = model_CNNLSTM.predict(X_valid_CNNLSTM)
        inv_yhat_valid_CNNLSTM = scaler.inverse_transform(yhat_valid_CNNLSTM)
        rmse_valid_CNNLSTM = mean_squared_error(inv_y_valid, inv_yhat_valid_CNNLSTM, squared=False)
        ce_valid_CNNLSTM = 1 - (sum((inv_y_valid - inv_yhat_valid_CNNLSTM) ** 2)) / (sum((inv_y_valid - mean(inv_y_valid)) ** 2))
        print('rmse_valid_CNNLSTM: {:.3f}℃, ce_valid_CNNLSTM: {:.3f}'.format(rmse_valid_CNNLSTM, ce_valid_CNNLSTM))

        # create stacked model input dataset as outputs from the ensemble  使用各子模型在验证集上的预测输出作为1级模型的训练数据集
        X_train4stacked_MLP = X_train4stacked.reshape(-1, n_steps * n_rows * n_cols)
        X_train4stacked_LSTM = X_train4stacked.reshape(-1, n_steps, n_rows * n_cols)
        X_train4stacked_CNN = X_train4stacked.reshape(-1, n_steps, n_rows * n_cols)
        X_train4stacked_CNNLSTM = X_train4stacked.reshape(-1, n_steps, n_rows * n_cols)
        # MLP
        yhat_train4stacked_MLP = model_MLP.predict(X_train4stacked_MLP)
        # LSTM
        yhat_train4stacked_LSTM = model_LSTM.predict(X_train4stacked_LSTM)
        # CNN
        yhat_train4stacked_CNN = model_CNN.predict(X_train4stacked_CNN)
        # CNNLSTM
        yhat_train4stacked_CNNLSTM = model_CNNLSTM.predict(X_train4stacked_CNNLSTM)

        stackX_train4stacked = dstack((yhat_train4stacked_MLP, yhat_train4stacked_LSTM, yhat_train4stacked_CNN, yhat_train4stacked_CNNLSTM))

        stackX_valid = dstack((yhat_valid_MLP, yhat_valid_LSTM, yhat_valid_CNN, yhat_valid_CNNLSTM))

        folder = study_area + '_lt' + str(lead_time) + '_hyperparams_ConvLSTMStacked'
        if path.exists(folder):
            rmtree(folder)
        makedirs(folder)

        rmse_valid_Stacked = []
        ce_valid_Stacked = []
        n_members = 20
        for i in range(n_members):
            # fit stacked model using the ensemble
            filename = folder + '/model_ConvLSTMStacked_' + str(i + 1) + '.h5'
            model = fit_stacked_model(stackX_train4stacked, y_train4stacked, epochs, batch_size, n_layers, n_steps,
                                      n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs, filename, valid_ratio)

            # evaluate model on validation set
            yhat_valid_Stacked = stacked_prediction(stackX_valid, model, n_rows, n_cols)

            inv_yhat_valid_Stacked = scaler.inverse_transform(yhat_valid_Stacked.reshape(-1, yhat_valid_Stacked.shape[1] *
                                                                                       yhat_valid_Stacked.shape[2] *
                                                                                       yhat_valid_Stacked.shape[3]))

            rmse_valid_temp = mean_squared_error(inv_y_valid, inv_yhat_valid_Stacked, squared=False)
            rmse_valid_Stacked.append(rmse_valid_temp)
            ce_valid_temp = 1 - (sum((inv_y_valid - inv_yhat_valid_Stacked) ** 2)) / (sum((inv_y_valid - mean(inv_y_valid)) ** 2))
            ce_valid_Stacked.append(ce_valid_temp)

        rmse_valid_MLP_new = []
        rmse_valid_LSTM_new = []
        rmse_valid_CNN_new = []
        rmse_valid_CNNLSTM_new = []
        for i in range(n_members):
            rmse_valid_MLP_new.append(rmse_valid_MLP)
            rmse_valid_LSTM_new.append(rmse_valid_LSTM)
            rmse_valid_CNN_new.append(rmse_valid_CNN)
            rmse_valid_CNNLSTM_new.append(rmse_valid_CNNLSTM)

        rmse = {'MLP': rmse_valid_MLP_new,
                'LSTM': rmse_valid_LSTM_new,
                'CNN': rmse_valid_CNN_new,
                'CNNLSTM': rmse_valid_CNNLSTM_new,
                'ConvLSTMStacked': rmse_valid_Stacked}
        dt_rmse = DataFrame(rmse)
        dt_rmse.to_csv(study_area + '_lt' + str(lead_time) + '_ConvLSTMStacked_valid_rmse.csv', index=False, header=True)

if __name__ == '__main__':
    # The parameter "is_training" is used to control whether hyper-parameter tuning (=1) or testing (=0) is conducted.
    # The parameters "study_area" and "lead_time" are used to select which ocean ('tw' for Taiwan Strait and 'dh' for East China Sea) and lead time to load for the 0-level model.
    StackedEnsembleConvLSTM(is_training=0, study_area='dh', lead_time=3, batch_size=64, filters=128, dropout_rate=0.0,
                            n_layers=1, epochs=400, kernel_size=(3, 3))