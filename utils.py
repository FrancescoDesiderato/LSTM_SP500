import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import torch.nn as nn
import torch

class ModLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,seq_lenght=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_lenght

        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=seq_lenght)
        self.fc = nn.Linear(hidden_size, 2) #Perchè voglio prevedere 2 classi
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out,_ = self.lstm(x)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(17, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 12),
            nn.ReLU(),
            nn.Linear(12, 17),
            #nn.Sigmoid()# Output values in range [0, 1]
        )

    def encode(self, x):
        # Flatten the input
        #x = x.view(x.size(0), -1)
        # Encode
        encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        # Decode
        decoded = self.decoder(encoded)
        # Reshape to original image dimensions
        #decoded = decoded.view(decoded.size(0), 1, 28, 28)
        return decoded

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

def fft_denoiser(data, threshold, to_real=True):
    """
    Parametri:
        - data: array-like, dati di input (ad esempio, serie temporale)
        - threshold: float, soglia per filtrare i coefficienti FFT (tra 0 e 1)
        - to_real: boolean, se True restituisce solo la parte reale del risultato
    """
    # Applicare la FFT
    fft_coefficients = np.fft.fft(data)
    # Calcolare l'ampiezza dei coefficienti
    magnitude = np.abs(fft_coefficients)
    # Determinare la soglia in base all'ampiezza massima
    cutoff = threshold * np.max(magnitude)
    # Filtrare i coefficienti inferiori alla soglia
    fft_coefficients[magnitude < cutoff] = 0
    # Trasformata inversa per riportare i dati al dominio del tempo
    denoised_data = np.fft.ifft(fft_coefficients)

    # Restituire solo la parte reale se richiesto
    return denoised_data.real if to_real else denoised_data

def split_function(data,selected_features,target_fature,type='ts'):
    if type == 'ts':
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(data):
            X_train = data.iloc[train_index][selected_features].values
            y_train = data.iloc[train_index][target_fature].values
            X_test = data.iloc[test_index][selected_features].values
            y_test = data.iloc[test_index][target_fature].values

    if type == 'mul-ts':
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(data):
            X_train = data.iloc[train_index][selected_features]
            y_train = data.iloc[train_index][target_fature].values #Non mi interessa trattenere la data
            X_test = data.iloc[test_index][selected_features]
            y_test = data.iloc[test_index][target_fature].values #Non mi interessa trattenere la data

    if type == 'classic':
        X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data[target_fature], test_size=0.2)

    return X_train, y_train, X_test, y_test


def multiple_gmm(data,all_features,features,target_fature):
    X_train, y_train, X_test, y_test = split_function(data,all_features,target_fature,type='mul-ts')
    predictions = np.zeros(shape=[data.shape[0],9])
    col = 0
    for arr in features:
        x_train = X_train[arr].values
        x_test = X_test[arr].values

        standard = StandardScaler()
        x_train = standard.fit_transform(x_train)
        x_test = standard.transform(x_test)

        #Allenamento del modello gmm
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(x_train)
        print(model.predict_proba(data[arr].values).shape)
        predictions[:,col:col+2] = np.array(model.predict_proba(data[arr].values))
        col += 2

    predictions[:,8] = np.array(data[target_fature].values).reshape(-1)
    df = pd.DataFrame(predictions,columns = ['market_state_1','market_state_0','nfp_state_1','nfp_state_0',
                                             'macro_state_1','macro_state_0','interest_state_1','interest_state_0',
                                             'real_pred']) #

    return df

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc



