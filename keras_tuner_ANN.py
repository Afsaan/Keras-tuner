
# pip install keras-tuner - this lib will only work with tf2
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

df=pd.read_csv('Real_Combine.csv')

print(df.head())

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

########### hyperparameter#############
# 1. how many number of hidden layers we should have
# 2. How many number of neurons we should have in hidden layers
# 3. Learning Rate

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

tuner = RandomSearch(
build_model,
objective='val_mean_absolute_error',
max_trials=20,
executions_per_trial=3,
directory='project',
project_name='Air Quality Index')

print(tuner.search_space_summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))

print(tuner.results_summary())