# Use this script to train model without luigi

import sys
import pandas as pd

from concurrent_lstm import DATA_CONCURRENCE, CONFIG_CONTAINER
from concurrent_lstm.build_tensor import add_sum, test_train_generation
from concurrent_lstm.training import training_model
from sklearn.preprocessing import MinMaxScaler


family = sys.argv[1]
data = pd.read_csv(DATA_CONCURRENCE / 'data_familly_' + family +'.csv', sep=";", index_col=['product', 'date'])
index = 'Vente lisse'
col_drop = [ 'Vente lisse', 'min_marche', 'Vente réelle', 'Niv. 1', 'Niv. 2', 'Niv. 3']
list_train, list_test = test_train_generation(data, index, '2017-01-01', '2019-01-01', col_drop, verbose=True)
training_model(list_train, list_test, verbose=True)
