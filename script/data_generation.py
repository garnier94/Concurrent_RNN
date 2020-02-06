"""Script using luigi"""


import pandas as pd
import luigi
from concurrent_lstm import DATA_CONCURRENCE, CONFIG_CONTAINER
from concurrent_lstm.build_tensor import add_sum, test_train_generation
from concurrent_lstm.training import training_model
from sklearn.preprocessing import MinMaxScaler
import sys




class Family_Data_Generation(luigi.Task):
    family = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(DATA_CONCURRENCE / 'data_familly_' + self.family +'.csv')


    def run(self):
        index = 'Vente lisse'
        data = pd.read_csv( DATA_CONCURRENCE / "filtered_dt_for_nn" , sep = ';')
        data.set_index(['product','date'], inplace = True)
        df = add_sum(data, self.family, index)
        #Scaling
        cols_to_scale =['prix_moy','prix_min_marche_moy','marge_srp', 'pump']
        scaler = MinMaxScaler(feature_range=(0,10))
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        with self.output().open('wb') as f_out:
            df.to_csv(f_out, sep=';')


class Build_Simple_Tensor(luigi.Task):
    family = luigi.Parameter()

    def requires(self):
        return [Family_Data_Generation(family =self.family)]

    def output(self):
        return []

    def run(self):
        with self.input()[0].open() as f_in:
            data = pd.read_csv(f_in, sep=';', index_col=['product','date'])
        index = 'Vente lisse'
        col_drop = ['Vente lisse', 'min_marche', 'Vente réelle', 'Niv. 1', 'Niv. 2', 'Niv. 3']
        list_train, list_test = test_train_generation(data, index, '2017-01-01', '2019-01-01', col_drop, verbose=True)
        training_model(list_train, list_test, verbose=True)




if __name__ == '__main__':
    luigi.run()
