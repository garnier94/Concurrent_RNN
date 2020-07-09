"""Script using luigi"""

import pandas as pd
import luigi
from concurrent_lstm import DATA_CONCURRENCE, CONFIG_CONTAINER, RESULTS
from concurrent_lstm.build_tensor import add_sum, test_train_generation
from concurrent_lstm.training import training_model, reference_prediction
from concurrent_lstm.systematisation import non_concurrent_evaluation_model, concurrent_evaluation_model, write_model
from sklearn.preprocessing import MinMaxScaler


class Family_Data_Generation(luigi.Task):
    family = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(DATA_CONCURRENCE / 'data_familly_' + self.family + '.csv')

    def run(self):
        index = 'Vente lisse'
        data = pd.read_csv(DATA_CONCURRENCE / "filtered_dt_for_nn.csv", sep=';')
        data.set_index(['product', 'date'], inplace=True)
        df = add_sum(data, self.family, index)
        df['diff_prix'] = df['prix_moy'].diff(6).fillna(0)
        # Scaling
        cols_to_scale = ['prix_moy', 'prix_min_marche_moy', 'marge_srp', 'pump', 'avg_review', 'diff_prix']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        with self.output().open('wb') as f_out:
            df.to_csv(f_out, sep=';')


class Build_Simple_Tensor(luigi.Task):
    family = luigi.Parameter()
    epoch = luigi.Parameter(default=200)
    concurrent = luigi.Parameter(default=True)

    def requires(self):
        return [Family_Data_Generation(family=self.family)]

    def output(self):
        return []

    def run(self):
        with self.input()[0].open() as f_in:
            data = pd.read_csv(f_in, sep=';', index_col=['product', 'date'])
        index = 'Vente lisse'
        col_drop = ['Vente lisse', 'min_marche', 'Vente réelle', 'Niv. 1', 'Niv. 2', 'Niv. 3']
        list_train, list_test = test_train_generation(data, index, '2017-01-01', '2019-03-01', col_drop, verbose=True,
                                                      concurrent=self.concurrent)
        if self.concurrent:
            n_param = list_test[0][0][0].shape[2]
        else:
            n_param = list_train[0][0].shape[2]
        model, loss_function, final_epoch, _, _ = training_model(list_train, list_test, verbose=True,
                                                                 early_stopping=False,
                                                                 concurrent=self.concurrent, n_hidden=4,
                                                                 n_param=n_param, epoch=int(self.epoch))
        print(self.family)
        if self.concurrent:
            print('Modele concurrent')
            err = concurrent_evaluation_model(model, loss_function, list_test, verbose=True)
            write_model(RESULTS / self.family + '_concurrent.txt', self.family, final_epoch, err)
        else:
            print("Modele non concurrent")
            err = non_concurrent_evaluation_model(model, loss_function, list_test, verbose=True)
            write_model(RESULTS / self.family + '_non_concurrent.txt', self.family, final_epoch, err)


class Build_Reference(luigi.Task):
    family = luigi.Parameter()

    def requires(self):
        return [Family_Data_Generation(family=self.family)]

    def run(self):
        with self.input()[0].open() as f_in:
            data = pd.read_csv(f_in, sep=';', index_col=['product', 'date'])
        index = 'Vente lisse'
        col_drop = ['Vente lisse', 'min_marche', 'Vente réelle', 'Niv. 1', 'Niv. 2', 'Niv. 3']
        list_train, list_test = test_train_generation(data, index, '2017-01-01', '2019-03-01', col_drop, verbose=True,
                                                      concurrent=False, reference=True)
        mape = reference_prediction(list_test)
        print('MAPE Ref : %.4f' % mape)


class Multi_comparaison(luigi.Task):
    epoch = luigi.Parameter(default=200)

    def requires(self):
        families = ['Autocuiseurs', 'Tondeuse électrique', "TV LED 4K entre 42 et 47''", 'Babyphone', 'Macbook',
                    'Canapé droit fixe', 'CAble optique']
        return [Build_Simple_Tensor(family=fam, epoch=self.epoch, concurrent=True) for fam in families]

if __name__ == '__main__':
    luigi.run()
