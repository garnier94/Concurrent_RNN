import torch
import datetime as dt


def filter_by_level(data, fam, niv='Niv. 3'):
    return data[data[niv] == fam]


def add_sum(data, fam, index, niv='Niv. 3'):
    df = filter_by_level(data, fam, niv)
    sum_df = df.groupby(level=1)[index].sum()
    return df.join(sum_df, rsuffix='_somme')


def adapt_data(data, index, col_drop=[], horizon=6):
    """Adaptation of the data, creation of the market share value"""
    X = data.copy()
    X.drop(columns=col_drop, inplace=True)
    X['pdM'] = 100 * (data[index] / data[index + '_somme'])
    X['pdM_shift'] = X['pdM'].shift(horizon)
    X['pdM_shift_1'] = X['pdM'].shift(horizon+1)
    return X.iloc[horizon+1:]


def dataframe_to_torch(serie, window, horizon, matrix, index=-1, min_sum=5, reference = False):
    """
    Generation of a list of tuple of torch.tensor [X, Y] for data from a serie using a slinding windows
    :param serie_prod: pd.Series con
    :param window: size of the sliding windows
    :param horizon: horizon of the prediction
    :param matrix: (if not False) matrix used to save the tuple X,y
    :param index_start: where to start in the matrix
    :param min_sum: Minimal sum of y in a batch
    """
    N = len(serie)
    for i in range(N - window  + 1):
        X = torch.tensor(serie.drop(columns='pdM').iloc[i:i+window, :].values, dtype=torch.float32).reshape(window, 1,-1)
        #X = torch.tensor(serie[['pdM_shift','pdM_shift_1']].iloc[i:i+window,:].values, dtype=torch.float32).reshape(window,1,-1)
        y = torch.tensor(serie['pdM'].iloc[i:i+window], dtype=torch.float32).reshape(window, 1,-1)
        y_ref = torch.tensor(serie['pdM_shift'].iloc[i:i+window], dtype=torch.float32).reshape(window,1,-1)
        if sum(y) >= min_sum:
            if index >= 0:
                if reference:
                    matrix[index + i].append([X, y,y_ref])
                else:
                    matrix[index + i].append([X, y])

            else:
                if reference:
                    matrix.append([X, y,y_ref])
                else:
                    matrix.append([X, y])


def test_train_generation(df, index, start_training_date, end_training_date, col_drop, concurrent=False, verbose=False,reference= False,
                          **kwargs):
    """
    Generation of a training and a testing set for our models
    :param df: DataFrame
    :param index: Name of the target colum
    :param start_training_date: start of the training period
    :param end_training_date: end of the training period
    :param col_drop: name of the useless columns in the data
    :param concurrent: Whether we generate test for a concurrent modelisation or not
    :param verbose: Should we display information on the process
    :return:
    """

    nb_ventes_mini = kwargs.get('nb_ventes_mini', 100)  # Minimal number of sales to keep a product
    minimal_length = kwargs.get('minimal_length', 20)  # Minimal sales length to keep product
    min_sum = kwargs.get('min_sum_share', 1)  # Minimal sum of share in a tupple (X,y)
    horizon = kwargs.get('horizon', 6)
    window = kwargs.get('window', 20)

    size_period = (dt.datetime.strptime(end_training_date, '%Y-%m-%d') - dt.datetime.strptime(start_training_date,
                                                                                              '%Y-%m-%d')).days // 7

    if concurrent:
        list_tuple_training = [[] for i in range(size_period)]
        list_tuple_testing = [[] for i in range(50)]
    else:
        list_tuple_training = []
        list_tuple_testing = []

    if verbose:
        keeped_products = []

    products = set(df.index.get_level_values(0))
    for prod in products:
        serie_prod = df.loc[prod].fillna(0)
        if sum(serie_prod[index]) > nb_ventes_mini:
            min_date = max(start_training_date, min(serie_prod[serie_prod[index] > 0].index))
            max_date = min(end_training_date, max(serie_prod[serie_prod[index] > 0].index))
            serie_prod = serie_prod[serie_prod.index >= min_date]

            min_date = dt.datetime.strptime(min_date, '%Y-%m-%d')
            max_date = dt.datetime.strptime(max_date, '%Y-%m-%d')

            if concurrent:
                index_start = (min_date - dt.datetime.strptime(start_training_date, '%Y-%m-%d')).days // 7
            else:
                index_start = -1

            if (max_date - min_date).days // 7 >= minimal_length:
                if verbose:
                    keeped_products.append(prod)
                train_df = serie_prod[serie_prod.index < dt.datetime.strftime(max_date, '%Y-%m-%d')]
                test_df = serie_prod[serie_prod.index > (
                    dt.datetime.strftime(max_date - dt.timedelta(days=7 * horizon), '%Y-%m-%d'))]
                dataframe_to_torch(adapt_data(train_df, index, col_drop=col_drop, horizon=horizon), window, horizon,
                                   list_tuple_training, index_start, min_sum=min_sum, reference=reference)
                if concurrent:
                    dataframe_to_torch(adapt_data(test_df, index, col_drop=col_drop, horizon=horizon), window, horizon,
                                       list_tuple_testing, 0, min_sum=min_sum, reference=reference)
                else:
                    dataframe_to_torch(adapt_data(test_df, index, col_drop=col_drop, horizon=horizon), window, horizon,
                                       list_tuple_testing, -1, min_sum=min_sum, reference=reference)
    if verbose:
        print('Nb de produits initial  %s ' % len(products))
        print('Nb de ventes initiales %s ' % sum(df[index]))
        print('Nb de produits conservés %s' % len(keeped_products))
        print('Nb de ventes conservées %s' % sum(df[index].loc[keeped_products]))

    return list_tuple_training, list_tuple_testing
