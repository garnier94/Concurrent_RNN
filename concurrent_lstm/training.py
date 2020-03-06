
from random import shuffle
import torch
import torch.nn as nn
from concurrent_lstm.LSTM_Models import LSTM_Model
from concurrent_lstm.systematisation import concurrent_evaluation_model


def init_model(**kwargs):

    learning_rate = float(kwargs.get('learning_rate', 4e-3))
    n_hidden = int(kwargs.get('n_hidden', 8))
    n_param = int(kwargs.get('n_param', 9))

    model = LSTM_Model(n_param, n_hidden, n_layer=int(kwargs.get('n_layer', 1)))
    loss_function = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer

def sum_model_non_conc(list_tuple, n_calc):
    sum_set = 0
    for _, y in list_tuple:
        sum_set += sum(y[-n_calc:, 0])
    return sum_set

def sum_model_conc(list_tuple, n_calc):
    sum_set = 0
    for mat_step in list_tuple:
        for _, y in mat_step:
            sum_set += sum(y[-n_calc:, 0])
    return sum_set


def predict(model, step):
    """Effectue la """
    pass

def non_concurrent_training(model, optimizer, loss_function, list_tuple_training, list_tuple_testing, verbose=False,
                            keep_trace=False,early_stopping = True,
                            **kwargs):
    err_train = []
    err_test = []
    n_calc = kwargs.get("n_calc", 10)
    epochs = kwargs.get("epoch", 200)
    copy_test_list = list_tuple_testing.copy()

    if early_stopping:
        N_test = len(copy_test_list) // 2
        set_valid = copy_test_list[-N_test:]
        copy_test_list = copy_test_list[:N_test]

    if verbose:
        sum_train = sum_model_non_conc(list_tuple_training, n_calc)
        sum_test = sum_model_non_conc(copy_test_list, n_calc)
        if early_stopping:
            sum_valid = sum_model_non_conc(set_valid, n_calc)

    old = 1000
    i = 0
    count_early_stopping = 0

    while (i < epochs and count_early_stopping < 5  ) :
        i+=1
        cumulated_train_loss = 0

        copy_train_list = list_tuple_training.copy()
        shuffle(copy_train_list)
        optimizer.zero_grad()

        loss = 0
        for X_train, y_train in copy_train_list:
            y_pred = model.predict_non_concurrent_step(X_train)
            single_loss = loss_function(y_pred[-n_calc:, 0], y_train[-n_calc:, 0])
            loss += single_loss
            cumulated_train_loss += single_loss.item()

        loss.backward()
        optimizer.step()

        cumulated_test_loss = 0

        if verbose:
            for X_test, y_test in copy_test_list:
                y_pred = model.predict_non_concurrent_step(X_test)
                single_loss = loss_function(y_pred[-n_calc:, 0], y_test[-n_calc:, 0])
                cumulated_test_loss += single_loss.item()
            if True:
                print('Epoch : %s/%s Training MAPE: %.4f  Testing MAPE: %.4f' % (
                    i, epochs, 100 * cumulated_train_loss / sum_train, 100 * cumulated_test_loss / sum_test))

        cumulated_valid_loss = 0
        if early_stopping:
            for X_test, y_test in set_valid:
                y_pred = model.predict_non_concurrent_step(X_test)
                single_loss = loss_function(y_pred[-n_calc:, 0], y_test[-n_calc:, 0])
                cumulated_valid_loss += single_loss.item()
            erreur_valid=cumulated_valid_loss/sum_valid
            if erreur_valid > old:
                count_early_stopping +=1
                old = erreur_valid
            else:
                count_early_stopping = 0
                old = erreur_valid

        if keep_trace:
            err_train.append(cumulated_train_loss)
            err_test.append(cumulated_test_loss)
    return i, err_train, err_test


def reference_prediction(list_tuple, **kwargs):
    sum_set = 0
    sum_loss = 0
    n_calc = kwargs.get("n_calc", 10)
    loss_function = nn.L1Loss(reduction='sum')

    for _, y , y_t in list_tuple:
        sum_set += sum(y[-n_calc:, 0])
        sum_loss += loss_function(y[-n_calc:, 0], y_t[-n_calc:, 0]).item()
    return 100 * sum_loss / sum_set




def concurrent_training_bis(model, optimizer, loss_function, list_tuple_training, list_tuple_testing, verbose=False,
                            keep_trace=True, early_stopping=True,
                            **kwargs):
    """
    Train a concurrent model with early stopping
    :param model: A torch.nn.Module model
    :param optimizer: torch.optim.Optimizer optimizer use in the model
    :param loss_function: torch.nn.L1Loss ou torch.nn.MSELoss
    :param list_tuple_training: training_set
    :param list_tuple_testing:
    :param verbose: whether to disp
    :param keep_trace:
    :param early_stopping: whether to use early stoppindg
    :return:
    """
    err_train_2 = []
    err_test_2 = []
    n_calc = kwargs.get("n_calc", 10)
    epochs = kwargs.get("epoch", 200)
    copy_test_list = list_tuple_testing.copy()
    if early_stopping:
        N_test = len(copy_test_list) // 2
        set_valid = copy_test_list[-N_test:]
        copy_test_list = copy_test_list[:N_test]
    if verbose:
        sum_train = sum_model_conc(list_tuple_training, n_calc)
        sum_test = sum_model_conc(copy_test_list, n_calc)
        if early_stopping:
            sum_valid = sum_model_conc(set_valid, n_calc)

    old =  1000
    i=0
    count_early_stopping = 0
    while (i < epochs and count_early_stopping < 10) or i < 40:
        i += 1
        cumulated_train_loss = 0
        copy_train_list = list_tuple_training.copy()
        shuffle(copy_train_list)
        optimizer.zero_grad()
        loss = 0
        for mat_step in copy_train_list:
            if len(mat_step) != 0:
                partial_sum = torch.tensor(mat_step[0][1])
                for j in range(1, len(mat_step)):
                    partial_sum += mat_step[j][1]
                model.reinitialize()
                cp_mat_step = mat_step.copy()
                sum_prediction = torch.ones(partial_sum.shape)
                for X_train, _ in cp_mat_step:
                    model.reinitialize()
                    sum_prediction += model(X_train)

                shuffle(cp_mat_step)
                for (X_train, y_train) in cp_mat_step:
                    model.reinitialize()
                    y_pred = partial_sum * model(X_train) / sum_prediction
                    single_loss = loss_function(y_pred[-n_calc:, 0], y_train[-n_calc:, 0])
                    loss += single_loss
                    cumulated_train_loss += single_loss.item()

        loss.backward()
        optimizer.step()
        erreur_test = concurrent_evaluation_model(model, loss_function, copy_test_list, current_sum=sum_test,
                                                  verbose=False, **kwargs)
        if early_stopping:
            erreur_valid = concurrent_evaluation_model(model, loss_function, set_valid, current_sum=sum_valid,
                                                  verbose=False, **kwargs)
            if erreur_valid > old:
                count_early_stopping +=1
                old = erreur_valid
            else:
                count_early_stopping = 0
                old = erreur_valid
        if verbose :
            print('Epoch : %s/%s Training MAPE: %.4f  Testing MAPE: %.4f' % (
                i, epochs, 100 * cumulated_train_loss / max(1, sum_train),  erreur_test))
        if keep_trace:
            err_train_2.append(cumulated_train_loss /sum_train)
            err_test_2.append(erreur_test)
    return i ,err_train_2, err_test_2


def training_model(list_tuple_training, list_tuple_testing, concurrent=False, verbose=False, **kwargs):
    model, loss_function, optimizer = init_model(**kwargs)
    if concurrent:
        final_epoch, err_train, err_test = concurrent_training_bis(model, optimizer, loss_function, list_tuple_training, list_tuple_testing, verbose=True,
                            keep_trace=verbose, **kwargs)
    else:
        final_epoch, err_train, err_test = non_concurrent_training(model, optimizer, loss_function, list_tuple_training, list_tuple_testing,
                                verbose=verbose, **kwargs)
    return model, loss_function, final_epoch, err_train, err_test




