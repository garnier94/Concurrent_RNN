from random import shuffle
import torch
import torch.nn as nn
from concurrent_lstm.LSTM_Models import LSTM_Model


def init_model(**kwargs):
    learning_rate = kwargs.get('learning_rate', 5e-3)
    n_hidden = kwargs.get('n_hidden', 20)
    n_param = kwargs.get('n_param', 9)  # TODO: Corriger ça avec le vrai n_param

    model = LSTM_Model(n_param, n_hidden, n_layer=kwargs.get('n_layer', 1))
    loss_function = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer


def predict(model, step):
    """Effectue la """


def non_concurrent_training(model, optimizer, loss_function, list_tuple_training, list_tuple_testing, verbose=False,
                            keep_trace=False,
                            **kwargs):
    err_train = []
    err_test = []
    n_calc = kwargs.get("n_calc", 20)
    epochs = kwargs.get("epoch", 100)

    if verbose:
        sum_train = 0
        for _, y_train in list_tuple_training:
            sum_train += sum(y_train[-n_calc:, 0])
        sum_test = 0
        for _, y_test in list_tuple_testing:
            sum_test += sum(y_test[-n_calc:, 0])

    for i in range(epochs):
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
            for X_test, y_test in list_tuple_testing:
                y_pred = model.predict_non_concurrent_step(X_test)
                single_loss = loss_function(y_pred[-n_calc:, 0], y_test[-n_calc:, 0])
                cumulated_test_loss += single_loss.item()

            if i % 5 == 0:
                print('Epoch : %s/%s Training MAPE: %.4f  Testing MAPE: %.4f' % (
                i, epochs, 100 * cumulated_train_loss / sum_train, 100 * cumulated_test_loss / sum_test))
        if keep_trace:
            err_train.append(cumulated_train_loss)
            err_test.append(cumulated_test_loss)


def training_model(list_tuple_training, list_tuple_testing, concurrent=False, verbose=False, **kwargs):
    model, loss_function, optimizer = init_model(**kwargs)
    if concurrent:
        print('Non implémenté actuellement')
        # TODO : Concurrent implémentation
    else:
        non_concurrent_training(model, optimizer, loss_function, list_tuple_training, list_tuple_testing,
                                verbose=verbose,
                                **kwargs)
