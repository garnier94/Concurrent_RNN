import torch

def non_concurrent_evaluation_model(model, loss_function,list_tuple, current_sum = None, verbose = False, **kwargs ):
    n_calc = kwargs.get("n_calc", 10)

    if current_sum is None:
        current_sum = 0
        for _, y in list_tuple:
            current_sum += sum(y[-n_calc:, 0])

    cumulated_loss = 0

    for X,y in list_tuple:
        y_pred = model.predict_non_concurrent_step(X)
        single_loss = loss_function(y_pred[-n_calc:, 0], y[-n_calc:, 0])
        cumulated_loss += single_loss.item()

    if verbose:
        print('MAPE : %.4f' % (100 * cumulated_loss/current_sum))
    return 100 * cumulated_loss/current_sum


def concurrent_evaluation_model(model, loss_function, list_tuple, current_sum=None, verbose=False, **kwargs):
    n_calc = kwargs.get("n_calc", 10)

    if current_sum is None:
        current_sum = 0
        for mat_step in list_tuple:
            for _, y in mat_step:
                current_sum += sum(y[-n_calc:, 0])

    cumulated_loss = 0

    for mat_step in list_tuple:
        if len(mat_step) != 0:
            partial_sum = torch.tensor(mat_step[0][1])
            for j in range(1, len(mat_step)):
                partial_sum += mat_step[j][1]

            sum_prediction = torch.ones(partial_sum.shape)

            for X_train, _ in mat_step:
                model.reinitialize()
                sum_prediction += model(X_train)

            for X,y in mat_step:
                y_pred = partial_sum * model(X) / sum_prediction
                single_loss = loss_function(y_pred[-n_calc:, 0], y[-n_calc:, 0])
                cumulated_loss += single_loss.item()

    if verbose:
        print('MAPE : %.4f' % (100 * cumulated_loss / current_sum))
    return 100 * cumulated_loss / current_sum


def write_model(file, famille, epoch,err):
    f = open(file,'a')
    f.write('Famille: %s\n' % famille)
    f.write("Epoch: %s\n" % epoch)
    f.write('MAPE: %.4f\n' % err)
    f.close()