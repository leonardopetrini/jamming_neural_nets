from alice import *


def train_model(model, data, learning_rate=.005, epochs_number=int(1e6), gen_error_flag=False):
    """Train model with data and plot running loss and N_Delta.

    If gen_error_flag == True splits data in 80% training and 20% test sets;
    Optimizer: Adam;
    Learning rate is diveded by 3 at 1/3 and 2/3 of epochs_number.

    Early stop is N_delta is small (i.e. 0.2% * P).

    Returns values of loss, N_delta, gen_error and the trained model at the last epoch."""
    optimizer, scheduler = optimizer_init(model.model, learning_rate, optimizer_name='Adam', scheduler_name='StepLR',
                                          epochs_number=epochs_number)
    running_loss = []
    running_N_delta = []
    gen_error = -99

    for e in range(epochs_number):
        for X, labels in data:
            if gen_error_flag:
                train_size = int(.8 * X.shape[0])
                test_ = X.narrow(0, train_size, X.shape[0] - train_size)
                test_labels = labels.narrow(0, train_size, X.shape[0] - train_size)
                labels = labels.narrow(0, 0, int(.8 * X.shape[0]))
                X = X.narrow(0, 0, int(.8 * X.shape[0]))

            output = model(X)
            loss, N_delta = hinge_loss(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            running_N_delta.append(N_delta.item())

        # Exit if N_Delta == 0
        if N_delta.item() < 0.002 * X.shape[0]:
            print('Early stop: SAT @ step {}'.format(e))
            break

        scheduler.step()

    print("Loss = ", loss.item(), ' N_delta = ', N_delta.item())

    if gen_error_flag:
        with torch.no_grad():
            gen_error = ((model(test_) * test_labels) < 0).float().mean()
            gen_error = gen_error.item()
            print('Generalization error = {:0.3f} %'.format(gen_error * 100))

    plot_loss_N_delta(running_loss, running_N_delta)

    return loss.item(), N_delta.item(), gen_error, model


def optimizer_init(model, learning_rate, optimizer_name='Adam', scheduler_name='StepLR', epochs_number=int(1e6)):
    """Define optimizer and learning rate scheduling for the training.
    Can choose between SGD or Adam optimizer and step or plateau detection scheduling."""

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Specify an optimizer, either Adam or SGD')

    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_number // 3, gamma=0.3)
    elif scheduler_name == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=1000, cooldown=10000,
                                                               verbose=False)
    else:
        raise ValueError('Specify a scheduler, either StepLR or Plateau')

    return optimizer, scheduler
