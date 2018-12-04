from alice import *

def train_model(model, data, learning_rate=.005, epochs_number=int(1e6), gen_error_flag=False):
    """Train model and plot loss and N_Delta"""
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
            print('Generalization error = {:0.1f} %'.format(gen_error * 100))

    plot_loss_N_delta(running_loss, running_N_delta)

    return loss.item(), N_delta.item(), model, gen_error


def optimizer_init(model, learning_rate, optimizer_name='Adam', scheduler_name='StepLR', epochs_number=int(1e6)):
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
