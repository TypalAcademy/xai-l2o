import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def print_model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    print(table)


def plot_dict_signal(net, loader, title="Signal Plot with Test Data",
                     inference_depth=1000):
    """ Plot prediction and its sparsification alongside true signal

        Using a batch of test data, an inference is computed with net. This is
        plotted alongside the sparsification (i.e. application of matrix K).
    """
    with torch.no_grad():
        net.to(device='cpu')
        u_true, d_batch = next(iter(loader))

        n       = u_true.shape[1]
        u_true  = u_true.t()
        u_pred  = net(d_batch.to('cpu'), max_depth=inference_depth).t()
        t       = np.linspace(1, n, n, endpoint=True)

        fig, axes = plt.subplots(2, 2, sharex="col", figsize=(7, 5))

        axes[0, 0].plot(t, u_true[:, 0].numpy())
        axes[1, 0].plot(t, u_pred[:, 0].numpy())
        axes[0, 1].plot(t, (net.K @ u_true)[:, 0].numpy())
        axes[1, 1].plot(t, (net.K @ u_pred)[:, 0].numpy())

        axes[0, 0].set_title("$u^\star$")
        axes[0, 1].set_title("$Ku^\star$")
        axes[1, 0].set_title("$N_\Theta(d)$")
        axes[1, 1].set_title("$K\ N_\Theta(d)$")
        fig.suptitle(title, fontsize=16)

        plt.show()
        net.to(device='cuda:0')


def plot_cmf(net, inference_model, loader_train):
    """ Plot CMFs
    """
    u_ref, d_batch = next(iter(loader_train))
    d_batch = d_batch.to(net.device())

    u_ref = inference_model(d_batch)
    u_ref = u_ref.to(net.device())

    res = net.get_property_value(u_ref, d_batch).detach()
    res = torch.sort(res)[0]
    cmf = net.get_cumulative_prob(res).detach().to(net.device())
    cmf_obs = torch.linspace(1/cmf.shape[0], 1, steps=cmf.shape[0])

    fig, axes = plt.subplots(1, 1, sharex="col")
    axes.plot(res.cpu().numpy(), cmf.cpu().numpy(), label='Predicted')
    axes.plot(res.cpu().numpy(), cmf_obs.cpu().numpy(), label='Observed')
    axes.legend()
    plt.show()


def EMD(cmf):
    device = cmf.device
    ones = torch.linspace(1/cmf.shape[0], 1, steps=cmf.shape[0]).to(device)
    weighted_emd_term = torch.linspace(1/cmf.shape[0], 1, steps=cmf.shape[0])
    weighted_emd_term = weighted_emd_term.to(device)
    cmf_diff = torch.multiply(weighted_emd_term, cmf-ones)
    emd_tot = torch.norm(cmf_diff, p=1) / cmf.shape[0]
    return emd_tot


def solve_least_squares(M, b, tol_fidelity=1.0e-6, max_iters=1.0e5):
    """ Find x such that 0.5 * || M * x - b||^2 is minimized.

        Gradient descent is used to iteratively update estimates.

        Note:
            Input b size  = [num_batches, m]
            Input M size  = [m, n]
            Output x size = [n, num_batches]
    """
    n           = M.shape[1]
    num_batches = b.shape[0]

    b    = b.clone().t()
    x    = torch.zeros(n, num_batches)
    conv = False
    lip  = torch.linalg.matrix_norm(M.t().mm(M), ord=2)
    iter = 0

    while not conv:
        grad          = M.t().mm(M.mm(x) - b)
        x             = x - (1.0 / lip) * grad 
        norm_fidelity = torch.max(torch.norm(M.t().mm(torch.mm(M, x) - b), dim=0)) 
        norm_data     = torch.mean(torch.norm(M.t().mm(b), dim=0))  
        conv          = norm_fidelity <= tol_fidelity * norm_data
        iter         += 1
        assert iter < max_iters, "Least squares method did not converge."

    return x.permute(1, 0)


def create_property_loaders(model,
                            train_loader_ref, test_loader_ref,
                            get_property_value,
                            device='cuda:0',
                            train_batch_size=1000,
                            test_batch_size=100):

    alpha = torch.zeros(1)

    with torch.no_grad():
        for _, d in train_loader_ref:
            d = d.to(device)
            u_inf = model(d)
            u_inf = u_inf.to(device)
            alpha_new = get_property_value(u_inf, d).cpu()
            u_inf = u_inf.cpu()
            d = d.cpu()
            if alpha.shape[0] == 1:
                alpha = alpha_new
            else:
                alpha = torch.cat((alpha, alpha_new), dim=0)

    dataset_train = TensorDataset(alpha)

    alpha = torch.zeros(1)

    with torch.no_grad():
        for _, d in test_loader_ref:
            d = d.to(device)
            u_inf = model(d)
            u_inf = u_inf.to(device)
            alpha_new = get_property_value(u_inf, d).cpu()
            u_inf = u_inf.cpu()
            d = d.cpu()
            if alpha.shape[0] == 1:
                alpha_test = alpha_new
            else:
                alpha_test = torch.cat((alpha_test, alpha_new), dim=0)

    dataset_test = TensorDataset(alpha_test)

    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def create_dict_loaders(train_batch_size=100,
                        test_batch_size=1000, m=100, n=250, r=50,
                        s_p=0.1, train_size=10000, test_size=1000):
    ''' Create data for Analysis Dictionary problem

        Signals s are generated as the composition of Bernoulli and Gaussian
        in r-dim space. This is mapped to a signal x in n-dim space by applying
        a Gaussian matrix M. Then measurements are generated using d = Ax
    '''
    torch.manual_seed(2021)
    tol = 1.0e-10

    data_size = train_size + test_size
    bernoulli_terms = torch.bernoulli(s_p * torch.ones((r, data_size)))
    gaussian_terms = torch.randn(r, data_size)
    s = bernoulli_terms * gaussian_terms

    A = torch.randn(m, n)
    A_col_norms = (tol + torch.norm(A, dim=0)) ** -1.0
    A = A_col_norms * A

    M = torch.randn(n, r)
    x = torch.mm(M, s)
    d = torch.mm(A, x)

    dataset = TensorDataset(x.permute(1, 0), d.permute(1, 0))

    train_data, test_data = random_split(dataset, [train_size,
                                                   test_size])
    loader_train = DataLoader(dataset=train_data,
                              batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(dataset=test_data,
                             batch_size=test_batch_size, shuffle=True)

    return loader_train, loader_test, A


def train_certificate(model, model_signal, optimizer, lr_scheduler, loader,
                      plot_signal, plot_cmf,
                      max_epochs=1000, device='cuda:0', save_dir='./',
                      plot_freq=100):
    loss_property_ave = 0.0
    certs_ave = 0.0
    best_loss = 1.0e10
    fmt = '[{:2d}/{:2d}]: train_loss = {:7.3e} | time = {:4.1f} sec'

    model = model.to(device)
    for epoch in range(max_epochs):

        start_time_epoch = time.time()
        for alpha in loader:

            alpha = alpha[0].to(device)

            model.train()
            optimizer.zero_grad()

            certs = model.get_cum_prob(alpha)
            certs = torch.sort(certs)[0].to(device)
            if epoch == 0:
                certs_ave = certs
            else:
                mov_ave = 0.9995 * certs_ave.detach() + 0.0005 * certs
                certs_ave = mov_ave
            certs_ave = certs_ave.to(device)

            loss = EMD(certs_ave)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = loss.detach().item()

            loss_property_ave *= 0.99
            loss_property_ave += 0.01 * loss_value

        end_time_epoch = time.time()

        time_epoch = end_time_epoch - start_time_epoch
        print(fmt.format(epoch+1, max_epochs, loss_property_ave, time_epoch))

        if epoch > 100 and loss_property_ave < best_loss:
            best_loss = loss_property_ave
            state = {
                'certificate_model_state': model.state_dict(),
            }
            torch.save(state, save_dir)
            print('\nModel weights saved to ' + save_dir)

        lr_scheduler.step()

        if epoch % plot_freq == 0:
            plot_signal(model_signal)
            plot_cmf(model, model_signal)
            stepsize = optimizer.param_groups[0]['lr']
            training_mess = '[{:5.0f}] property loss = {:2.3e} | lr = {:2.3e}'
            print(training_mess.format(epoch, loss_property_ave, stepsize))
    return model
