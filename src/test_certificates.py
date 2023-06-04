import torch
from certificate import CertificateEnsemble
probability = float

torch.manual_seed(42)


def get_nonzeros_cum_prob(x, d=None) -> probability:
    ''' Return cumulative probability, assuming nonzeros <= 10%
    '''
    percent_nonzero = 0.1
    nonzero_proportion = torch.count_nonzero(x, dim=1) / x.shape[1]
    cum_prob = nonzero_proportion / percent_nonzero
    cum_prob[cum_prob > 1.0] = 1.0
    return cum_prob


def get_fidelity_cum_prob(x, d) -> probability:
    ''' Return cumulative probability for hard ball constraint
    '''
    norm_tol = 1.0e-2
    norm_fidelity = torch.norm(x - d, dim=1)
    norm_data = torch.norm(d, dim=1)

    cum_prob = torch.zeros(norm_fidelity.shape)
    cum_prob[norm_fidelity > norm_tol * norm_data] = 1.0
    return cum_prob


def test_Ensemble():
    ''' Create and test ensemble with 2 dummy cumulative probability functions
    '''
    names_prop = ['sparsity', 'fidelity']
    models_certs = [get_nonzeros_cum_prob, get_fidelity_cum_prob]
    certs_ensemble = CertificateEnsemble(names_prop, models_certs)

    x_size = 100
    num_samples = 1
    prob_nonzero = 0.05
    shape = (num_samples, x_size)

    bernoulli = torch.bernoulli(prob_nonzero * torch.ones(shape))
    gaussian = torch.randn(shape)
    x = bernoulli * gaussian * torch.rand(shape)
    d = x + 1.0e-4 * torch.randn(shape)

    certs = certs_ensemble.get_certs(x, d)
    thresh_prob_fail = 0.99
    for name in names_prop:
        assert certs[name] < thresh_prob_fail, 'Certificate has fail label'
