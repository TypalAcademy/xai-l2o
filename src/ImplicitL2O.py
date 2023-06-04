from abc import ABC, abstractmethod
import torch
import torch.nn as nn

inference = torch.tensor
input_data = torch.tensor


class ImplicitL2OModel(ABC, nn.Module):
    def device(self):
        return next(self.parameters()).data.device

    def assign_cert_model(self, cert_model):
        ''' Assign certificate model for identifying inference trustworthiness
        '''
        self._cert_model = cert_model

    def get_certs(self, x: inference, d: input_data):
        ''' Get cumulative probabilities for each trustworthiness certificate

            Returned size is x.shape[0] by c, where c is the number of
            certificates.
        '''
        valid_cert_model = self._cert_model is not None
        assert valid_cert_model, 'Certificate model must be assigned first'

        return self._cert_model.get_certs(x, d)

    @abstractmethod
    def _apply_T(self):
        ''' Apply optimization algorithm update
        '''
        pass

    @abstractmethod
    def _get_conv_crit(self):
        ''' Identify stopping criteria for forward propagation
        '''
        pass

    @abstractmethod
    def forward(self):
        pass
