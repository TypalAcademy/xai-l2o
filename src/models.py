import torch
import torch.nn as nn
import numpy as np
from ImplicitL2O import ImplicitL2OModel

inference = torch.tensor
input_data = torch.tensor
dual = torch.tensor


class ImpDictModel(ImplicitL2OModel):
    """ Model to recover signal from measurements by leveraging sparse structure

        Inferences are defined by

        $$
            \mathsf{model(d) = argmin_x\ \| Kx \|_1  \quad s.t. \quad Ax = d,}
        $$     

        where $\mathsf{K}$ is a tunable matrix. Because the model is
        equivalent under scaling of $\mathsf{K}$, we fix
        $\mathsf{|| K ||_2 = 1}$. This is enforced during training by
        dividing by the matrix norm at the beginning of forward propagation.
        The forward iteration is Linearized ADMM (L-ADMM).
    """
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.A_norm = torch.linalg.matrix_norm(A, ord=2)
        self.lip = 1.0 + self.A_norm
        self.alpha = 0.5
        self.beta = 1.0 / self.lip
        self.lambd = 1.0

        x_size = A.size()[1]
        self.K = nn.Parameter(torch.randn(x_size, x_size))
        self.shrink = nn.Softshrink(lambd=self.lambd)

    def _apply_T(self, x: inference, p: torch.tensor, v1: dual, v2: dual,
                 d: input_data, return_tuple=False) -> inference:
        r""" Apply model operator using L-ADMM update

            Core functionality is single iteration update for Linearized ADMM,
            which is rearranged to make the signal $\mathsf{x}$ update last.
            This is needed to ensure the JFB backprop attaches gradients. Here
            the tuple $\mathsf{(\hat{x}, \hat{p}, \hat{v}_1, \hat{v}_2)}$ is
            given as input. Each update is given by the following.

            $\mathsf{p \leftarrow shrink(\hat{p} + {\lambda} (\hat{v}_1
            + a (K\hat{x} - \hat{p})))}$

            $\mathsf{\hat{v}_1 \leftarrow v_1$
            
            $\mathsf{\hat{v}_2 \leftarrow v_2$
            
            $\mathsf{v_1 \leftarrow \hat{v}_1 + \alpha (K\hat{x} - p)}$

            $\mathsf{v_2 \leftarrow \hat{v}_2 + \alpha (Ax - d)}$

            $\mathsf{r \leftarrow  K^\top (2v_1 - \hat{v}_1)
            + A^\top (2v_2 - \hat{v}_2)}$

            $\mathsf{x \leftarrow x - {\beta} r}$

            Args:
                x (tensor): Signal Estimate
                p (tensor): Sparse transform $\mathsf{Kx}$ of signal
                v1 (tensor): Dual variable for sparsity transform constraint $\mathsf{Kx=p}$
                v2 (tensor): Dual variable for linear constraint

            Returns:
                x (tensor): Updated Signal
        """
        x  = x.permute(1, 0).float()
        p  = p.permute(1, 0).float()
        d  = d.permute(1, 0).float()
        v1 = v1.permute(1, 0).float()
        v2 = v2.permute(1, 0).float()

        Kx = torch.mm(self.K.float(), x)
        Ax = torch.mm(self.A.to(device=self.device()), x)
        p  = self.shrink(p + self.lambd * (v1 + self.alpha * (Kx - p)))

        v1_prev = v1.to(self.device())
        v2_prev = v2.to(self.device())

        v1 = v1 + self.alpha * (Kx - p)
        v2 = v2 + self.alpha * (Ax - d)
        r  = self.K.t().mm(2 * v1 - v1_prev).to(self.device())
        r += self.A.to(device=self.device()).t().mm(2 * v2 - v2_prev)
        x  = x - self.beta * r

        x  = x.permute(1, 0)
        p  = p.permute(1, 0)
        v1 = v1.permute(1, 0)
        v2 = v2.permute(1, 0)
        if return_tuple:
            return x, p, v1, v2
        else:
            return x

    def _get_conv_crit(self, x, x_prev, v1, v1_prev, v2, v2_prev, d,
                       tol_fidelity=1.0e-2, tol_residual=1.0e-4,
                       tol_num_stability=1.0e-8):
        """ Identify criteria for whether forward iteration to converges

            Convergence Criteria:
                1. Fidelity must satisfy $\mathsf{\| Ax - d\| \leq tol \cdot ||d||}$
                2. Update residual should be small for x and v, i.e. the
                   expression $\mathsf{\|x^{k+1} - x^k|| + ||v^{k+1} - v^k||}$ is close
                   to zero relative to $\mathsf{\|x^k\| + \|v^k\|}$.

            Note:
                Tolerance is added to `norm_data` to handle the case where
                $\mathsf{d = 0}$.
        """
        norm_res = torch.max(torch.norm(v1 - v1_prev, dim=1))
        norm_res += torch.max(torch.norm(v2 - v2_prev, dim=1))
        norm_res += torch.max(torch.norm(x - x_prev, dim=1))

        norm_res_ref = torch.max(torch.norm(x_prev, dim=1))
        norm_res_ref += torch.max(torch.norm(v1_prev, dim=1))
        norm_res_ref += torch.max(torch.norm(v2_prev, dim=1))

        fidelity = torch.mm(x, self.A.t().to(self.device())) - d
        norm_fidelity = torch.min(torch.norm(fidelity, dim=1))

        norm_data = torch.max(torch.norm(d, dim=1))
        norm_data += tol_num_stability * norm_fidelity

        residual_conv = norm_res <= tol_residual * norm_res_ref
        feasible_sol = norm_fidelity <= tol_fidelity * norm_data
        return residual_conv and feasible_sol

    def forward(self, d: input_data, max_depth=5000,
                depth_warning=False, return_depth=False,
                normalize_K=False, return_certs=False) -> inference:
        """ Compute inference using L-ADMM.

            The aim is to find $\mathsf{v^\star = T(v^\star; d)}$ where
            $\mathsf{v^\star}$ is the dual variable for minimization problem, 
            and $\mathsf{T}$ is the update operation for L-ADMM.
            Associated with optimal dual, we obtain the inference x*.

            Note:
                We write the dual as a tuple $\mathsf{v = (v_1, v_2)}$.
        """
        self.A = self.A.to(self.device())
        d = d.to(self.device()).float()

        with torch.no_grad():
            if normalize_K:
                K_norm = torch.linalg.matrix_norm(self.K, ord=2)
                self.K /= K_norm

            self.depth = 0.0
            x_size = self.A.size()[1]
            d_size = self.A.size()[0]

            x = torch.zeros((d.size()[0], x_size),
                            device=self.device(), dtype=float)
            p = torch.zeros((d.size()[0], x_size),
                            device=self.device(), dtype=float)
            v1 = torch.zeros((d.size()[0], x_size),
                             device=self.device(), dtype=float)
            v2 = torch.zeros((d.size()[0], d_size),
                             device=self.device(), dtype=float)

            x_prev = x.clone()
            all_samp_conv = False
            while not all_samp_conv and self.depth < max_depth:
                v1_prev = v1.clone()
                v2_prev = v2.clone()
                x_prev = x.clone()

                x, p, v1, v2 = self._apply_T(x, p, v1, v2, d,
                                             return_tuple=True)
                all_samp_conv = self._get_conv_crit(x, x_prev, v1, v1_prev,
                                                    v2, v2_prev, d)

                self.depth += 1

        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        Tx = self._apply_T(x, p, v1, v2, d)
        output = [Tx]
        if return_depth:
            output.append(self.depth)
        if return_certs:
            output.append(self.get_certs(Tx.detach(), d))
        return output if len(output) > 1 else Tx


class CT_L2O_Model(ImplicitL2OModel):
    ''' Model to reconstruct CT image from measurements.

        Inferences are defined by

            model(d) = argmin f_theta(Kx)   s.t.   ||Ax - d|| < delta,

        where K, theta, and delta are tunable parameters.
        The forward iteration is Linearized ADMM (L-ADMM) and
        the stepsizes in the algorithm are tunable too.
    '''
    def __init__(self,
                 A,
                 lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 delta=-5.0,
                 K_out_channels=2,
                 max_depth=200):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.max_depth = max_depth
        self.K_out_channels = K_out_channels
        self.fixed_point_error = 0.0
        self.fidelity_rel_norm_error = 0.0

        # trainable parameters
        self.delta = nn.Parameter(delta * torch.ones(1, device=A.device))
        self.alpha = nn.Parameter(alpha*torch.ones(1, device=A.device))
        self.lambd = nn.Parameter(lambd*torch.ones(1, device=A.device))
        self.beta = nn.Parameter(beta*torch.ones(1, device=A.device))
        self.leaky_relu = nn.LeakyReLU(0.1)

        # layers for R
        self.conv1 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        # layers for K
        self.convK = nn.Conv2d(in_channels=1,
                               out_channels=K_out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.convK_T = nn.ConvTranspose2d(in_channels=K_out_channels,
                                          out_channels=1,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False)

    def _get_conv_crit(self, x, x_prev, d, tol=1.0e-2):
        ''' Identify criteria for whether forward iteration to converges

            Criteria implies update residual should be small for x, i.e. the
                   expression |x^{k+1} - x^k|| is close to zero
        '''
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        d = d.view(batch_size, -1)
        x_prev = x_prev.view(batch_size, -1)

        res_norm = torch.max(torch.norm(x - x_prev, dim=1))
        residual_conv = res_norm <= tol

        return residual_conv

    def name(self) -> str:
        return "CTModel"

    def device(self):
        return next(self.parameters()).data.device

    def box_proj(self, x):
        return torch.clamp(x, min=0.0, max=1.0)

    def K(self, x):
        batch_size = x.shape[1]
        x = x.permute(1, 0).view(batch_size, 1, 128, 128)
        x = self.convK(x)
        x = x.view(batch_size, -1).permute(1, 0)
        return x

    def Kt(self, p):
        batch_size = p.shape[1]
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p = self.convK_T(p)
        # reshape back to n_features x n_batches
        p = p.view(batch_size, -1).permute(1, 0)
        return p

    def ball_proj(self, w, d, delta, proj_weight=0.99):
        ''' Project w onto the ball B(d, eps)
        '''

        delta = torch.exp(self.delta)

        dist = torch.norm(w - d, dim=0)
        d_norm = torch.norm(d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              delta*d_norm/dist)
        proj = d + scale * (w - d)

        other = d + delta * d_norm * (w - d) / dist

        if self.training:
            return proj_weight * proj + (1.0 - proj_weight) * other
        else:
            return proj

    def R(self, p):
        batch_size = p.shape[1]
        # reshape to feed into NN
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)

        p = p + self.leaky_relu(self.conv1(p))
        p = p + self.leaky_relu(self.conv2(p))
        p = p + self.leaky_relu(self.conv3(p))
        p = p.view(batch_size, -1).permute(1, 0)
        p = p.view(self.K_out_channels*(128**2), batch_size)

        return p

    def _apply_T(self, x: inference, d: input_data, return_tuple=False):
        ''' Apply model operator using L-ADMM update

            Core functionality is single iteration update for Linearized ADMM,
            which is rearranged to make the signal 'u' update last. This is
            needed to ensure the JFB attaches gradients.
        '''
        batch_size = x.shape[0]

        d = d.view(d.shape[0], -1).to(self.device())
        d = d.permute(1, 0)
        xk = x.view(x.shape[0], -1)
        xk = xk.permute(1, 0)
        pk = self.K(xk)
        wk = torch.matmul(self.A, xk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d.size(), device=self.device())

        alpha = torch.clamp(self.alpha.data, min=0, max=2)
        beta = torch.clamp(self.beta.data, min=0, max=2)
        lambd = torch.clamp(self.lambd.data, min=0, max=2)
        delta = self.delta.data

        # pk step
        pk = pk + lambd*(nuk1 + alpha * (self.K(xk) - pk))
        pk = self.R(pk)

        # wk step
        Axk = torch.matmul(self.A, xk)
        res_temp = nuk2 + alpha * (Axk - wk)
        temp_term = wk + lambd * res_temp
        # temp_term = self.S(wk + lambd * res_temp)
        wk = self.ball_proj(temp_term, d, delta)

        # nuk1 step
        res_temp = self.K(xk) - pk
        nuk1_plus = nuk1 + alpha * res_temp

        # nuk2 step
        res_temp = Axk - wk
        nuk2_plus = nuk2 + alpha * res_temp

        # rk step
        self.convK_T.weight.data = self.convK.weight.data
        rk = self.Kt(2*nuk1_plus - nuk1)
        rk = rk + torch.matmul(self.At, 2*nuk2_plus - nuk2)

        # xk step
        xk = torch.clamp(xk - beta * rk, min=0, max=1)
        
        if return_tuple:
            return xk.permute(1, 0).view(batch_size, 1, 128, 128), nuk1_plus, pk
        else:
            return xk.permute(1, 0).view(batch_size, 1, 128, 128)

    def forward(self, d, depth_warning=False, return_depth=False, tol=1e-3, return_all_vars=False):
        ''' Compute inference using L-ADMM.

            The aim is to find nu* = T(nu*; d) where nu* is the dual variable
            for minimization problem, and T is the update operation for L-ADMM.
            Associated with optimal dual, we obtain the inference u*.
        '''
        with torch.no_grad():

            self.depth = 0.0
            x = torch.zeros((d.size()[0], 1, 128, 128),
                            device=self.device())
            x_prev = np.Inf*torch.ones(x.shape, device=self.device())
            all_samp_conv = False

            while not all_samp_conv and self.depth < self.max_depth:
                x_prev = x.clone()
                x = self._apply_T(x, d)
                all_samp_conv = self._get_conv_crit(x,
                                                    x_prev,
                                                    d,
                                                    tol=tol)

                self.depth += 1

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.fixed_point_error = torch.max(torch.norm(x - x_prev, dim=1))

        if return_depth and return_all_vars==False:
            Tx = self._apply_T(x, d)
            return Tx, self.depth
        elif return_all_vars:
            Tx, nuk1, pk = self._apply_T(x, d, return_tuple=True)
            return Tx, nuk1, pk
        else:
            Tx = self._apply_T(x, d)
            return Tx


class CT_FFPN_Model(nn.Module):
    def __init__(self, D, M, res_net_contraction=0.99):
        super().__init__()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.gamma = res_net_contraction
        self.num_channels = 44
        self.D = D
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_channels,
                                              kernel_size=5, stride=1,
                                              padding=(2, 2)),
                                    nn.Conv2d(self.num_channels,
                                              self.num_channels, kernel_size=5,
                                              stride=1, padding=(2, 2)),
                                    nn.Conv2d(self.num_channels,
                                              self.num_channels, kernel_size=5,
                                              stride=1, padding=(2, 2)),
                                    nn.Conv2d(self.num_channels, 1,
                                              kernel_size=5, stride=1,
                                              padding=(2, 2))])
        self.M = M
        self.Mt = M.t()

    def name(self) -> str:
        return "Regularizer_Net"

    def device(self):
        return next(self.parameters()).data.device

    def _T(self, u, d):
        batch_size = u.shape[0]

        # Learned Gradient
        for idx, conv in enumerate(self.convs):
            if idx + 1 < len(self.convs):
                u_ref = u
            else:
                u_ref = u[:, 0, :, :].view(batch_size, 1, 128, 128)
            u = u_ref + self.leaky_relu(conv(u))
        u = torch.clamp(u, min=0, max=1.0e1)

        # Constraints Projection
        u_vec = u.view(batch_size, -1).to(self.device())
        u_vec = u_vec.permute(1, 0).to(self.device())
        d = d.view(batch_size, -1).to(self.device())
        d = d.permute(1, 0)
        res = torch.matmul(self.Mt, self.M.matmul(u_vec) - d)
        res = 1.99 * torch.matmul(self.D.to(self.device()), res)
        res = res.permute(1, 0)
        res = res.view(batch_size, 1, 128, 128).to(self.device())
        return u - res

    def normalize_lip_const(self, u, d):
        ''' Scale convolutions in R to make it gamma Lipschitz

            It should hold that |R(u,v) - R(w,v)| <= gamma * |u-w| for all u
            and w. If this doesn't hold, then we must rescale the convolution.
            Consider R = I + Conv. To rescale, ideally we multiply R by

                norm_fact = gamma * |u-w| / |R(u,v) - R(w,v)|,

            averaged over a batch of samples, i.e. R <-- norm_fact * R. The
            issue is that ResNets include an identity operation, which we don't
            wish to rescale. So, instead we use

                R <-- I + norm_fact * Conv,

            which is accurate up to an identity term scaled by (norm_fact - 1).
            If we do this often enough, then norm_fact ~ 1.0 and the identity
            term is negligible.
        '''
        noise_u = torch.randn(u.size(), device=self.device())
        w = u.clone() + noise_u
        w = w.to(self.device())
        Twd = self._T(w, d)
        Tud = self._T(u, d)
        T_diff_norm = torch.mean(torch.norm(Twd - Tud, dim=1))
        u_diff_norm = torch.mean(torch.norm(w - u, dim=1))
        R_is_gamma_lip = T_diff_norm <= self.gamma * u_diff_norm
        if not R_is_gamma_lip:
            normalize_factor = (self.gamma * u_diff_norm / T_diff_norm)
            normalize_factor = normalize_factor ** (1.0 / len(self.convs))
            for i in range(len(self.convs)):
                self.convs[i].weight.data *= normalize_factor
                self.convs[i].bias.data *= normalize_factor

    def forward(self, d, eps=1.0e-3, max_depth=100,
                depth_warning=False):
        ''' FPN forward prop

            With gradients detached, find fixed point.
            During forward iteration, u is updated via R(u,Q(d))
            and Lipschitz constant estimates are
            refined. Gradient are attached performing one final step.
        '''
        with torch.no_grad():
            self.depth = 0.0
            u = torch.zeros((d.size()[0], 1, 128, 128),
                            device=self.device())
            u_prev = np.Inf*torch.ones(u.shape, device=self.device())
            all_samp_conv = False
            while not all_samp_conv and self.depth < max_depth:
                u_prev = u.clone()
                u = self._T(u, d)
                res_norm = torch.max(torch.norm(u - u_prev, dim=1))
                self.depth += 1.0
                all_samp_conv = res_norm <= eps

            if self.training:
                self.normalize_lip_const(u, d)

        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        return self._T(u, d)


class CT_UNet_Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
                            torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride=1,
                                            padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels,
                                            out_channels,
                                            kernel_size,
                                            stride=1,
                                            padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1,
                                                     output_padding=1)
                            )
        return expand


class CT_TVM_Model(nn.Module):
    def __init__(self,
                 A, lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 eps=1e-1):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.shrink = torch.nn.Softshrink(lambd=lambd)
        self.eps = eps
        self.model_device = 'cpu'

    def name(self) -> str:
        return "CT_TVM_Model"

    def device(self):
        return self.model_device

    def box_proj(self, u):
        return torch.clamp(u, min=0.0, max=1.0)

    def D(self, u):
        u = u.view(128, 128, u.shape[-1])
        Dux = torch.roll(u, 1, 0) - u
        Dux = Dux.view(128 ** 2, u.shape[-1])
        Duy = torch.roll(u, 1, 1) - u
        Duy = Duy.view(128 ** 2, u.shape[-1])
        Du = torch.cat((Dux, Duy), 0)
        return Du

    def Dt(self, p):
        p = p.view(256, 128, p.shape[1])
        px = p[0:128, :, :]
        Dtpx = torch.roll(px, -1, 0) - px
        Dtpx = Dtpx.view(128 ** 2, p.shape[2])

        py = p[128:256, :, :]
        Dtpy = torch.roll(py, -1, 1) - py
        Dtpy = Dtpy.view(128 ** 2, p.shape[2])
        Dtp = Dtpx + Dtpy
        return Dtp

    def ball_proj(self, w, d, eps):
        ''' Project w onto the ball B(d, eps)
        '''
        dist = torch.norm(w - d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        d_norm = torch.norm(d, dim=0)

        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              self.eps*d_norm/dist)
        proj = d + scale * (w - d)

        return proj

    def forward(self, d, tol=1.0e-3, max_depth=500,
                depth_warning=False):

        self.depth = 0.0

        # Initialize sequences
        self.model_device = d.device
        d = d.view(d.size()[0], -1).to(self.device())
        d = d.permute(1, 0)
        uk = torch.zeros((128 ** 2, d.size()[1]), device=self.device())
        pk = self.D(uk)
        wk = torch.matmul(self.A, uk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d.size(), device=self.device())

        for _ in range(max_depth):

            # TVM updates
            res1 = self.Dt(nuk1 + self.alpha * (self.D(uk) - pk))
            Auk = torch.matmul(self.A, uk)
            res2 = torch.matmul(self.At, nuk2 + self.alpha * (Auk - wk))
            rk = self.beta * (res1 + res2)
            uk = self.box_proj(uk - rk)

            res = self.lambd * (nuk1 + self.alpha * (self.D(uk)-pk))
            pk = self.shrink(pk + res)

            Auk = torch.matmul(self.A, uk)
            res = self.lambd * (nuk2 + self.alpha * (Auk - wk))
            wk = self.ball_proj(wk + res, d, self.eps)

            nuk1 = nuk1 + self.alpha * (self.D(uk) - pk)
            nuk2 = nuk2 + self.alpha * (torch.matmul(self.A, uk) - wk)

        self.depth = max_depth
        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        uk = uk.permute(1, 0)
        return uk.view(uk.shape[0], 1, 128, 128)
