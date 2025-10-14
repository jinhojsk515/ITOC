import torch
from torch.autograd import grad
from diffusers.utils.torch_utils import randn_tensor


class TFGGuidance:
    def __init__(self, args, **kwargs):
        #super(TFGGuidance, self).__init__(args, **kwargs)
        self.args = args
        self.device = args.device
        self.generator = torch.manual_seed(self.args.seed)

        def noise_fn(x, sigma, **kwargs):
            noise = randn_tensor(x.shape, generator=self.generator, device=self.args.device, dtype=x.dtype)
            return sigma * noise + x
        self.noise_fn = noise_fn

    def rescale_grad(self, grad: torch.Tensor, clip_scale, **kwargs):  # [B, N, 3+5]
        node_mask = kwargs.get('node_mask', None)

        scale = (grad ** 2).mean(dim=-1)
        if node_mask is not None:  # [B, N, 1]
            scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
            clipped_scale = torch.clamp(scale, max=clip_scale)
            co_ef = clipped_scale / scale  # [B]
            grad = grad * co_ef.view(-1, 1, 1)
        return grad

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):
        #flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        #outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)
        #avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
        reward_values = kwargs['rewardfn'](x0 + mc_eps.squeeze(0))
        if return_logp:
            return reward_values

        _grad = torch.autograd.grad(reward_values.sum(), x0)[0]
        _grad = self.rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad

    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)])
        # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':  # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':  # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':  # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)
        elif self.args.sigma_schedule == 'zero':  # 0
            scheduler = torch.zeros_like(alpha_prod_ts)

        return self.args.sigma * scheduler[t]

    def guide_step(self, x: torch.Tensor, t: int, unet, ts: torch.LongTensor, alpha_prod_ts: torch.Tensor, alpha_prod_t_prevs: torch.Tensor, eta: float, **kwargs, ):
        # ts: [50], 999, ..., 19
        # alpha_prod_ts: [50], 0.0047, 0.0060, ..., 0.9832
        # alpha_prod_t_prevs: [50], 0.0060, ..., 0.9832, 0.9991
        # x: [1, 4, 64, 64], t: int(15, 16, ..., 49), eta=0.0
        #print(len(ts), len(alpha_prod_ts), len(alpha_prod_t_prevs), x.shape, t, eta)
        #print(alpha_prod_ts[:5], alpha_prod_t_prevs[:5])

        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]

        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(t, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(t, alpha_prod_ts, alpha_prod_t_prevs)

        #t = ts[t]  # convert from int space to tensor space

        for recur_step in range(self.args.recur_steps):

            # sample noise to estimate the \tilde p distribution
            mc_eps = self.get_noise(std, x.shape, self.args.eps_bsz, **kwargs)
            mc_eps.requires_grad_(False)

            # Compute guidance on x_t, and obtain Delta_t
            if rho >= 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()
                    #x0 = self._predict_x0(x_g, unet(x_g, t), alpha_prod_t, **kwargs)
                    x0 = (x_g - (1 - alpha_prod_t) ** (0.5) * unet(x_g, t)) / (alpha_prod_t ** (0.5))
                    logprobs = self.tilde_get_guidance(x0, mc_eps, return_logp=True, **kwargs)
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = self.rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                    Delta_t = Delta_t * rho

            else:
                Delta_t = torch.zeros_like(x)
                #x0 = self._predict_x0(x, unet(x, t), alpha_prod_t, **kwargs)
                x0 = (x - (1 - alpha_prod_t) ** (0.5) * unet(x, t)) / (alpha_prod_t ** (0.5))

            # Compute guidance on x_{0|t}
            new_x0 = x0.clone().detach()
            for _ in range(self.args.iter_steps):
                if mu != 0.0:
                    new_x0 += mu * self.tilde_get_guidance(new_x0.detach().requires_grad_(), mc_eps, **kwargs)
            Delta_0 = new_x0 - x0

            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            alpha_t = alpha_prod_t / alpha_prod_t_prev
            x_prev = self._predict_x_prev_from_zero(x, x0, alpha_prod_t, alpha_prod_t_prev, eta, ts[t], **kwargs)
            x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5

            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)

        #return x_prev, (logprobs.item(), x0)
        return x_prev.detach(), (logprobs.item(), None)

    def _predict_x_prev_from_zero(self, xt: torch.Tensor, x0: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, eta: float, t: torch.LongTensor, **kwargs, ) -> torch.Tensor:
        new_epsilon = ((xt - alpha_prod_t ** (0.5) * x0) / (1 - alpha_prod_t) ** (0.5))
        return self._predict_x_prev_from_eps(xt, new_epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

    def _predict_x_prev_from_eps(self, xt: torch.Tensor, eps: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, eta: float, t: torch.LongTensor, **kwargs, ) -> torch.Tensor:
        sigma = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev - sigma ** 2) ** (0.5) * eps
        pred_x0_direction = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_x0_direction + pred_sample_direction
        if eta > 0 and t.item() > 0:    prev_sample = self.noise_fn(prev_sample, sigma, **kwargs)
        return prev_sample

    def _predict_xt(self, x_prev: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, **kwargs, ) -> torch.Tensor:
        xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** (0.5) * x_prev
        noise = randn_tensor( x_prev.shape, generator=self.generator, device=self.args.device, dtype=x_prev.dtype)
        return xt_mean + (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) * noise


class TFGGuidance_SD3:
    def __init__(self, args, **kwargs):
        #super(TFGGuidance, self).__init__(args, **kwargs)
        self.args = args
        self.device = args.device
        self.generator = torch.manual_seed(self.args.seed)

        def noise_fn(x, sigma, **kwargs):
            noise = randn_tensor(x.shape, generator=self.generator, device=self.args.device, dtype=x.dtype)
            return sigma * noise + x
        self.noise_fn = noise_fn

    def rescale_grad(self, grad: torch.Tensor, clip_scale, **kwargs):  # [B, N, 3+5]
        node_mask = kwargs.get('node_mask', None)

        scale = (grad ** 2).mean(dim=-1)
        if node_mask is not None:  # [B, N, 1]
            scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
            clipped_scale = torch.clamp(scale, max=clip_scale)
            co_ef = clipped_scale / scale  # [B]
            grad = grad * co_ef.view(-1, 1, 1)
        return grad

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):
        #print(x0.shape, x0.dtype, mc_eps.shape, mc_eps.dtype)
        #flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        #outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)
        #avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
        reward_values = kwargs['rewardfn'](x0 + mc_eps.squeeze(0))
        if return_logp:
            return reward_values

        _grad = torch.autograd.grad(reward_values.sum(), x0)[0]
        _grad = self.rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad

    def get_noise(self, std, x, eps_bsz=4, **kwargs):
        shape = x.shape
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device, dtype=x.dtype)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device, dtype=x.dtype), std, **kwargs) for _ in range(eps_bsz)])
        # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':  # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':  # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':  # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)
        elif self.args.sigma_schedule == 'zero':  # 0
            scheduler = torch.zeros_like(alpha_prod_ts)

        return self.args.sigma * scheduler[t]

    def guide_step(self, x: torch.Tensor, t: int, unet, ts: torch.LongTensor, t_prevs, eta: float, **kwargs, ):
        # ts: [50], 999, ..., 19
        # alpha_prod_ts: [50], 0.0047, 0.0060, ..., 0.9832
        # alpha_prod_t_prevs: [50], 0.0060, ..., 0.9832, 0.9991
        # x: [1, 4, 64, 64], t: int(15, 16, ..., 49), eta=0.0

        # ts: [28], 1.000, 0.9873, ..., 0.0089
        # t_prevs: [28], 0.9873, ..., 0.0089, 0.0000
        # x: [1, 4, 64, 64], t: int(10, 16, ..., 27), eta=0.0
        alpha_prod_ts = ts
        alpha_prod_t_prevs = t_prevs


        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        dt = alpha_prod_t - alpha_prod_t_prev

        rho = self.get_rho(t, 1-alpha_prod_ts, 1-alpha_prod_t_prevs)
        mu = self.get_mu(t, 1-alpha_prod_ts, 1-alpha_prod_t_prevs)
        std = self.get_std(t, 1-alpha_prod_ts, 1-alpha_prod_t_prevs)

        #t = ts[t]  # convert from int space to tensor space

        for recur_step in range(self.args.recur_steps):

            # sample noise to estimate the \tilde p distribution
            mc_eps = self.get_noise(std, x, self.args.eps_bsz, **kwargs)
            mc_eps.requires_grad_(False)

            # Compute guidance on x_t, and obtain Delta_t
            if rho >= 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()

                    #x0 = (x_g - (1 - alpha_prod_t) ** (0.5) * unet(x_g, t)) / (alpha_prod_t ** (0.5))
                    x0 = x_g - alpha_prod_t * unet(x_g, alpha_prod_t*1000.)
                    logprobs = self.tilde_get_guidance(x0, mc_eps, return_logp=True, **kwargs)
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = self.rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                    Delta_t = Delta_t * rho

            else:
                Delta_t = torch.zeros_like(x)
                #x0 = (x - (1 - alpha_prod_t) ** (0.5) * unet(x, t)) / (alpha_prod_t ** (0.5))
                x0 = x_g - alpha_prod_t * unet(x, alpha_prod_t*1000.)

            # Compute guidance on x_{0|t}
            new_x0 = x0.clone().detach()
            for _ in range(self.args.iter_steps):
                if mu != 0.0:
                    new_x0 += mu * self.tilde_get_guidance(new_x0.detach().requires_grad_(), mc_eps, **kwargs)
            Delta_0 = new_x0 - x0

            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            #alpha_t = alpha_prod_t / alpha_prod_t_prev
            #x_prev = self._predict_x_prev_from_zero(x, x0, alpha_prod_t, alpha_prod_t_prev, eta, ts[t], **kwargs) + Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5
            x_prev = x - dt * unet(x, alpha_prod_t*1000.) + Delta_t + Delta_0 * (1-alpha_prod_t_prev)

            #x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)
            drift = -x_prev / (1.0 - alpha_prod_t_prev)
            sigma = (2.0 * alpha_prod_t_prev * dt / (1.0 - alpha_prod_t_prev)) ** 0.5
            x = x_prev + drift * dt + sigma * torch.randn_like(x_prev)

        #return x_prev, (logprobs.item(), x0)
        return x_prev.detach(), (logprobs.item(), None)

    def _predict_x_prev_from_zero(self, xt: torch.Tensor, x0: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, eta: float, t: torch.LongTensor, **kwargs, ) -> torch.Tensor:
        new_epsilon = ((xt - alpha_prod_t ** (0.5) * x0) / (1 - alpha_prod_t) ** (0.5))
        return self._predict_x_prev_from_eps(xt, new_epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

    def _predict_x_prev_from_eps(self, xt: torch.Tensor, eps: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, eta: float, t: torch.LongTensor, **kwargs, ) -> torch.Tensor:
        sigma = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev - sigma ** 2) ** (0.5) * eps
        pred_x0_direction = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_x0_direction + pred_sample_direction
        if eta > 0 and t.item() > 0:    prev_sample = self.noise_fn(prev_sample, sigma, **kwargs)
        return prev_sample

    def _predict_xt(self, x_prev: torch.Tensor, alpha_prod_t: torch.Tensor, alpha_prod_t_prev: torch.Tensor, **kwargs, ) -> torch.Tensor:
        xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** (0.5) * x_prev
        noise = randn_tensor( x_prev.shape, generator=self.generator, device=self.args.device, dtype=x_prev.dtype)
        return xt_mean + (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) * noise
