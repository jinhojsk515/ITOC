"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""
from typing import Any, Callable, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm
import numpy as np
from metrics import reward_function


class StableDiffusion():
    def __init__(self,
                 NFE: int = 50,
                 model_key: str = "runwayml/stable-diffusion-v1-5",  # "runwayml/stable-diffusion-v1-5" "pt-sk/stable-diffusion-1.5"
                 device: Optional[torch.device] = torch.device("cuda"), deterministic: bool = False,
                 **kwargs):
        self.device = device
        self.NFE = NFE
        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.unet = pipe.unet.eval()
        self.soc_pipeline = pipe

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()

        self.sigmas = (1 - self.total_alphas).sqrt() / self.total_alphas.sqrt()
        self.log_sigmas = self.sigmas.log()

        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(NFE, device=device)
        self.skip = total_timesteps // NFE

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

        self.reward_func = None
        self.reward_fn = lambda x, prompt: reward_function(
            x, prompt,
            model=self.soc_pipeline,
            reward_func=self.reward_func,
            verbose=False
        )
        self.DETERMINISTIC = deterministic

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        tokens = self.tokenizer([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        return self.text_encoder(tokens).last_hidden_state.detach()

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        zt = 1 / 0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def get_eps_prediction(self, z_t, timestep, text_embeddings, guidance_scale=1.0):
        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            unet = self.unet_init if hasattr(self, 'unet_init') else self.unet
            e_t = unet(latent_input, timestep, embedd).sample
            e_t_uncond, e_t = e_t.chunk(2)
            e_t_w = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t_w).all()
        return e_t_w, (e_t, e_t_uncond)

    @torch.no_grad()
    def inversion(self, z0, embedding_source, noise_limit=1000):
        zt = z0.clone().to(self.device)
        skip = self.scheduler.alphas_cumprod.shape[0] // len(self.scheduler.timesteps)
        alpha = lambda tt: self.scheduler.alphas_cumprod[tt] if tt >= 0 else self.scheduler.final_alpha_cumprod

        traj = [zt]
        pbar = reversed(range(self.scheduler.alphas_cumprod.shape[0] - 2, -1, -skip))
        for i, t in enumerate(pbar):  # 0,1,2,...     19,39,59,...
            if t >= noise_limit:
                traj.append(zt)
                continue
            at, at_prev = alpha(t), alpha(t - skip)
            noise_pred, _ = self.get_eps_prediction(zt, torch.tensor([t]).to(self.device), embedding_source)
            z0t = (zt - (1 - at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred
            traj.append(zt)
        return zt, traj

    def initialize_latent(self,
                          method: str = 'random',
                          src_img: Optional[torch.Tensor] = None,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            z = torch.randn(size).to(self.device, dtype=self.dtype)
        elif method == 'random_kdiffusion':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else:
            raise NotImplementedError

        return z.requires_grad_()

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def calculate_term(self, eps_n, t, zt):
        at = self.alpha(t)
        at_prev = self.alpha(t - self.skip)
        if self.DETERMINISTIC:  sigma_t = 0.
        else:   sigma_t = ((1 - at_prev) / (1 - at)).sqrt() * (1 - at / at_prev).sqrt()
        z0t = (zt - (1 - at).sqrt() * eps_n) / at.sqrt()
        zt = at_prev.sqrt() * z0t + (1 - at_prev - sigma_t ** 2).sqrt() * eps_n
        return zt

    def grad_rewards(self, x, prompts, get_grad=True):
        # Original implementation
        if not get_grad:
            with torch.enable_grad():
                image = self.decode(x)
                reward_values = self.reward_fn(image.clamp(-1, 1), prompts)
                output = torch.zeros(x.shape)
                return output.detach(), reward_values  # .detach()
        with torch.enable_grad():
            x = x.requires_grad_(True)
            image = self.decode(x)
            reward_values = self.reward_fn(image.clamp(-1, 1), prompts)
            output = torch.autograd.grad(
                reward_values.sum(), x
            )[0]
            return output.detach(), reward_values  # .detach()

    def grad_inner_product(
            self,
            x: torch.Tensor,
            t: int,
            vectors: torch.Tensor,
            prompt_embeds: torch.Tensor, b_grad=None,
            eta: float = 1.0,
            use_clipped_model_output: bool = False,
            generator=None, adjoint_cfg_scale = 1.0
    ):
        def inner_product(x):
            # x with shape (batch_size, 4, 64, 64)
            x_model_input = torch.cat([x] * 2) if adjoint_cfg_scale != 1.0 else x
            #x_model_input = self.soc_pipeline.scheduler.scale_model_input(x_model_input, t)
            noise_pred = self.unet(x_model_input.to(self.unet.dtype), t, encoder_hidden_states=prompt_embeds.to(self.unet.dtype), return_dict=False, )[0]

            # perform guidance
            if adjoint_cfg_scale != 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + adjoint_cfg_scale * (noise_pred_text - noise_pred_uncond)

            prev_sample_init = self.calculate_term(noise_pred, t, x)

            sum_inner_prod = torch.sum((prev_sample_init - x) * vectors, dim=[0, 1, 2, 3])
            return sum_inner_prod, noise_pred, prev_sample_init

        with torch.enable_grad():
            if b_grad is not None:
                return b_grad * vectors, torch.zeros_like(x), torch.zeros_like(x)
            x = x.requires_grad_(True)  # bchw
            sum_inner_prod, noise_pred, prev_pred = inner_product(x)  # scalar, bchw
            output = torch.autograd.grad(  # bchw
                sum_inner_prod,
                x,
            )[0]
            x = x.requires_grad_(False)
            return output.detach(), noise_pred.detach(), prev_pred.detach()

    def compute_adjoints(
            self,
            all_x_t: torch.Tensor,
            all_t: torch.Tensor,
            prompt_embeds: torch.Tensor,
            prompts: list[str],
            get_grad: bool = True, initial_grad: Optional[tuple] = None, reward_multiplier=1.0, b_grad=None,
            **kwargs,
    ):
        if initial_grad is not None:
            reward_grads, reward_values = initial_grad
        else:
            reward_grads, reward_values = self.grad_rewards(all_x_t[:, -1], prompts, get_grad=get_grad)
            # reward_grads, reward_values = torch.randn_like(all_x_t[:,-1]), torch.randn(all_x_t.shape[0], device=all_x_t.device)
        assert all_x_t[:, :-1].shape[1] == len(all_t)
        num_timesteps = all_x_t.shape[1]

        with torch.no_grad():
            adjoint_states = torch.zeros_like(all_x_t)
            all_noise_pred_init = torch.zeros_like(all_x_t[:, :-1])
            all_prev_pred_init = torch.zeros_like(all_x_t[:, :-1])
            if not get_grad:    return adjoint_states, reward_values, all_noise_pred_init
            a = -reward_multiplier * reward_grads.to(torch.float32)
            adjoint_states[:, -1] = a

            for k in range(num_timesteps - 2, -1, -1):
                grad_inner_prod, noise_pred_init, prev_pred_init = self.grad_inner_product(
                    all_x_t[:, k],
                    all_t[k],
                    a,
                    prompt_embeds, b_grad=b_grad[k] if b_grad is not None else None,
                    **kwargs,
                )
                a += grad_inner_prod
                adjoint_states[:, k] = a
                all_noise_pred_init[:, k] = noise_pred_init
                all_prev_pred_init[:, k] = prev_pred_init
                del grad_inner_prod, noise_pred_init, prev_pred_init

        return adjoint_states, reward_values, all_noise_pred_init, all_prev_pred_init
