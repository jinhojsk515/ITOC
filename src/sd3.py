from typing import List, Tuple, Optional
from diffusers import StableDiffusion3Pipeline
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from metrics import reward_function
from torchvision.io import read_image


class StableDiffusion3Base():
    def __init__(self, model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', dtype=torch.float16, NFE: int = 28, deterministic=False):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        self.pipe = pipe

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = pipe.vae
        self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8)

        self.NFE = NFE
        self.scheduler.set_timesteps(NFE, device=self.device)

        self.reward_func = None
        self.reward_fn = lambda x, prompt: reward_function(
            x, prompt,
            model=self.pipe,
            reward_func=self.reward_func,
            verbose=False
        )
        self.DETERMINISTIC = deterministic

        del pipe

    def encode_prompt(self, prompt: List[str], batch_size: int = 1):
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.text_enc_1.device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_1.device)

        text_clip2_ids = self.tokenizer_2(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.text_enc_2.device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_2.device)

        # T5 encode (used for text condition)
        text_t5_ids = self.tokenizer_3(prompt,
                                       padding="max_length",
                                       max_length=77,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors='pt').input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.text_enc_3.device)

        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb

    def initialize_latent(self, img_size: Tuple[int, int], batch_size: int = 1, **kwargs):
        H, W = img_size
        lH, lW = H // self.vae_scale_factor, W // self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def noise_inject(self, z: torch.Tensor, eps: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None):
        if eps is None:         eps = torch.randn_like(z, device=self.device, dtype=self.dtype)
        if timestep is None:    timestep = torch.rand(z.shape[0], device=self.device, dtype=self.dtype) * self.scheduler.config.num_train_timesteps
        t = timestep / self.scheduler.config.num_train_timesteps
        return z*(1-t) + eps*t, eps, timestep, t

    def inversion_uncond(self, src_img, null_emb, limit_t:int):

        # encode text prompts
        with torch.no_grad():
            #null_prompt_emb, null_pooled_emb = self.encode_prompt([""])
            null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        #src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        #with torch.no_grad():
            #z = self.encode(src_img).to(self.transformer.device)
            #z0 = z.clone()
        z = src_img
        z_traj = [z]

        # timesteps (default option. You can make your custom here.)
        #self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        for i, t in enumerate(timesteps[:-1]):
            if i >= limit_t:    break
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                v_final, _ = self.get_v_prediction(z, timestep, None, None, null_prompt_emb, null_pooled_emb, cfg_scale=0)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            z = z + (sigma_next - sigma) * v_final
            z_traj.append(z)
        return z, z_traj

    def get_q_traj(self, zt, limit_t):
        timesteps = self.scheduler.timesteps  # 1000 -> 0
        ts = timesteps / self.scheduler.config.num_train_timesteps  # 1.0 -> 0.0
        noise_injected = []

        traj = [zt]
        for k in range(len(timesteps)):
            if k > limit_t:    break
            t = ts[-(k+1)]  # 1(noise) → 0(clean)
            #t = ts[-k] if k!=0 else 1e-4 # 1(noise) → 0(clean)
            dt = ts[-(k+1)] if k==0 else ts[-(k+1)] - ts[-k]
            #print(k, t, dt)
            drift = -zt / (1.0 - t)
            sigma = (2.0 * t / (1.0 - t)) ** 0.5
            zt = zt + drift * dt + sigma * dt**0.5 * torch.randn_like(zt)
            traj.append(zt)
            noise_injected.append((zt - (1-t) * traj[0]) / t)
        return zt, traj, noise_injected

    def get_v_prediction(self, z, t, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale=0.0):
        pred_null_v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=null_pooled_emb,
                             encoder_hidden_states=null_prompt_emb,
                             return_dict=False)[0]
        if cfg_scale == 0.0:    return pred_null_v, (None, pred_null_v)
        pred_v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return pred_null_v + cfg_scale * (pred_v - pred_null_v), (pred_v, pred_null_v)

    def sample(self, prompts: List[str], img_shape: Optional[Tuple[int, int]] = None, cfg_scale: float = 1.0, batch_size: int = 1,
               latent: Optional[List[torch.Tensor]] = None,
               prompt_emb: Optional[List[torch.Tensor]] = None,
               null_emb: Optional[List[torch.Tensor]] = None,
               ):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        timesteps = self.scheduler.timesteps        # 1000 -> 0
        sigmas = timesteps / self.scheduler.config.num_train_timesteps      # 1.0 -> 0.0

        # Solve SDE
        pbar = tqdm(timesteps, total=self.NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            v_final, _ = self.get_v_prediction(z, timestep, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale=cfg_scale)

            sigma = sigmas[i]# if i!=0 else 1-1e-3
            sigma_next = sigmas[i + 1] if i + 1 < self.NFE else 0.0
            dt = sigma - sigma_next
            if i==0:    z = z - dt * v_final
            else:   z = z - (2*v_final + z/(1-sigma_next)) * dt + dt**0.5 * (2*sigma_next/(1-sigma_next))**0.5 * torch.randn_like(z)

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

    def calculate_term(self, v, t, zt, dt):
        if self.DETERMINISTIC:  return zt - v * dt/self.scheduler.config.num_train_timesteps
        else:   return zt - (2*v + zt/(1-(t-dt)/self.scheduler.config.num_train_timesteps)) * dt/self.scheduler.config.num_train_timesteps

    def grad_rewards(self, x, prompts, get_grad=True):
        # Original implementation
        if not get_grad:
            with torch.enable_grad():
                image = self.decode(x)
                reward_values = self.reward_fn(image.clamp(-1, 1), prompts)
                output = torch.zeros(x.shape)
                return output.detach(), reward_values
        with torch.enable_grad():
            x = x.requires_grad_(True)
            image = self.decode(x)
            reward_values = self.reward_fn(image.clamp(-1, 1), prompts)
            output = torch.autograd.grad(
                reward_values.sum(), x
            )[0]
            return output.detach(), reward_values

    def grad_inner_product(
            self,
            x: torch.Tensor,
            t,
            vectors: torch.Tensor,
            null_prompt_emb, null_pooled_emb, dt, b_grad=None,
            eta: float = 1.0,
            use_clipped_model_output: bool = False,
            generator=None, adjoint_cfg_scale = 1.0
    ):
        def inner_product(x):
            # x with shape (batch_size, 4, 64, 64)
            x_model_input = torch.cat([x] * 2) if adjoint_cfg_scale != 1.0 else x
            #x_model_input = self.soc_pipeline.scheduler.scale_model_input(x_model_input, t)
            # noise_pred = self.unet(x_model_input.to(self.unet.dtype), t, encoder_hidden_states=prompt_embeds.to(self.unet.dtype), return_dict=False, )[0]
            noise_pred, _ = self.get_v_prediction(x_model_input.to(self.dtype), t, None, None, null_prompt_emb, null_pooled_emb, cfg_scale=0.0)

            # perform guidance
            if adjoint_cfg_scale != 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + adjoint_cfg_scale * (noise_pred_text - noise_pred_uncond)

            prev_sample_init = self.calculate_term(noise_pred, t, x, dt)

            sum_inner_prod = torch.sum((prev_sample_init - x) * vectors, dim=[0, 1, 2, 3])
            return sum_inner_prod, noise_pred, prev_sample_init

        with torch.enable_grad():
            if b_grad is not None:
                print(b_grad.shape, vectors.shape)
                return b_grad*vectors, torch.zeros_like(x), torch.zeros_like(x)
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
            null_prompt_emb, null_pooled_emb,
            prompts,
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
            a = -reward_multiplier * reward_grads.to(self.dtype)
            adjoint_states[:, -1] = a

            for k in range(num_timesteps - 2, -1, -1):
                #print('aa', torch.isnan(b_grad[k]).any() if b_grad is not None else None, torch.isnan(a).any())
                grad_inner_prod, noise_pred_init, prev_pred_init = self.grad_inner_product(
                    all_x_t[:, k],
                    all_t[k],
                    a,
                    null_prompt_emb, null_pooled_emb, dt=all_t[k] if k==num_timesteps-2 else all_t[k]-all_t[k + 1], b_grad=b_grad[k] if b_grad is not None else None,
                    **kwargs,
                )
                a += grad_inner_prod
                adjoint_states[:, k] = a
                #print('aaa', torch.isfinite(a.all()))
                all_noise_pred_init[:, k] = noise_pred_init
                all_prev_pred_init[:, k] = prev_pred_init
                del grad_inner_prod, noise_pred_init, prev_pred_init

        return adjoint_states, reward_values, all_noise_pred_init, all_prev_pred_init


if __name__ == "__main__":
    sd3 = StableDiffusion3Base(NFE=28)
    sd3.text_enc_1.to(sd3.device)
    sd3.text_enc_2.to(sd3.device)
    sd3.text_enc_3.to(sd3.device)
    sd3.vae.to(sd3.device)
    sd3.transformer.to(sd3.device)

    print("Stable Diffusion 3 Base initialized successfully.")
    prompts = ["A beautiful landscape with mountains and a river", ]

    img = sd3.sample(prompts, img_shape=(1024, 1024), cfg_scale=4.0)
    print(f"Generated image shape: {img.shape}, {img.min()}, {img.max()}")
    save_image(img.clamp(-1, 1), f'./output_sd3.png', normalize=True)

    image_pix = read_image("./output_sd3.png").unsqueeze(0) / 255.0
    image_pix = image_pix * 2 - 1
    image_pix = image_pix.to(sd3.device, dtype=sd3.dtype)
    image_enc = sd3.encode(image_pix)
    image_pix_reconstructed = sd3.decode(image_enc).clamp(-1, 1)
    save_image(image_pix_reconstructed.clamp(-1, 1), f'./output_sd3_recon.png', normalize=True)
    print("error in reconstruction:", (image_pix - image_pix_reconstructed).abs().mean().item())

    with torch.no_grad():
        prompt_emb, pooled_emb = sd3.encode_prompt(prompts, 1)
        null_prompt_emb, null_pooled_emb = sd3.encode_prompt([""], 1)
    #zt = sd3.initialize_latent((1024, 1024), 1)
    #v_final = sd3.get_v_prediction(zt, torch.tensor([500]).to(sd3.device), prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale=4.0)
    #v_final, _ = sd3.get_v_prediction(zt, torch.tensor([500.25]).to(sd3.device), prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale=0.0)
    #print(f"V prediction shape: {v_final.shape}")
    sd3.text_enc_1.to('cpu')
    sd3.text_enc_2.to('cpu')
    sd3.text_enc_3.to('cpu')
    del sd3.text_enc_1, sd3.text_enc_2, sd3.text_enc_3
    print("Tokenizers and text encoders removed to save memory.")

    sd3.reward_func = 'Clip-Score'
    prompts = {'text': prompts}
    image = torch.randn(1, 16, 128, 128).to(sd3.device, dtype=sd3.dtype)  # Example image tensor
    trajectory = torch.randn(1, 15, 16, 128, 128).to(sd3.device, dtype=sd3.dtype)  # Example trajectory tensor
    all_t = torch.tensor([[500.25]] * 14).to(sd3.device, dtype=sd3.dtype)  # Example timesteps
    rg, rv = sd3.grad_rewards(image, prompts)
    #rg, rv = torch.randn_like(image), torch.randn(1, device=image.device)  # Example gradients and rewards
    adjoint_states, reward_values, all_noise_pred_init = sd3.compute_adjoints(trajectory, all_t, null_prompt_emb, null_pooled_emb,
                                                                               prompts, initial_grad=(rg, rv), reward_multiplier=200)
    print(f"Adjoint states shape: {adjoint_states.shape}, {reward_values.shape}, {all_noise_pred_init.shape}")
    print(torch.isfinite(adjoint_states).all(), torch.isfinite(reward_values).all(), torch.isfinite(all_noise_pred_init).all())
