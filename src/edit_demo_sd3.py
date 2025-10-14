import torch
import torch.nn.functional as F
import argparse
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tfg import TFGGuidance_SD3
from sd3 import StableDiffusion3Base



config = argparse.ArgumentParser()
# task(method, reward)
config.add_argument("--method_name", type=str, default=None, choices=["gradient_ascent", "itoc", "inversion_tfg", "inversion_dps", "inversion_freedom"])
config.add_argument("--reward_name", type=str, default=None, choices=['ImageReward', 'ImageNet1k_classifier', 'Clip_Score', 'Gram_Diff'])
config.add_argument("--resolution", type=int, default=768)
# input
config.add_argument("--image_path", type=str, default=None)
config.add_argument("--reward_prompt", type=str, default="colorful painting, river flowing grass field with flowers.")
config.add_argument("--style_image_path", type=str, default="style_ref.png")
config.add_argument("--reward_class", type=int, default=306)
# hyperparameters
config.add_argument("--deterministic", type=bool, default=True)             # ITOC
config.add_argument("--n_iter", type=int, default=20)                       # GA, ITOC
config.add_argument("--reward_multiplier", type=float, default=200.0)       # ITOC
config.add_argument("--depth", type=float, default=0.7)                     # DPS, FreeDoM, TFG, ITOC
config.add_argument("--lr", type=float, default=5e-3)                       # GA, ITOC
config.add_argument("--tfg_rho", type=float, default=5.0)                   # DPS, FreeDoM, TFG
config.add_argument("--tfg_mu", type=float, default=0.0)                    # TFG
config = config.parse_args()

config.seed = 42
resolution = config.resolution
METHOD_NAME = config.method_name
REWARD_NAME = config.reward_name
DETERMINISTIC = config.deterministic

def save_latent(pipe, latent, filename, source_img=None):
    image_pix = pipe.decode(latent.to(pipe.dtype)).clamp(-1, 1)
    save_image(image_pix[0], filename, normalize=True)
    if source_img is not None:
        diff = ((latent - source_img)[0]**2).sum(dim=0).sqrt().clamp(0., 1.)
        plt.matshow(diff.detach().cpu().numpy())
        plt.savefig(filename.replace('.png', '_diff.png'))

def all_grad_rewards(trainer, image, prompts):
    metrics, metrics_w = [REWARD_NAME], [1.0]

    all_rg, all_rv = [], []
    for w, metric in zip(metrics_w, metrics):
        trainer.reward_func = metric

        if 'inversion' in METHOD_NAME:    rg, rv = trainer.grad_rewards(image, prompts, get_grad=False)
        else:   rg, rv = trainer.grad_rewards(image, prompts)

        all_rg.append(w * rg)
        all_rv.append(w * rv)
    all_rg = torch.stack(all_rg, dim=0).sum(dim=0)
    all_rv = torch.stack(all_rv, dim=0).sum(dim=0)
    return all_rg, all_rv

def main():
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.rho_schedule = "increase"  # guidance scale on xt
    config.mu_schedule = "increase"  # guidance scale on x0|t
    config.sigma = 0.1  # noise scale on tweedie, gamma in the paper
    config.eps_bsz = 1
    config.clip_scale = 1.0
    if 'tfg' in METHOD_NAME:    config.recur_steps, config.iter_steps, config.mu, config.sigma_schedule, config.rho = 1, 4, config.tfg_mu, "constant", config.tfg_rho  # TFG
    if 'dps' in METHOD_NAME:    config.recur_steps, config.iter_steps, config.mu, config.sigma_schedule, config.rho = 1, 0, 0, "zero", config.tfg_rho  # DPS
    if 'freedom' in METHOD_NAME:    config.recur_steps, config.iter_steps, config.mu, config.sigma_schedule, config.rho = 2, 0, 0, "zero", config.tfg_rho  # FreeDoM
    TFG = TFGGuidance_SD3(args=config)

    device = config.device
    sd3 = StableDiffusion3Base(NFE=56, device=device, dtype=torch.float16, deterministic=DETERMINISTIC)
    sd3.text_enc_1.to(sd3.device)
    sd3.text_enc_2.to(sd3.device)
    sd3.text_enc_3.to(sd3.device)
    sd3.vae.to(sd3.device)
    sd3.transformer.to(sd3.device)


    # ==================== Load images ==================== #
    if REWARD_NAME == 'ImageNet1k_classifier':
        prompts =config.reward_class
    elif REWARD_NAME == 'Gram_Diff':
        prompts = config.style_image_path
    else:   prompts = [config.reward_prompt]
    image_pix = read_image(config.image_path).unsqueeze(0).float() / 255.0
    if image_pix.shape[2:] != (resolution, resolution): image_pix = F.interpolate(image_pix, size=(resolution, resolution), mode='bicubic', align_corners=True)
    image_pix = image_pix*2-1
    image_pix = image_pix.to(device, dtype=sd3.dtype)
    image_source = sd3.encode(image_pix)

    image = image_source.to(device)

    with torch.no_grad():
        null_prompt_emb, null_pooled_emb = sd3.encode_prompt([""], 1)
        z_target = image.clone()

    sd3.text_enc_1.to('cpu')
    sd3.text_enc_2.to('cpu')
    sd3.text_enc_3.to('cpu')
    del sd3.text_enc_1, sd3.text_enc_2, sd3.text_enc_3
    print("Tokenizers and text encoders removed to save memory.")



    reward_all = []
    method = METHOD_NAME
    # ==================== Gradient Ascent ==================== #
    if method == "gradient_ascent":
        pbar = tqdm(range(config.n_iter))
        for i in pbar:
            reward_grads, reward_values = all_grad_rewards(sd3, image, prompts)
            image = image + reward_grads * config.lr
            pbar.set_description(f"Reward: {reward_values.item():.4f}")
            if (i + 1) % 10 == 0:
                save_latent(sd3, image, f'./edit_output.png', source_img=image_source)



    # ==================== DPS, FreeDoM, TFG ==================== #
    elif "inversion" in method:
        pipe = sd3
        limit_t = int(pipe.NFE*config.depth)
        zT, _ = pipe.inversion_uncond(z_target, (null_prompt_emb, null_pooled_emb), limit_t)

        zt = zT.to(pipe.device)
        all_timesteps = sd3.scheduler.timesteps
        all_ts = (all_timesteps / sd3.scheduler.config.num_train_timesteps).to(device=pipe.device, dtype=sd3.dtype)
        all_t_prevs = torch.cat([all_ts[1:], torch.zeros_like(all_ts[:1])], dim=0)

        pbar = tqdm(all_ts)
        with torch.no_grad():
            for i, t in enumerate(pbar):
                if i < all_ts.shape[0]-limit_t:    continue
                def get_eps(xt, ii):
                    return pipe.get_v_prediction(xt, torch.tensor([ii], device=pipe.device), None, None, null_prompt_emb, null_pooled_emb, cfg_scale=0.0)[0]
                def rewardfn(x0):
                    reward_grads, reward_values = all_grad_rewards(sd3, x0, prompts)
                    return reward_values
                with torch.enable_grad():
                    zt, (rv, _) = TFG.guide_step(zt, i, get_eps, all_ts, all_t_prevs, 0., rewardfn=rewardfn)
                    pbar.set_description(f"Reward: {rv:.4f}")
        z0 = zt
        save_latent(sd3, z0, f'./edit_output.png', source_img=image_source)



    # ==================== Ours ==================== #
    elif method == "itoc":
        pipe = sd3
        limit_t = int(pipe.NFE*config.depth)
        all_timesteps = sd3.scheduler.timesteps[-limit_t:]
        all_ts = (all_timesteps / sd3.scheduler.config.num_train_timesteps).to(device=pipe.device, dtype=sd3.dtype)[None, :, None, None, None]
        all_dts = torch.cat([all_ts[:, :-1] - all_ts[:, 1:],  all_ts[:, -1:]], dim=1)
        pbar = tqdm(range(config.n_iter))

        if DETERMINISTIC:   _, z_source_traj = sd3.inversion_uncond(image, (null_prompt_emb, null_pooled_emb), limit_t)
        else:   _, z_source_traj, _ = sd3.get_q_traj(image, limit_t)

        z_source_traj = torch.stack(z_source_traj[:limit_t + 1][::-1], dim=1).detach().to(sd3.dtype)
        us = torch.zeros_like(z_source_traj[:, 1:])
        us_32 = us.clone().to(dtype=torch.float32)
        us_32.requires_grad = True

        optimizer = torch.optim.Adam(params=[us_32], lr=config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: max(0.9 ** epoch, 0.5))

        noises = None
        b_grad_all = None
        with torch.no_grad():
            for i in pbar:
                z_source_traj_nograd = z_source_traj.detach().to(dtype=sd3.dtype).requires_grad_(True)
                with torch.enable_grad():
                    rg, rv = all_grad_rewards(sd3, z_source_traj_nograd[:, -1], prompts)
                    reward_all.append(rv.item())
                    adjoint_states, reward_values, _, all_prev_pred_init = sd3.compute_adjoints(z_source_traj_nograd, all_timesteps.unsqueeze(-1),
                                                                                              null_prompt_emb, null_pooled_emb,
                                                                                              prompts, initial_grad=(rg, rv), reward_multiplier=config.reward_multiplier,
                                                                                              b_grad = None)
                if noises is None:
                    if DETERMINISTIC:   noises = torch.zeros_like(z_source_traj_nograd[:, 1:])
                    else:   noises = z_source_traj_nograd[:, 1:] - all_prev_pred_init
                if DETERMINISTIC:   u_target = -adjoint_states[:, :-1] * all_dts #* (2*all_ts/(1-all_ts)) * 1.0
                else:   u_target = -adjoint_states[:, :-1] * all_dts #* (2*all_ts/(1-all_ts)) * 1.0

                with torch.enable_grad():
                    optimizer.zero_grad()
                    loss = ((us_32 - u_target.to(torch.float32).detach()) ** 2).mean()
                    (loss*1.).backward()
                    optimizer.step()
                    scheduler.step()
                #z_source_traj = torch.cat([z_source_traj[:, :1], (z_source_traj_nograd[:, 1:] + torch.cumsum(us - us_old, dim=1))], dim=1).detach()

                z_source_traj_new = [z_source_traj[:, 0]]
                b_grad_all = []
                allt = all_timesteps.unsqueeze(-1)
                alldt = all_dts.squeeze()
                for k in range(all_timesteps.shape[0]):
                    dt = alldt[k]

                    v, _ = sd3.get_v_prediction(z_source_traj_new[-1], allt[k], None, None, null_prompt_emb, null_pooled_emb, cfg_scale=0.0)
                    #prev_sample_init = z_source_traj_new[-1] - v * dt
                    prev_sample_init = sd3.calculate_term(v, allt[k], z_source_traj_new[-1], alldt[k]*sd3.scheduler.config.num_train_timesteps)
                    '''
                    with torch.enable_grad():
                        x = z_source_traj_new[-1]
                        x = x.requires_grad_(True)  # bchw
                        v, _ = sd3.get_v_prediction(x.to(sd3.dtype), allt[k], None, None, null_prompt_emb, null_pooled_emb, cfg_scale=0.0)
                        prev_sample_init = x - v * dt
                        sum_inner_prod = torch.sum((prev_sample_init - x), dim=[0, 1, 2, 3])
                        b_grad = torch.autograd.grad(  # bchw
                            sum_inner_prod,
                            x,
                        )[0]
                        print('bb', k, torch.isnan(b_grad).any(), torch.isnan(us_32[:, k]).any())
                    '''
                    next = (prev_sample_init + noises[:, k] + us_32[:, k]*(1 if k<100 else 0)).detach().to(sd3.dtype)
                    z_source_traj_new.append(next)
                    #b_grad_all.append(b_grad)
                z_source_traj = torch.stack(z_source_traj_new, dim=1).detach()

                pbar.set_description(f"Reward: {rv.item():.4f}, Loss: {loss.item():.4f}")
                save_latent(sd3, z_source_traj[:, -1], f'./edit_output.png', source_img=image_source)
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.plot(reward_all)
        plt.savefig(f'./edit_reward.png')


if __name__ == "__main__":
    main()
