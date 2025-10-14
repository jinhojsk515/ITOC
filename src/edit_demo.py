import torch
import argparse
import time
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tfg import TFGGuidance
from sd15 import StableDiffusion



config = argparse.ArgumentParser()
# task(method, reward)
config.add_argument("--method_name", type=str, default=None, choices=["gradient_ascent", "itoc", "inversion_tfg", "inversion_dps", "inversion_freedom"])
config.add_argument("--reward_name", type=str, default=None, choices=['ImageReward', 'ImageNet1k_classifier', 'Clip_Score', 'Gram_Diff'])
# input
config.add_argument("--image_path", type=str, default=None)
config.add_argument("--reward_prompt", type=str, default="colorful painting, river flowing grass field with flowers.")
config.add_argument("--style_image_path", type=str, default="style_ref.png")
config.add_argument("--reward_class", type=int, default=306)
# hyperparameters
config.add_argument("--deterministic", type=bool, default=True)             # ITOC
config.add_argument("--n_iter", type=int, default=20)                       # GA, ITOC
config.add_argument("--reward_multiplier", type=float, default=200.0)       # ITOC
config.add_argument("--depth", type=float, default=0.5)                     # DPS, FreeDoM, TFG, ITOC
config.add_argument("--lr", type=float, default=5e-3)                       # GA, ITOC
config.add_argument("--tfg_rho", type=float, default=1.0)                   # DPS, FreeDoM, TFG
config.add_argument("--tfg_mu", type=float, default=0.0)                    # TFG
config = config.parse_args()

config.seed = 42
METHOD_NAME = config.method_name
REWARD_NAME = config.reward_name
DETERMINISTIC = config.deterministic

def save_latent(pipe, latent, filename, source_img=None):
    image_pix = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image_pix = (image_pix / 2 + 0.5).clamp(0, 1)
    save_image(image_pix[0], filename, normalize=True, value_range=(0, 1))
    if source_img is not None:
        diff = ((latent - source_img)[0]**2).sum(dim=0).sqrt().clamp(0., 1.)
        plt.matshow(diff.detach().cpu().numpy())
        #plt.colorbar()
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
    TFG = TFGGuidance(args=config)

    device = config.device
    sd15 = StableDiffusion(pipe_dtype=torch.float16, deterministic=DETERMINISTIC, device=device)


    # ==================== Load images ==================== #
    if REWARD_NAME == 'ImageNet1k_classifier':
        prompts = config.reward_class
    elif REWARD_NAME == 'Gram_Diff':
        prompts = config.style_image_path
    else:
        prompts = [config.reward_prompt]
    with torch.no_grad():
        image_pix = read_image(config.image_path).unsqueeze(0).float() / 255.0
        image_pix = image_pix*2-1
        image_pix = image_pix.to(device, dtype=sd15.dtype)
        image_source = sd15.soc_pipeline.vae.encode(image_pix)['latent_dist'].mean * 0.18215
        image = image_source.to(device)

        embedding_null = sd15.get_text_embeddings("")
        embedding_source = torch.stack([embedding_null, embedding_null], dim=1)
    sd15.text_encoder.to('cpu')
    sd15.vae.encoder.to('cpu')
    del sd15.vae.encoder
    del sd15.text_encoder


    reward_all = []
    method = METHOD_NAME
    # ==================== Gradient Ascent ==================== #
    if method == "gradient_ascent":
        st = time.time()
        pbar = tqdm(range(config.n_iter))
        for i in pbar:
            reward_grads, reward_values = all_grad_rewards(sd15, image, prompts)
            image = image + reward_grads * config.lr
            pbar.set_description(f"Reward: {reward_values.item():.4f}")
            if (i + 1) % 10 == 0:
                save_latent(sd15.soc_pipeline, image, f'./edit_output.png', source_img=image_source)
        print("time:", time.time() - st)



    # ==================== DPS, FreeDoM, TFG ==================== #
    elif "inversion" in method:
        pipe = sd15
        noise_limit = int(1000 * config.depth)
        zT, _ = pipe.inversion(image, embedding_source, noise_limit=noise_limit)

        zt = zT.to(pipe.device)
        skip = pipe.scheduler.alphas_cumprod.shape[0] // len(pipe.scheduler.timesteps)
        alpha = lambda tt: pipe.scheduler.alphas_cumprod[tt] if tt >= 0 else pipe.scheduler.final_alpha_cumprod

        all_t = torch.tensor(list(range(pipe.scheduler.alphas_cumprod.shape[0] - 2, -1, -skip)), device=pipe.device, dtype=torch.long)
        all_alpha_t = torch.tensor([alpha(t) for t in all_t], device=pipe.device, dtype=torch.float32)
        all_alpha_t_prev = torch.tensor([alpha(t - skip) for t in all_t], device=pipe.device, dtype=torch.float32)

        pbar = tqdm(range(pipe.scheduler.alphas_cumprod.shape[0] - 2, -1, -skip))
        with torch.no_grad():
            for i, t in enumerate(pbar):
                if t >= noise_limit:    continue

                def get_eps(xt, ii):
                    return pipe.get_eps_prediction(xt, all_t[ii:ii + 1].to(pipe.device), embedding_source)[1][1]

                def rewardfn(x0):
                    reward_grads, reward_values = all_grad_rewards(sd15, x0, prompts)
                    return reward_values

                with torch.enable_grad():
                    zt, (rv, _) = TFG.guide_step(zt, i, get_eps, all_t, all_alpha_t, all_alpha_t_prev, 0., rewardfn=rewardfn)
                    pbar.set_description(f"Reward: {rv:.4f}")
        z0 = zt
        save_latent(sd15.soc_pipeline, z0, f'./edit_output.png', source_img=image_source)


    # ==================== Ours ==================== #
    elif method == "itoc":
        pipe = sd15
        alpha = lambda tt: pipe.scheduler.alphas_cumprod[tt] if tt >= 0 else pipe.scheduler.final_alpha_cumprod
        skip = pipe.scheduler.alphas_cumprod.shape[0] // len(sd15.scheduler.timesteps)
        limit = int(1000 * config.depth)
        limit_t = limit // skip

        all_t = torch.tensor(list(range(pipe.scheduler.alphas_cumprod.shape[0] - 2, -1, -skip)), device=pipe.device, dtype=torch.int)[-limit_t:]
        all_alpha_t = torch.tensor([alpha(t) for t in all_t], device=pipe.device, dtype=pipe.dtype)[None, :, None, None, None]
        all_alpha_t_prev = torch.tensor([alpha(t - skip) for t in all_t], device=pipe.device, dtype=pipe.dtype)[None, :, None, None, None]

        pbar = tqdm(range(config.n_iter))

        with torch.no_grad():
            if DETERMINISTIC:
                zT, z_source_traj = sd15.inversion(image, embedding_source, noise_limit=limit)
            else:
                zT = image.clone().to(pipe.device)
                z_source_traj = [zT]
                pbarr = reversed(range(pipe.scheduler.alphas_cumprod.shape[0] - 2, -1, -skip))
                for i, t in enumerate(pbarr):  # 0,1,2,...     19,39,59,...
                    if t >= limit:
                        z_source_traj.append(zT)
                        continue
                    at, at_prev = alpha(t), alpha(t - skip)
                    zT = (at / at_prev).sqrt() * zT + (1 - at / at_prev).sqrt() * torch.randn_like(zT)
                    z_source_traj.append(zT)

        z_source_traj = torch.stack(z_source_traj[:limit_t + 1][::-1], dim=1).detach()

        us = torch.zeros(z_source_traj[:, 1:].shape).to(pipe.device, dtype=pipe.dtype)
        us.requires_grad = True
        optimizer = torch.optim.Adam(params=[us], lr=config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: max(0.95 ** epoch, 0.5))

        noises = None
        b_grad_all = None
        with torch.no_grad():
            for i in pbar:
                z_source_traj_nograd = z_source_traj.detach()#.requires_grad_(True)
                with torch.enable_grad():
                    rg, rv = all_grad_rewards(sd15, z_source_traj.detach()[:, -1], prompts)
                    reward_all.append(rv.item())
                    adjoint_states, reward_values, all_noise_pred_init, all_prev_pred_init = sd15.compute_adjoints(z_source_traj_nograd, all_t, embedding_source[0, 1:],
                                                                                                                   prompts, initial_grad=(rg, rv),
                                                                                                                   reward_multiplier=config.reward_multiplier, b_grad = b_grad_all)
                if noises is None:
                    if DETERMINISTIC:   noises = torch.zeros_like(z_source_traj[:, 1:])
                    else:   noises = z_source_traj[:, 1:] - all_prev_pred_init

                if DETERMINISTIC:
                    u_target = -adjoint_states[:, :-1] / pipe.NFE
                else:   u_target = -adjoint_states[:, :-1] * ((all_alpha_t_prev - all_alpha_t)/all_alpha_t)

                with torch.enable_grad():
                    loss = ((us-u_target.detach())**2).mean()
                    optimizer.zero_grad()
                    (loss*1.).backward()
                    optimizer.step()
                    scheduler.step()

                z_source_traj_new = [z_source_traj[:, 0]]
                b_grad_all = []
                for k in range(all_t.shape[0]):
                    with torch.enable_grad():
                        x = z_source_traj_new[-1]
                        x = x.requires_grad_(True)  # bchw
                        noise_pred = sd15.unet(z_source_traj_new[-1], all_t[k], encoder_hidden_states=embedding_source[0, 1:], return_dict=False, )[0]
                        prev_sample_init = sd15.calculate_term(noise_pred, all_t[k], z_source_traj_new[-1])
                        sum_inner_prod = torch.sum((prev_sample_init - x), dim=[0, 1, 2, 3])
                        b_grad = torch.autograd.grad(sum_inner_prod, x, )[0]

                    next = (prev_sample_init + noises[:, k] + 1 * us[:, k]).detach()
                    z_source_traj_new.append(next)
                    b_grad_all.append(b_grad)
                z_source_traj = torch.stack(z_source_traj_new, dim=1).detach()

                with torch.enable_grad():  rg, rv = all_grad_rewards(sd15, z_source_traj[:, -1], prompts)
                pbar.set_description(f"Reward: {rv.item():.4f}, Loss: {loss.item():.4f}")
                save_latent(sd15.soc_pipeline, z_source_traj[:, -1], f'./edit_output.png', source_img=image_source)
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.plot(reward_all)
        plt.savefig(f'./edit_reward.png')

if __name__ == "__main__":
    main()
