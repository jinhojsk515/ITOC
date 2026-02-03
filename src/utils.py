import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt



def save_latent(pipe, latent, filename, source_img=None):
    image_pix = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image_pix = (image_pix / 2 + 0.5).clamp(0, 1)
    save_image(image_pix[0], filename, normalize=True, value_range=(0, 1))
    if source_img is not None:
        diff = ((latent - source_img)[0]**2).sum(dim=0).sqrt().clamp(0., 1.)
        plt.matshow(diff.detach().cpu().numpy())
        #plt.colorbar()
        plt.savefig(filename.replace('.png', '_diff.png'))


def all_grad_rewards(trainer, image, prompts, reward_name, method_name):
    metrics, metrics_w = [reward_name], [1.0]

    all_rg, all_rv = [], []
    for w, metric in zip(metrics_w, metrics):
        trainer.reward_func = metric

        if 'inversion' in method_name:    rg, rv = trainer.grad_rewards(image, prompts, get_grad=False)     # DPS, FreeDoM, TFG calculates gradient on its own(see tfg.py)
        else:   rg, rv = trainer.grad_rewards(image, prompts)

        all_rg.append(w * rg)
        all_rv.append(w * rv)
    all_rg = torch.stack(all_rg, dim=0).sum(dim=0)
    all_rv = torch.stack(all_rv, dim=0).sum(dim=0)
    return all_rg, all_rv