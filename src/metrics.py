import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2
from contextlib import nullcontext

from image_reward_utils import rm_load
from aesthetic_scorer import resize, center_crop, normalize
from transformers import AutoProcessor, AutoModel

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "ImageNet1k_classifier": None,
    "Gram_Diff": None,
}



def do_imagenet1k_classifier(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageNet1k_classifier"] is None:
        from aesthetic_scorer import ImageNet1kClassifier
        classifier = ImageNet1kClassifier().cuda()
        REWARDS_DICT["ImageNet1k_classifier"] = classifier

    scores = REWARDS_DICT["ImageNet1k_classifier"](images*255, prompts)
    return scores


def do_GramDiff(*, images, target_images=None):
    global REWARDS_DICT
    if REWARDS_DICT["Gram_Diff"] is None:
        from gram_matrix import StyleCLIP
        get_gramdiff = StyleCLIP('openai/clip-vit-base-patch16', device='cuda')
        REWARDS_DICT["Gram_Diff"] = get_gramdiff

    get_gramdiff = REWARDS_DICT["Gram_Diff"]
    images = torch.nn.functional.interpolate(images*255, size=get_gramdiff.image_size, mode='bicubic')/255.0       # 0~1
    images = get_gramdiff.transforms(images)
    scores = get_gramdiff(images, target_images)
    return scores


def do_image_reward(*, images, prompts, use_no_grad=True, use_score_from_prompt_batched=False):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")
    context = torch.no_grad() if use_no_grad else nullcontext()
    with context:
        if use_score_from_prompt_batched:
            image_reward_result = REWARDS_DICT["ImageReward"].score_from_prompt_batched(prompts, images)
        else:
            image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
    return image_reward_result


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )
        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)
    def score(self, prompt, pil_image, return_feature=False, i2i=False):
        # text encode
        if i2i:
            image2 = resize(prompt, {"shortest_edge": 224})
            image2 = center_crop(image2, {"height": 224, "width": 224})
            image2 = normalize(image2, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).unsqueeze(0)
            txt_features = F.normalize(self.clip_model.encode_image(image2))
        else:
            text = clip.tokenize(prompt, truncate=True).to(self.device)
            txt_features = F.normalize(self.clip_model.encode_text(text))
        # image encode
        pil_image = resize(pil_image, {"shortest_edge": 224})
        pil_image = center_crop(pil_image, {"height":224, "width":224})
        pil_image = normalize(pil_image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).unsqueeze(0)
        image_features = F.normalize(self.clip_model.encode_image(pil_image))
        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )
        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}
        return rewards.squeeze(-1)
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    clip_result = torch.cat([REWARDS_DICT["Clip-Score"].score(prompt, images[i]) for i, prompt in enumerate(prompts)], dim=0)
    return clip_result


def reward_function(x, prompt, model, reward_func="ImageReward", use_no_grad=False, use_score_from_prompt_batched=True, verbose=False):
    if model is not None:   imagestensor = model.image_processor.postprocess(x, output_type="pt")   # -1~1 -> 0~1
    elif x.min() < 0.:      imagestensor = (x + 1.) / 2.
    else:                   imagestensor = x
    imagesx = [image for image in imagestensor]

    if reward_func == "ImageReward":
        rewards = do_image_reward(
            images=imagesx, 
            prompts=prompt, 
            use_no_grad=use_no_grad, 
            use_score_from_prompt_batched=use_score_from_prompt_batched,
        )
    elif reward_func == "Clip_Score":
        rewards = do_clip_score(images=imagesx, prompts=prompt)
    elif reward_func == "ImageNet1k_classifier":
        rewards = do_imagenet1k_classifier(images=imagestensor, prompts=prompt)
    elif reward_func == "Gram_Diff":
        rewards = do_GramDiff(images=imagestensor, target_images=prompt)
    else:
        raise ValueError(f"Unknown metric: {reward_func}")

    if use_no_grad:
        return torch.tensor(rewards).to(x.device)
    else:
        if verbose:
            print(f'rewards.requires_grad in reward_function: {rewards.requires_grad}')
        return rewards
