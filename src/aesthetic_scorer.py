# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import List, Union, Optional, Dict
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.models import resnet50
import dill


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load("./models/sac+logos+ava1-l14-linearMSE.pth")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        #inputs = self.processor(images=images, return_tensors="pt")
        #inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        inputs = {'pixel_values': my_processor(images)}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

robust = True
class ImageNet1kClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if robust:
            self.model = resnet50(pretrained=True)
            checkpoint = torch.load("./models/imagenet_l2_3_0.pt", pickle_module=dill)
            sd = checkpoint['model']
            sd = {k[len('module.'):]: v for k, v in sd.items()}
            new_sd = {}
            for k, v in sd.items():
                if k.startswith('model.'):
                    new_sd[k[len('model.'):]] = v
                elif 'attacker.' in k:
                    continue
                else:
                    new_sd[k] = v
            msg = self.model.load_state_dict(new_sd, strict=False)
        else:
            self.model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-large-imagenet1k-1-layer')
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large-imagenet1k-1-layer')
        self.eval()

    def __call__(self, images, label, return_p=False):
        if type(label) is dict: prompts = label["text"]
        if robust:
            logits = self.model(my_processor(images, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]))
        else:
            inputs = {'pixel_values': my_processor(images, image_mean=self.processor.image_mean, image_std=self.processor.image_std)}
            outputs = self.model(**inputs)
            logits = outputs.logits
        if return_p:
            return torch.tensor(1.) if logits.argmax(dim=1).item() == label else torch.tensor(0.)
            #return F.softmax(logits, dim=1)[:, label]
        else:   return logits[:, label]

def my_processor(images,
                 size = {"shortest_edge": 224},
                 crop_size = {"height":224, "width":224},
                 rescale_factor = 1/255.,
                 image_mean = [0.48145466, 0.4578275, 0.40821073],
                 image_std = [0.26862954, 0.26130258, 0.27577711],
                 ):
    #images = list(torch.unbind(images, dim=0))
    images = [image for image in images]
    output_images = []
    for image in images:
        image = resize(image, size)
        image = center_crop(image, crop_size)
        image = rescale(image, rescale_factor)
        image = normalize(image, image_mean, image_std)
        output_images.append(image)
    return torch.stack(output_images)

def resize(image: torch.Tensor, size: Dict[str, int], interpolation: str = "bilinear") -> torch.Tensor:
    h, w = image.shape[-2:]
    if "shortest_edge" in size:
        scale = size["shortest_edge"] / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    else:
        new_h, new_w = size["height"], size["width"]
    return torch.nn.functional.interpolate(image.float().unsqueeze(0), size=(new_h, new_w), mode=interpolation, align_corners=False).squeeze(0)

def center_crop(image: torch.Tensor, size: Dict[str, int]) -> torch.Tensor:
    crop_h, crop_w = size["height"], size["width"]
    h, w = image.shape[-2:]
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return image[..., top:top+crop_h, left:left+crop_w]

def rescale(image: torch.Tensor, factor: float) -> torch.Tensor:
    return image * factor

def normalize(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor
