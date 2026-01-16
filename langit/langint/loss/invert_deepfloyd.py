from typing import Union, List, Optional, Dict, Any
# import torch
# import os
from torchvision.transforms import ToPILImage
# import argparse
import copy
import os
from pathlib import Path
# #################################
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

class Loss_Stable_Diffusion:
    
    def __init__(self):
        self.config_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml"
        self.ckpt_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/Text_location_215/2025-01-03T12-29-27_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=11-step=299.ckpt"
        self.device = "cuda:0"
        self.global_model = self.load_model_from_config(self.config_path, self.ckpt_path, self.device)
        self.model = copy.deepcopy(self.global_model)
        self.model.eval()
        self.clip = None
        self.sampler = None 
        self.seed = 42

    def load_img(self, path, target_size=256):
        """Load an image, resize and output -1..1"""
        if isinstance(path, torch.Tensor):
            if path.ndimension() == 4:
                to_pil = ToPILImage()
                image = to_pil(path[0]).convert("RGB")  # Convert first image in batch
            else:
                # If it's a single image (3D tensor), convert it directly
                to_pil = ToPILImage()
                image = to_pil(path).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")

        tform = transforms.Compose([
            # transforms.Resize(target_size),
            # transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        image = tform(image)
        return 2. * image - 1.
    
    def decode_to_im(self, samples, n_samples=1, nrow=1):
        """Decode a latent and return PIL image"""
        samples = self.model.decode_first_stage(samples)
        ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(
            ims.cpu().numpy(),
            '(n1 n2) c h w -> (n1 h) (n2 w) c',
            n1=n_samples // nrow,
            n2=nrow)
        return Image.fromarray(x_sample.astype(np.uint8))
    def load_model_from_config(self, config, ckpt, device="cpu", verbose=False):
        """Loads a model from config and a ckpt
        if config is a path will use omegaconf to load
        """
        if isinstance(config, (str, Path)):
            config = OmegaConf.load(config)

        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        model.cond_stage_model.device = device
        return model
    def __call__(self, out: Dict[str, torch.Tensor], data, writer=None) -> Dict[str, Any]:
        """定义训练时的优化逻辑：模型预测 -> 计算 loss 并记录。

        """
        embeddings = out['embeddings']
        terms = dict()
        batch_size = embeddings.shape[0]
        x_start = data['image']   #*(batch,3,256,256)
        # init_image = self.load_img(x_start).to("cuda:0") # [1, 3, 256, 256]
        # init_image = init_image.unsqueeze(0)
        gaussian_distribution = self.model.encode_first_stage(x_start)
        init_latent = self.model.get_first_stage_encoding(gaussian_distribution)
        
        torch.manual_seed(self.seed)

        
        noise = torch.randn_like(init_latent)
        target = noise  #* (batch,4,32,32)
        # t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device='cuda', dtype=torch.long)
        t_enc = torch.randint(1000, (batch_size,), device=self.device)

        z = self.model.q_sample(init_latent, t_enc, noise=noise)

        

        pred_noise = self.model.apply_model(z, t_enc, None,embeddings)

        assert pred_noise.shape == target.shape
        assert not torch.isnan(pred_noise).any() and not torch.isnan(target).any(), (torch.isnan(pred_noise).any(), torch.isnan(target).any())
        
        
        criteria = torch.nn.MSELoss()

        loss = criteria(pred_noise, target)
        terms['loss'] = loss
        writer.add_scalar('loss_terms/fruit_blip_loss', out['fruit_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/mat_blip_loss', out['mat_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/color_blip_loss', out['color_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/location_blip_loss', out['location_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/diff_loss', terms['loss'].mean().float(), out['iteration'])
        
        return terms['loss'].mean() + out['fruit_blip_loss'] + out['mat_blip_loss'] + out['color_blip_loss'] + out['location_blip_loss']
    
    def visualize(self, out: Dict[str, torch.Tensor], data) -> Dict[str, Any]:
        

        # sampler = DDIMSampler(model)
        embeddings = out['embeddings']
        # negative_embeddings = out.get('negative_embeddings', None)

        batch_size = embeddings.shape[0]

        clip = FrozenCLIPEmbedder(device="cpu")
        
        text_list_cpu = [""] *batch_size
        uc = clip.encode(text_list_cpu)
        uc = uc.to(embeddings.device)

        stageI_generations, _ = self.model.sample_log(cond=None,text = embeddings, batch_size=batch_size, ddim=True,
                                      ddim_steps=50,unconditional_guidance_scale=7.5,
                                                         unconditional_conditioning=uc,                    
                                                         eta=1.0,multi = False)
        img = self.decode_to_im(stageI_generations,n_samples = batch_size,nrow = 1)
        return {'image': img}

    # def dream(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = None):

    #     if_I = self.if_I
    #     t5 = self.t5
    #     if isinstance(prompt, str):
    #         prompt = [prompt]
    #     t5_embs = t5.get_text_embeddings(prompt)

    #     if_I_kwargs = dict()
    #     if_I_kwargs['t5_embs'] = t5_embs

    #     if_I_kwargs['guidance_scale'] = 7.0
    #     if_I_kwargs['sample_timestep_respacing'] = 'smart100'
    #     if_I_kwargs['aspect_ratio'] = '1:1'
    #     if_I_kwargs['progress'] = False

    #     if negative_prompt is not None:
    #         if isinstance(negative_prompt, str):
    #             negative_prompt = [negative_prompt]
    #         negative_t5_embs = t5.get_text_embeddings(negative_prompt)
    #         if_I_kwargs['negative_t5_embs'] = negative_t5_embs

    #     stageI_generations, _ = if_I.embeddings_to_image(**if_I_kwargs)
    #     image: List[Image.Image] = if_I.to_images(stageI_generations, disable_watermark=True)
    #     return image