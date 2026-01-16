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
import glob
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
import random
from attention import ConcatFusion


class Loss_Stable_Diffusion:
    
    def __init__(self):
        self.config_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml"
        self.ckpt_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/Text_location_215/2025-01-03T12-29-27_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=11-step=299.ckpt"
        self.device = "cuda:0"
        self.global_model = self.load_model_from_config(self.config_path, self.ckpt_path, self.device)
        self.model = copy.deepcopy(self.global_model)
        self.clip = None
        self.sampler = None 
        self.seed = 0
        

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
        # writer.add_scalar('loss_terms/fruit_blip_loss', out['fruit_blip_loss'], out['iteration'])
        # writer.add_scalar('loss_terms/mat_blip_loss', out['mat_blip_loss'], out['iteration'])
        # writer.add_scalar('loss_terms/color_blip_loss', out['color_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/diff_loss', terms['loss'].mean().float(), out['iteration'])
        return terms['loss'].mean()
    
    def visualize(self, out: Dict[str, torch.Tensor], data) -> Dict[str, Any]:


        # sampler = DDIMSampler(model)
        embeddings = out['embeddings']
        # negative_embeddings = out.get('negative_embeddings', None)
        batch_size = embeddings.shape[0]
        #*加入V*
        emb_opt_text = torch.load('/home/hyl/yujia/test_few_imagic/v6.jpg/mask_2/emb_opt_text.pt') 
        emb_opt_text = emb_opt_text.to(self.device)
        emb_opt_text = emb_opt_text.repeat(batch_size, 1, 1)        
        
        # new_embeddings = emb_opt_text
        # new_embeddings = embeddings
        new_embeddings = 0.7*emb_opt_text + 0.3*embeddings
        clip = FrozenCLIPEmbedder(device="cpu")
        
        text_list_cpu = [""] *batch_size
        uc = clip.encode(text_list_cpu)
        uc = uc.to(embeddings.device)

        stageI_generations, _ = self.model.sample_log(cond=None,text = new_embeddings, batch_size=batch_size, ddim=True,
                                      ddim_steps=200,unconditional_guidance_scale=7.5,
                                                         unconditional_conditioning=uc,                    
                                                         eta=1.0,multi = False)
        img = self.decode_to_im(stageI_generations,n_samples = batch_size,nrow = 1)
        return {'image': img}

    def visualize_loop(self, out: Dict[str, torch.Tensor], data,opt_embs_file_path) -> Dict[str, Any]:
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 假设 out['embeddings'] 的 batch_size = 1，如果不是1，请根据需要进行修改
        embeddings = out['embeddings']  # shape: [batch, emb_dim_1, emb_dim_2 ...]
        # 若原始batch>1，想重复100次整批
        # 这里示例假设batch=1：
        embeddings = embeddings.repeat(100, 1, 1)  # 在batch维度重复100次，使得batch_size=100
        batch_size = embeddings.shape[0]

        # 定义文件路径
        save_root = "/home/hyl/yujia/A_few_shot/paper_data_1_27/class_3/fake_data/ours_no_prototype"
        # opt_embs_file_path = '/home/hyl/yujia/A_few_shot/evaluation/eval_fake_data/opt_embs/210527XJYk_emb_opt.pt'
        opt_embs_file_path = opt_embs_file_path 
        

        # 构建 clip 和 unconditional embeddings
        clip = FrozenCLIPEmbedder(device="cpu")
        text_list_cpu = [""] * batch_size
        uc = clip.encode(text_list_cpu).to(embeddings.device)

        # 从 opt_embs_file_path 中提取文件名前缀并去掉 "_emb_opt"
        filename = os.path.basename(opt_embs_file_path)  # 例如 "0120ZJHb_emb_opt.pt"
        prefix = os.path.splitext(filename)[0]           # "0120ZJHb_emb_opt"
        prefix = prefix.replace("_emb_opt", "")          # "0120ZJHb"
        save_dir = os.path.join(save_root, prefix)
        os.makedirs(save_dir, exist_ok=True)

        # 加载 emb_opt_text
        # emb_opt_text = torch.load(opt_embs_file_path, map_location=self.device)
        # # 假设 emb_opt_text shape: [1, emb_dim1, emb_dim2], 与 embeddings 相匹配
        # emb_opt_text = emb_opt_text.repeat(batch_size, 1, 1)

        # 混合 embeddings

        # aggregator = ConcatFusion().to(self.device)
        
        
        # new_embeddings = aggregator(emb_opt_text,embeddings)
        new_embeddings = embeddings

        # 使用新 embeddings 生成图像
        stageI_generations, _ = self.model.sample_log(
            cond=None,
            text=new_embeddings,
            batch_size=batch_size,
            ddim=True,
            ddim_steps=200,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=1.0,
            multi=False
        )

        # 保存生成的100张图像
        for i in range(batch_size):
            sample = stageI_generations[i].unsqueeze(0)  # [1, C, H, W]
            img = self.decode_to_im(sample, n_samples=1, nrow=1)
            # 如果decode_to_im返回list，则取img[0]
            if isinstance(img, list):
                img = img[0]

            # 将tensor转换为PIL图片（如果需要）
            if isinstance(img, torch.Tensor):
                arr = (img.detach().cpu().numpy().transpose(1,2,0)*255).astype("uint8")
                img = Image.fromarray(arr)

            save_path = os.path.join(save_dir, f'generated_{i+1}.png')
            img.save(save_path)

        return {}

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