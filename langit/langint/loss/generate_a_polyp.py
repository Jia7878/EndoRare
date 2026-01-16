from typing import Union, List, Optional, Dict, Any
from torchvision.transforms import ToPILImage
import copy
import os
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from polypdiffusion.ldm.modules.encoders.modules import FrozenCLIPEmbedder
import random

class YourModelClass:
    def __init__(self):
        self.config_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml"
        self.ckpt_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/resume_sd/2024-11-20T21-18-03_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=49-step=699.ckpt"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 加载模型配置和检查点
        self.global_model = self.load_model_from_config(self.config_path, self.ckpt_path, self.device)
        self.model = self.global_model  # 如果不需要修改，直接使用 global_model
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 初始化 CLIP 模型
        self.clip = FrozenCLIPEmbedder(device="cpu")
        
        # 初始化 DDIM 采样器（如果需要）
        self.sampler = None 
        
        # 设置随机种子
        self.seed = 0

    def set_seed(self, seed: int = 0):
        """
        设置随机种子以确保结果的可重复性。
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_img(self, path, target_size=256):
        """加载图像，调整大小并输出范围为 -1..1 的张量"""
        if isinstance(path, torch.Tensor):
            if path.ndimension() == 4:
                to_pil = ToPILImage()
                image = to_pil(path[0]).convert("RGB")  # 转换第一张图像
            else:
                # 如果是单张图像（3D 张量），直接转换
                to_pil = ToPILImage()
                image = to_pil(path).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")

        tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        image = tform(image)
        return 2. * image - 1.

    def decode_to_im(self, samples, n_samples=1, nrow=1):
        """将潜在表示解码为 PIL 图像"""
        samples = self.model.decode_first_stage(samples)
        ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(
            ims.cpu().numpy(),
            '(n1 n2) c h w -> (n1 h) (n2 w) c',
            n1=n_samples // nrow,
            n2=nrow)
        return Image.fromarray(x_sample.astype(np.uint8))

    def load_model_from_config(self, config, ckpt, device="cpu", verbose=False):
        """从配置和检查点加载模型"""
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

    def test_clip_text_encoder(self, texts):
        # 检查是否有可用的 GPU，否则使用 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 加载 CLIP 的分词器和文本模型
        tokenizer = self.clip.tokenizer
        model = self.clip.transformer
        model.to(device)
        model.eval()  # 设置模型为评估模式

        # 示例文本
        max_length = 77
        # 分词并转换为模型输入所需的张量
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        print(inputs['input_ids'])
        # 禁用梯度计算（节省内存和计算资源）
        with torch.no_grad():
            # 获取文本嵌入
            outputs = model(**inputs)
            # 通常使用最后一个隐藏层的平均作为句子的表示
            embeddings = outputs.last_hidden_state

        # 打印每个文本的嵌入向量
        for text, embedding in zip(texts, embeddings):
            print(f"文本: {text}")
            print(f"嵌入向量: {embedding.shape}\n")
        
        return embeddings

    def test_clip_embedding_encoder(self, texts):
        # 检查是否有可用的 GPU，否则使用 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 加载 CLIP 的分词器和文本模型
        tokenizer = CLIPTokenizer.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        model = CLIPTextModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        model.to(device)
        model.eval()  # 设置模型为评估模式

        max_length = 77
        # 分词并转换为模型输入所需的张量
        text_tokens_and_mask = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=max_length,
            return_overflowing_tokens=False,
            return_length=True
        )                                 
        
        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)

        print(input_ids)
        # 输入嵌入
        inputs_embeds_ = model.get_input_embeddings()(input_ids)
        # 使用平均池化
        inputs_embeds = model.text_model.embeddings(inputs_embeds=inputs_embeds_)
        
        last_token_embeds = inputs_embeds.mean(dim=1, keepdim=True)
        # last_token_embeds = inputs_embeds.mean(dim=1, keepdim=True)

        # 测试平均
        fake_input_ids = torch.zeros(4, 1).to(device)
        fake_attention_mask = torch.ones(4, 1).to(device)
        
        with torch.no_grad():
            # 获取文本嵌入
            outputs = model(
                input_ids=input_ids,                
                inputs_embeds=inputs_embeds
                # attention_mask=attention_mask
            )
            # 通常使用最后一个隐藏层的平均作为句子的表示
            embeddings = outputs.last_hidden_state

        # 打印每个文本的嵌入向量
        for text, embedding in zip(texts, embeddings):
            print(f"文本: {text}")
            print(f"嵌入向量: {embedding.shape}\n")
        
        return embeddings

    def generate_images(self, emb_text: torch.Tensor, uc: torch.Tensor, num_images: int, save_dir: str):
        """
        生成图片并保存到指定目录。

        Args:
            emb_text (torch.Tensor): 文本嵌入，形状 [batch_size, embedding_dim]。
            uc (torch.Tensor): 无条件嵌入，形状 [batch_size, clip_embedding_dim]。
            num_images (int): 需要生成的图片数量。
            save_dir (str): 保存图片的目录路径。
        """
        # 确保 save_dir 存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成图片
        samples, _ = self.model.sample_log(
            cond=None, 
            text=emb_text, 
            batch_size=num_images, 
            ddim=True,
            ddim_steps=200,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,                    
            eta=1.0,
            multi=False
        )
        
        # 逐张解码和保存图片，以避免显存溢出
        for i in range(num_images):
            img = self.decode_to_im(samples[i].unsqueeze(0), n_samples=1, nrow=1)
            img.save(os.path.join(save_dir, f'generated_{i+1}.png'))
            if (i + 1) % 10 == 0:
                print(f"Saved {i+1}/{num_images} images")

    def visualize_and_save(self, out: Dict[str, torch.Tensor], data, save_dir: str) -> Dict[str, Any]:
        """
        根据嵌入生成图像并保存到指定路径。

        Args:
            out (Dict[str, torch.Tensor]): 包含嵌入的输出字典。
            data: 额外的输入数据（这里未直接使用）。
            save_dir (str): 保存生成图像的目录路径。

        Returns:
            dict: 包含生成图像的字典。
        """
        embeddings = out['embeddings']  # 形状 [batch_size, embedding_dim]
        batch_size = embeddings.shape[0]

        # 准备无条件的条件
        text_list_cpu = [""] * batch_size
        
        self.clip.to("cpu")
        with torch.no_grad():
            uc = self.clip.encode(text_list_cpu)  # 假设返回形状为 [batch_size, clip_embedding_dim]
        uc = uc.to(embeddings.device)

        # 生成并保存图片，逐张处理以避免超显存
        self.generate_images(
            emb_text=embeddings, 
            uc=uc, 
            num_images=batch_size, 
            save_dir=save_dir
        )
        
        print(f"Saved {batch_size} images to {save_dir}")

        return {'images': None}  # 由于图片已保存，不返回具体图像

    def test(self):
        # 设置随机种子为0
        self.set_seed(self.seed)
        torch.manual_seed(self.seed)  # 确保PyTorch的种子也被设置

        # 定义输入文本
        texts = ["a polyp."]   # 根据你的需要调整批量大小
        out_embedding = self.test_clip_embedding_encoder(texts=texts)

        data = {}
        print(f"Initial embedding shape: {out_embedding.shape}")

        # 加载所有 .pt 文件的路径
        emb_dir = '/home/hyl/yujia/A_few_shot/evaluation/eval_fake_data/opt_embs'
        emb_files = [f for f in os.listdir(emb_dir) if f.endswith('.pt')]

        # 定义输出主目录
        output_main_dir = "/home/hyl/yujia/A_few_shot/evaluation/eval_fake_data/a_polyp"

        for emb_file in emb_files:
            emb_path = os.path.join(emb_dir, emb_file)
            
            # 加载 embeddings
            emb_opt_text = torch.load(emb_path, map_location=self.device)
            emb_opt_text = emb_opt_text.to(self.device)
            
            # 计算批大小
            batch_size = 100
            # 假设 emb_opt_text 的形状是 [1, embedding_dim]
            emb_opt_text_batch = emb_opt_text.repeat(batch_size, 1, 1)  # [100, 1, embedding_dim]
            out_embedding_batch = out_embedding.repeat(batch_size, 1, 1)  # [100, 1, embedding_dim]
            
            # 混合 embeddings
            out_embedding_mixed = 0.5 * emb_opt_text_batch + 0.5 * out_embedding_batch  # [100, 1, embedding_dim]
            # out_embedding_mixed = out_embedding_mixed.squeeze(1)  # [100, embedding_dim]

            # 准备输出字典
            out = {'embeddings': out_embedding_mixed}  # 形状 [100, embedding_dim]

            # 生成子文件夹名称：去掉前缀和 '_emb_opt'
            subfolder_name = emb_file.replace('_emb_opt', '').replace('.pt', '')
            save_subdir = os.path.join(output_main_dir, subfolder_name)
            os.makedirs(save_subdir, exist_ok=True)

            # 生成 100 张图片
            self.visualize_and_save(out, data, save_subdir)
            print(f"Saved 100 images to {save_subdir}")

# 示例用法：
if __name__ == "__main__":
    a = YourModelClass()
    a.test()
