import torch
from transformers import CLIPTokenizer, CLIPTextModel
# from langit.polypdiffusion.polyp_diffusion_pipeline import PolypDiffusionPipeline
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from einops import rearrange
from omegaconf import OmegaConf
# import sys
# sys.path.append('/home/user/projects/ldm')
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from torchvision.transforms.functional import to_tensor
from PIL import Image
from pytorch_lightning import seed_everything
from ldm.data.kvasir import Fake_mask
import argparse
from typing import List
# export PYTHONPATH=$PYTHONPATH:/home/hyl/yujia/A_few_shot/langit/polypdiffusion/ldm

class PolypDiffusionPipeline:
    def __init__(self, config_path, ckpt_path, batch_size=1, seed=55):
        # Initialize with config, checkpoint, batch size, and seed
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.seed = seed
        seed_everything(self.seed)
        
        # Load model
        self.model, _ = self.load_model(self.config_path, self.ckpt_path)

        # Prepare dataset (unused in this case, removed for clarity)
        self.dataset = Fake_mask(size=256)  # Placeholder dataset

        # Setup results directory (optional, not needed in this format)
        self.results_dir = '/home/hyl/yujia/A_few_shot/langit/polypdiffusion'
        
        os.makedirs(self.results_dir, exist_ok=True)

    def load_model(self, config_path, ckpt_path):
        """Load model from config and checkpoint"""
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu") if ckpt_path else {"state_dict": None}
        model = self.load_model_from_config(config.model, pl_sd["state_dict"])
        return model, pl_sd.get("global_step", None)

    def load_model_from_config(self, config, sd):
        """Instantiate the model and load weights"""
        model = instantiate_from_config(config)
        model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        return model

    def generate_images(self, prompt: str, count: int = 4) -> List[Image.Image]:
        """Generate images from model using text conditioning"""
        clip = FrozenCLIPEmbedder(device="cpu")
        
        # Check if prompt is already an embedding (tensor)
        if isinstance(prompt, torch.Tensor):  # If the prompt is already an embedding
            text = prompt.to('cuda')  # Move it to GPU
        else:  # If the prompt is a string, encode it
            text_list = [prompt] * count  # Repeat the prompt for batch size
            text = clip.encode(text_list).to('cuda')  # Encode and move to GPU

        # Prepare unconditional conditioning (similar to 'uc' in the original pipeline)
        uc = clip.encode([""] * count)  # Unconditional conditioning (empty string prompts)
        uc = uc.to('cuda')  # Move to GPU

        # Generate samples using the model's sampling method
        samples, _ = self.model.sample_log(
            cond=None,
            text=text,
            batch_size=count,
            ddim=True,
            ddim_steps=200,
            unconditional_guidance_scale=5.5,
            unconditional_conditioning=uc,
            eta=1.0,
            multi=False,
            quantize_denoised=False
        )

        # Decode the samples to get images
        samples = self.model.decode_first_stage(samples)
        return samples

    def run_pipeline(self, prompts: List[str], num_repeats: int = 4) -> List[Image.Image]:
        """Run the pipeline with given prompts and generate images"""
        images_all = []
        if isinstance(prompts, str):  # 如果prompts是单个字符串
            prompts = [prompts]  # 将其转换成列表
    
        for prompt in prompts:
            images = self.generate_images(prompt, count=num_repeats)
            images_all.extend(images)
        return images_all

    def get_results(self, prompts: List[str], count: int = 1) -> List[Image.Image]:
        """Generate and retrieve the images from the pipeline"""
        # Run the pipeline to generate images
        images = self.run_pipeline(prompts, count)
        return images

def test_clip_text_encoder(texts):
    # 检查是否有可用的 GPU，否则使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载 CLIP 的分词器和文本模型
    tokenizer = CLIPTokenizer.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
    model.to(device)
    model.eval()  # 设置模型为评估模式

    # 示例文本
    
    max_length = 77
    # 分词并转换为模型输入所需的张量
    inputs = tokenizer(texts, padding=False, truncation=True, return_tensors="pt" )
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

if __name__ == "__main__":

    
    texts = [ "a red polyp with blood attached."]
    pipeline = PolypDiffusionPipeline(
        config_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml',
        ckpt_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/resume_sd/2024-11-21T00-12-28_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=49-step=699.ckpt',
        batch_size=1,
        seed=42)
    out_embedding = test_clip_text_encoder(texts=texts)
    out_embedding = out_embedding[:, -1, :].unsqueeze(1)
    results = pipeline.get_results(out_embedding.unsqueeze(0), count=1)
    for i, img in enumerate(results):
        img.save(f"generated_image_{i}.png")