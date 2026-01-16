from typing import Union, List, Optional, Dict, Any
# import torch
# import os
from torchvision.transforms import ToPILImage
# import argparse
import copy
import os
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
# #################################
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
from attention import BidirectionalSimilarityAttentionFusion_
import matplotlib.pyplot as plt

def visualize_weights(weights, title="Fusion Weights"):
    """
    参数:
        weights: [batch, seq_len, 1] 的张量
        title: 图表标题
    """
    weights = weights.detach().cpu().numpy()
    batch_size, seq_len, _ = weights.shape
    
    for i in range(min(batch_size, 4)):  # 仅可视化前4个样本
        plt.figure(figsize=(10, 2))
        plt.plot(range(seq_len), weights[i].squeeze(), marker='o')
        plt.title(f"{title} - Sample {i+1}")
        plt.xlabel("Token Position")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.show()
        save_dir = "/home/hyl/yujia"
        file_path = os.path.join(save_dir, f"sample_{i+1}.png")
        plt.savefig(file_path)

class YourModelClass:
    def __init__(self):
        self.config_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml"
        self.ckpt_path = "/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/Text_location_215/2025-01-03T12-29-27_stable-diffusion/polyp_text_ckpt/epoch=11-step=299.ckpt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model from config and checkpoint
        self.global_model = self.load_model_from_config(self.config_path, self.ckpt_path, self.device)
        self.model = self.global_model  # Directly use global_model if no modification needed
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Initialize CLIP model only once during initialization
        self.clip = FrozenCLIPEmbedder(device="cpu")
        
        # DDIM sampler initialization (if needed)
        self.sampler = None 
        
        # Set random seed
        self.seed = 20
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
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
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

    def visualize_and_save(self, out: Dict[str, torch.Tensor], data, save_path: str) -> Dict[str, Any]:
        """
        Generates an image from embeddings and saves it to the specified path.

        Args:
            out (Dict[str, torch.Tensor]): Output dictionary containing embeddings.
            data: Additional input data (not used here directly).
            save_path (str): Path to save the generated image.

        Returns:
            dict: A dictionary with the 'image' key containing the generated image.
        """
        embeddings = out['embeddings']
        
        batch_size = embeddings.shape[0]

        # Prepare the unconditional conditioning
        text_list_cpu = [""] * batch_size
        
        
        
        prompt = ['a polyp']* batch_size
        cond = self.test_clip_embedding_encoder(texts = prompt)
        
        cond = cond.to(embeddings.device)
        
        
        
        clip = self.clip.to("cpu")
        uc = clip.encode(text_list_cpu)
        uc = uc.to(embeddings.device)

        # Sample generation from the model (using DDIM sampling)
        stageI_generations, _ = self.model.sample_log(
            cond=None,
            text=embeddings,
            batch_size=batch_size,
            ddim=True,
            ddim_steps=200,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=1.0,
            multi=False
        )

        # Decode latent representations into image
        img = self.decode_to_im(stageI_generations, n_samples=batch_size, nrow=1)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save image to the specified path
        img.save(save_path)

        # Return the generated image for further processing if needed
        return {'image': img}
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
        # tokenizer = self.clip.tokenizer
        # model = self.clip.transformer
        model.to(device)
        model.eval()  # 设置模型为评估模式

        max_length = 77
        # 分词并转换为模型输入所需的张量
        #* 得到rtokenizer编码后的token和attention mask
        text_tokens_and_mask = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length = max_length,
                                         return_overflowing_tokens=False,return_length=True)                                 
        
        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)

        print(input_ids)
        #* input_embeddings
        
        inputs_embeds_ = model.get_input_embeddings()(input_ids)
        # 使用最大池化
        
        inputs_embeds = model.text_model.embeddings(inputs_embeds = inputs_embeds_)
        

        
        with torch.no_grad():
            # 获取文本嵌入
            outputs = model(
                input_ids = input_ids,                
                inputs_embeds = inputs_embeds
                # attention_mask = attention_mask
                )
            # 通常使用最后一个隐藏层的平均作为句子的表示
            embeddings = outputs.last_hidden_state

        # 打印每个文本的嵌入向量
        for text, embedding in zip(texts, embeddings):
            print(f"文本: {text}")
            print(f"嵌入向量: {embedding.shape}\n")
        
        return embeddings
    
    def test(self):
        torch.manual_seed(self.seed)
        texts = [ "Polyp, 1.0cm, 0-Is type, central depression at the apex, similar color to surrounding mucosa, adenomatous polyp with low-grade epithelial dysplasia, located at the center of the image."]
        out_embedding = self.test_clip_embedding_encoder(texts=texts)

        # # 确保 inputs_embeds 也在 GPU 上，然后再比较
        # # inputs_embeds = inputs_embeds_input  # 假设 inputs_embeds 是输入的嵌入张量，你需要保证它也在同一设备
        # # are_equal = torch.all(torch.eq(input_embeds_from_embeds.cuda(), inputs_embeds.cuda()))
        # # print(are_equal)

         

        
        data = {}
        print(out_embedding.shape)
        #! V*
        batch_size = out_embedding.shape[0]
        emb_opt_text = torch.load('/home/hyl/yujia/A_few_shot/paper_experiments/data/opt_embs/6_emb_opt.pt') 
        emb_opt_text = emb_opt_text.to(self.device)
        emb_opt_text = emb_opt_text.repeat(batch_size, 1, 1) 
        aggregator = BidirectionalSimilarityAttentionFusion_().to(self.device)
        
        
        # out_embedding = aggregator(emb_opt_text,out_embedding)  
        out_embedding = 0.85*emb_opt_text + 0.2*out_embedding
        # out_embedding = aggregator(out_embedding,emb_opt_text)  
        # out_embedding = 0.5*emb_opt_text + 0.5*out_embedding
        # out_embedding = emb_opt_text
        # visualize_weights(weights, title="Fusion Weights")
        
        
        
        
        out = {'embeddings': out_embedding}
        save_path = "/home/hyl/yujia/A_few_shot/test_image_data/init_input_emb_global_bi.png"
        result = self.visualize_and_save(out, data, save_path)

a = YourModelClass()
a.test()