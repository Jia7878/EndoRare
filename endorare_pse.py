import argparse
import copy
import os
from pathlib import Path
from pytorch_lightning.trainer import Trainer
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

# 全局设置随机种子和确定性选项
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 如果使用了其他随机库（如 random），也需要设置
    import random
    random.seed(seed)
    
    # 确保计算的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置全局随机种子
# 在命令行参数中设置seed，默认值为0
# 这里先定义parser，稍后解析参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer()
parser = argparse.ArgumentParser()

# directories
parser.add_argument('--config', type=str, default='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/stable-diffusion.yaml')
parser.add_argument('--ckpt', type=str, default='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/resume_sd/2024-11-20T21-18-03_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=49-step=699.ckpt')
parser.add_argument('--save_folder', type=str, default='/home/hyl/yujia/test_few_imagic')

# hyperparameters
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--stage1_lr', type=float, default=0.001)
parser.add_argument('--stage1_num_iter', type=int, default=1000)
parser.add_argument('--stage2_lr', type=float, default=1e-6)
parser.add_argument('--stage2_num_iter', type=int, default=1000)

# 移除 --input_image_path 和 --set_random_seed 参数，因为我们批量处理文件夹中的图片，并已在全局设置了随机种子
parser.add_argument('--save_checkpoint', type=bool, default=False)

args = parser.parse_args()

# 设置全局随机种子
set_seed(args.seed)

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt"""
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd.get("global_step", 0)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def load_img(path, target_size=256):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2. * image - 1.

def decode_to_im(model, samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(
        ims.cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c',
        n1=n_samples // nrow,
        n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))

def optimize_embedding(model, clip, init_image, stage1_lr, stage1_num_iter):
    """仅优化嵌入"""
    sampler = DDIMSampler(model)

    # Prepare input image
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image.unsqueeze(0).to(device)))

    # Obtain text embedding
    text_list_cpu = [""] 
    uc = clip.encode(text_list_cpu).to(device)

    prompt = ["V* polyp"] 
    text = clip.encode(prompt).to(device)
    emb_text = text.clone()
    emb_text.requires_grad = True

    opt_text = torch.optim.Adam([emb_text], lr=stage1_lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(stage1_num_iter), desc="Optimizing Embedding")
    for i in pbar:
        opt_text.zero_grad()

        # 使用全局随机种子，无需在此设置
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1, ), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, None, emb_text)
        loss = criteria(pred_noise, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt_text.step()


    emb_opt_text = emb_text.detach()
    

    return emb_opt_text

def fine_tune_model(model, emb_text, init_latent, stage2_lr, stage2_num_iter):
    """仅微调模型"""
    model.train()
    emb_text.requires_grad = False

    opt = torch.optim.Adam(model.model.parameters(), lr=stage2_lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(stage2_num_iter), desc="Fine-Tuning Model")
    for i in pbar:
        opt.zero_grad()

        # 使用全局随机种子，无需在此设置
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(model.num_timesteps, (1, ), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, None, emb_text)
        loss = criteria(pred_noise, noise)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss has NaN or Inf values.")
            continue  # 跳过这一步骤

        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()


    model.eval()
    return model

def generate_images(model, emb_text, uc, num_images, save_dir):
    """生成图片"""
    # 确保emb_text和uc的batch size与num_images匹配
    emb_text_batch = emb_text.repeat(num_images, 1, 1)  # 假设emb_text形状为 [1, ...]
    uc_batch = uc.repeat(num_images, 1, 1)  # 假设uc形状为 [1, ...]

    # 生成图片
    samples, _ = model.sample_log(
        cond=None, 
        text=emb_text_batch, 
        batch_size=num_images, 
        ddim=True,
        ddim_steps=200,
        unconditional_guidance_scale=7.5,
        unconditional_conditioning=uc_batch,                    
        eta=1.0,
        multi=False
    )
    
    # 确保样本数量与num_images一致
    for i in range(num_images):
        img = decode_to_im(model, samples[i].unsqueeze(0))
        img.save(os.path.join(save_dir, f'generated_{i+1}.png'))
        
    del samples, _

def main():
    # 加载模型
    global_model = load_model_from_config(args.config, args.ckpt, device)
    clip = FrozenCLIPEmbedder(device="cpu")

    # 读取所有图片
    eval_real_data_dir = '/home/hyl/yujia/A_few_shot/evaluation/eval_real_data'
    eval_fake_data_dir = '/home/hyl/yujia/A_few_shot/evaluation/eval_fake_data'
    opt_embs_dir = os.path.join(eval_fake_data_dir, 'opt_embs')
    os.makedirs(eval_fake_data_dir, exist_ok=True)
    os.makedirs(opt_embs_dir, exist_ok=True)

    image_paths = [
        os.path.join(eval_real_data_dir, fname) 
        for fname in os.listdir(eval_real_data_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    for img_path in image_paths:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'Processing image: {image_name}')

        # 创建保存目录
        save_dir_v_only = os.path.join(eval_fake_data_dir, 'V_only', image_name)
        save_dir_finetune_only = os.path.join(eval_fake_data_dir, 'finetune_only', image_name)
        os.makedirs(save_dir_v_only, exist_ok=True)
        os.makedirs(save_dir_finetune_only, exist_ok=True)

        # 读取输入图片
        init_image = load_img(img_path).to(device)

        # 获取初始潜在表示
        init_latent = global_model.get_first_stage_encoding(global_model.encode_first_stage(init_image.unsqueeze(0)))

        # 获取原始嵌入
        text_list_cpu = [""] 
        uc = clip.encode(text_list_cpu).to(device)

        prompt = ["V* polyp"] 
        text = clip.encode(prompt).to(device)
        emb_text_original = text.clone()

        # ------------- 仅优化嵌入 -------------
        print('Starting embedding optimization...')
        emb_opt_text = optimize_embedding(
            model=global_model,
            clip=clip,
            init_image=init_image,
            stage1_lr=args.stage1_lr,
            stage1_num_iter=args.stage1_num_iter
        )

        # 保存优化后的嵌入
        opt_emb_path = os.path.join(opt_embs_dir, f'{image_name}_emb_opt.pt')
        torch.save(emb_opt_text, opt_emb_path)
        print(f'Optimized embedding saved to {opt_emb_path}')

        # 生成100张图片
        print('Generating images with optimized embedding...')
        generate_images(
            model=global_model,
            emb_text=emb_opt_text,
            uc=uc,
            num_images=100,
            save_dir=save_dir_v_only
        )
        
        torch.cuda.empty_cache()
        # ------------- 仅微调模型 -------------
        print('Starting model fine-tuning...')
        finetuned_model = copy.deepcopy(global_model)
        finetuned_model = fine_tune_model(
            model=finetuned_model,
            emb_text=emb_text_original,
            init_latent=init_latent,
            stage2_lr=args.stage2_lr,
            stage2_num_iter=args.stage2_num_iter
        )

        # 生成100张图片
        print('Generating images with fine-tuned model...')
        generate_images(
            model=finetuned_model,
            emb_text=emb_text_original,
            uc=uc,
            num_images=100,
            save_dir=save_dir_finetune_only
        )

        print(f'Finished processing image: {image_name}')

if __name__ == '__main__':
    main()
