import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import lpips
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTImageProcessor, ViTModel

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# 自定义数据集类
# ===========================

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        初始化自定义图像数据集。

        Args:
            folder_path (str): 图像文件夹路径。
            transform (torchvision.transforms.Compose, optional): 图像预处理方法。
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', 'jpeg', '.png'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个全黑图像或其他占位符
            image = Image.new('RGB', (299, 299), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        return image, 0  # 标签为0，因不需要实际标签

# ===========================
# 工具函数
# ===========================

def pair_images(real_images, fake_images):
    """
    按文件名配对 real 和 fake 图像。

    Args:
        real_images (list): Real 图像路径列表。
        fake_images (list): Fake 图像路径列表。

    Returns:
        paired (list of tuples): 配对后的图像路径列表。
    """
    real_dict = {os.path.basename(path): path for path in real_images}
    fake_dict = {os.path.basename(path): path for path in fake_images}
    paired = []
    for filename, real_path in real_dict.items():
        if filename in fake_dict:
            paired.append((real_path, fake_dict[filename]))
    return paired

def convert_to_float(value):
    """
    将 NumPy 或 PyTorch 类型转换为原生 Python 类型以便 JSON 序列化。

    Args:
        value: 需要转换的值。

    Returns:
        转换后的值。
    """
    if isinstance(value, np.float32) or isinstance(value, np.float64):
        return float(value)
    elif isinstance(value, torch.Tensor):
        return float(value.item())
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    return value

# ===========================
# 指标计算函数
# ===========================

def compute_class_fid(real_folder, generated_folder, device='cuda'):
    """
    计算每个类别的 FID，并返回所有类别的平均 FID。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        average_fid (float): 所有类别的平均 FID 分数。
    """
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    fid_scores = []
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        fid_value = fid_score.calculate_fid_given_paths(
            [real_class_folder, generated_class_folder],
            batch_size=50,
            device=device,
            dims=2048
        )
        fid_scores.append(fid_value)
        print(f"Class '{class_name}': FID = {fid_value}")
    
    average_fid = sum(fid_scores) / len(fid_scores) if fid_scores else None
    print(f"Average FID across all classes: {average_fid}")
    return average_fid

def compute_inception_score(real_folder, generated_folder, batch_size=32, device='cuda'):
    """
    计算每个类别的 Inception Score (IS)，并返回所有类别的均值和标准差。

    Args:
        real_folder (str): 包含真实图像的文件夹路径（不用于 IS 计算）。
        generated_folder (str): 包含生成图像的文件夹路径。
        batch_size (int): 批处理大小。
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        mean_score (float), std_score (float): IS 分数的均值和标准差。
    """
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    class_folders = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    class_is_scores = []
    
    for class_name in class_folders:
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in the generated folder.")
            continue
        
        image_files = [
            f for f in os.listdir(generated_class_folder)
            if f.lower().endswith(('.jpg', 'jpeg', 'png'))
        ]
        
        if not image_files:
            print(f"No images found for class '{class_name}'. Skipping.")
            continue
        
        preds_list = []
        
        for image_file in image_files:
            image_path = os.path.join(generated_class_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = inception_model(image)
                preds_list.append(preds)
        
        if not preds_list:
            print(f"No valid images found for class '{class_name}'. Skipping.")
            continue
        
        preds = torch.cat(preds_list, dim=0)
        p_y = F.softmax(preds, dim=1)
        p_y_mean = torch.mean(p_y, dim=0)
        
        kl_div = p_y * (torch.log(p_y) - torch.log(p_y_mean))
        kl_div = torch.sum(kl_div, dim=1)
        
        class_is_score = torch.exp(torch.mean(kl_div)).item()
        class_is_scores.append(class_is_score)
        print(f"Class '{class_name}': Inception Score = {class_is_score}")
    
    mean_score = np.mean(class_is_scores) if class_is_scores else None
    std_score = np.std(class_is_scores) if class_is_scores else None
    print(f"Inception Score Mean: {mean_score}, Std: {std_score}")
    
    return mean_score, std_score

def compute_class_mmd(real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
    """
    计算每个类别的 MMD，并返回所有类别的平均 MMD。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。
        gamma (float): RBF 核参数。

    Returns:
        average_mmd (float): 所有类别的平均 MMD 分数。
    """
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    mmd_scores = []
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        # 加载预训练的 InceptionV3 模型
        model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        model.eval()
        
        # 提取特征
        real_features = extract_features_from_folder(real_class_folder, model, device, batch_size)
        generated_features = extract_features_from_folder(generated_class_folder, model, device, batch_size)
        
        if real_features.size(0) == 0 or generated_features.size(0) == 0:
            print(f"No features extracted for class '{class_name}'. Skipping.")
            continue
        
        # 计算 MMD
        mmd_value = compute_mmd(real_features, generated_features, kernel='rbf', gamma=gamma)
        mmd_scores.append(mmd_value)
        print(f"Class '{class_name}': MMD = {mmd_value}")
    
    average_mmd = sum(mmd_scores) / len(mmd_scores) if mmd_scores else None
    print(f"Average MMD across all classes: {average_mmd}")
    return average_mmd

def compute_mmd(real_features, generated_features, kernel='rbf', gamma=1.0):
    """
    计算 MMD 值。

    Args:
        real_features (torch.Tensor): 真实图像的特征。
        generated_features (torch.Tensor): 生成图像的特征。
        kernel (str): 核类型（目前仅支持 'rbf'）。
        gamma (float): RBF 核参数。

    Returns:
        mmd_value (float): 计算得到的 MMD 值。
    """
    def rbf_kernel_func(x, y, gamma=1.0):
        dist_matrix = torch.cdist(x, y, p=2)
        return torch.exp(-gamma * dist_matrix ** 2)
    
    real_real_kernel = rbf_kernel_func(real_features, real_features, gamma)
    gen_gen_kernel = rbf_kernel_func(generated_features, generated_features, gamma)
    real_gen_kernel = rbf_kernel_func(real_features, generated_features, gamma)
    
    mmd_squared = real_real_kernel.mean() + gen_gen_kernel.mean() - 2 * real_gen_kernel.mean()
    return mmd_squared.item()

def compute_class_kid(real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
    """
    计算每个类别的 KID，并返回所有类别的平均 KID。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。
        gamma (float): RBF 核参数。

    Returns:
        average_kid (float): 所有类别的平均 KID 分数。
    """
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    kid_scores = []
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        # 加载预训练的 InceptionV3 模型
        model = get_inception_model(device)
        model.eval()
        
        # 提取特征
        real_features = extract_features_from_images_from_folder(real_class_folder, model, device, batch_size)
        generated_features = extract_features_from_images_from_folder(generated_class_folder, model, device, batch_size)
        
        if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
            print(f"No features extracted for class '{class_name}'. Skipping.")
            continue
        
        # 计算 KID（即 MMD 值）
        kid_value = compute_mmd_kid(real_features, generated_features, gamma)
        kid_scores.append(kid_value)
        print(f"Class '{class_name}': KID = {kid_value}")
    
    average_kid = sum(kid_scores) / len(kid_scores) if kid_scores else None
    print(f"Average KID across all classes: {average_kid}")
    return average_kid

def compute_mmd_kid(real_features, generated_features, gamma=1.0):
    """
    计算 MMD 值用于 KID 计算。

    Args:
        real_features (numpy.ndarray): 真实图像的特征。
        generated_features (numpy.ndarray): 生成图像的特征。
        gamma (float): RBF 核参数。

    Returns:
        mmd_value (float): 计算得到的 MMD 值。
    """
    kernel_real = compute_rbf_kernel_matrix(real_features, real_features, gamma)
    kernel_generated = compute_rbf_kernel_matrix(generated_features, generated_features, gamma)
    kernel_cross = compute_rbf_kernel_matrix(real_features, generated_features, gamma)
    
    mmd_value = (np.mean(kernel_real) + np.mean(kernel_generated) - 2 * np.mean(kernel_cross))
    return mmd_value

def compute_rbf_kernel_matrix(X, Y, gamma=1.0):
    """
    计算 RBF 核矩阵。

    Args:
        X (numpy.ndarray): 输入数据 X。
        Y (numpy.ndarray): 输入数据 Y。
        gamma (float): RBF 核参数。

    Returns:
        kernel_matrix (numpy.ndarray): 计算得到的核矩阵。
    """
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim == 3:
        Y = Y.reshape(Y.shape[0], -1)
    
    return rbf_kernel(X, Y, gamma=gamma)

def compute_clip_similarity_per_class(real_folder, generated_folder, model, preprocess, device):
    """
    计算每个类别的 CLIP 相似度，并返回所有类别的平均相似度。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        model (CLIPModel): 加载的 CLIP 模型。
        preprocess (CLIPProcessor): CLIP 的预处理器。
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        mean_similarity (float): 所有类别的平均相似度。
    """
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    all_similarities = []
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        real_image_paths = [
            os.path.join(real_class_folder, img) for img in os.listdir(real_class_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        fake_image_paths = [
            os.path.join(generated_class_folder, img) for img in os.listdir(generated_class_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(real_image_paths) != len(fake_image_paths):
            print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
            continue
        
        paired_images = pair_images(real_image_paths, fake_image_paths)
        
        if not paired_images:
            print(f"No paired images found for class '{class_name}'. Skipping.")
            continue
        
        similarities = []
        for real_path, fake_path in paired_images:
            real_features = extract_clip_features([real_path], model, preprocess, device)
            fake_features = extract_clip_features([fake_path], model, preprocess, device)
            similarity = torch.cosine_similarity(real_features, fake_features, dim=-1).item()
            similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            all_similarities.append(avg_similarity)
            print(f"Class '{class_name}': CLIP Similarity = {avg_similarity}")
    
    mean_similarity = np.mean(all_similarities) if all_similarities else None
    print(f"Average CLIP Similarity across all classes: {mean_similarity}")
    return mean_similarity

def compute_dino_alignment_per_class(real_folder, generated_folder, model, processor, device):
    """
    计算每个类别的 DINO 对齐度，并返回所有类别的平均对齐度。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        model (ViTModel): 加载的 DINO 模型。
        processor (ViTImageProcessor): DINO 的预处理器。
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        mean_alignment (float): 所有类别的平均对齐度。
    """
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    all_alignments = []
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        real_image_paths = [
            os.path.join(real_class_folder, img) for img in os.listdir(real_class_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        fake_image_paths = [
            os.path.join(generated_class_folder, img) for img in os.listdir(generated_class_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(real_image_paths) != len(fake_image_paths):
            print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
            continue
        
        paired_images = pair_images(real_image_paths, fake_image_paths)
        
        if not paired_images:
            print(f"No paired images found for class '{class_name}'. Skipping.")
            continue
        
        alignments = []
        for real_path, fake_path in paired_images:
            real_features = extract_dino_features([real_path], model, processor, device)
            fake_features = extract_dino_features([fake_path], model, processor, device)
            alignment = torch.cosine_similarity(real_features, fake_features, dim=-1).item()
            alignments.append(alignment)
        
        if alignments:
            avg_alignment = np.mean(alignments)
            all_alignments.append(avg_alignment)
            print(f"Class '{class_name}': DINO Alignment = {avg_alignment}")
    
    mean_alignment = np.mean(all_alignments) if all_alignments else None
    print(f"Average DINO Alignment across all classes: {mean_alignment}")
    return mean_alignment

def calculate_ic_lpips_for_subfolders(real_folder: str, generated_folder: str):
    """
    计算每个类别的 IC-LPIPS 分数，并返回所有类别的平均分数。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。

    Returns:
        avg_ic_lpips (float): 所有类别的平均 IC-LPIPS 分数。
    """
    loss_fn = lpips.LPIPS(net='vgg')
    
    real_classes = sorted([
        d for d in os.listdir(real_folder)
        if os.path.isdir(os.path.join(real_folder, d))
    ])
    generated_classes = sorted([
        d for d in os.listdir(generated_folder)
        if os.path.isdir(os.path.join(generated_folder, d))
    ])
    
    assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"
    
    total_score = 0
    total_count = 0
    
    for class_name in real_classes:
        real_class_folder = os.path.join(real_folder, class_name)
        generated_class_folder = os.path.join(generated_folder, class_name)
        
        if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
            print(f"Skipping class '{class_name}' as it does not exist in both folders.")
            continue
        
        real_images = [
            os.path.join(real_class_folder, f) for f in os.listdir(real_class_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        generated_images = [
            os.path.join(generated_class_folder, f) for f in os.listdir(generated_class_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(real_images) != len(generated_images):
            print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
            continue
        
        paired_images = pair_images(real_images, generated_images)
        
        if not paired_images:
            print(f"No paired images found for class '{class_name}'. Skipping.")
            continue
        
        subfolder_score = 0
        subfolder_count = 0
        
        for real_img_path, gen_img_path in paired_images:
            try:
                real_image = Image.open(real_img_path).convert("RGB")
                generated_image = Image.open(gen_img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading images '{real_img_path}' or '{gen_img_path}': {e}")
                continue
            
            real_image_np = np.array(real_image).astype(np.float32)
            generated_image_np = np.array(generated_image).astype(np.float32)
            
            real_image_tensor = lpips.im2tensor(real_image_np)
            generated_image_tensor = lpips.im2tensor(generated_image_np)
            
            score = loss_fn(real_image_tensor, generated_image_tensor)
            subfolder_score += score.item()
            subfolder_count += 1
        
        avg_subfolder_score = subfolder_score / subfolder_count if subfolder_count > 0 else None
        if avg_subfolder_score is not None:
            total_score += avg_subfolder_score
            total_count += 1
            print(f"Class '{class_name}': IC-LPIPS = {avg_subfolder_score}")
    
    avg_ic_lpips = total_score / total_count if total_count > 0 else None
    print(f"Average IC-LPIPS across all classes: {avg_ic_lpips}")
    return avg_ic_lpips

# ===========================
# 特征提取函数
# ===========================

def extract_features_from_folder(folder_path, model, device, batch_size=32):
    """
    从文件夹中提取图像特征。

    Args:
        folder_path (str): 图像文件夹路径。
        model (torch.nn.Module): 用于特征提取的模型。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。

    Returns:
        all_features (torch.Tensor): 所有图像的特征。
    """
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    model.eval()
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f'Extracting features from {folder_path}'):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    return all_features

def extract_features_from_images_from_folder(folder_path, model, device, batch_size=32):
    """
    从文件夹中提取图像特征并返回 NumPy 数组。

    Args:
        folder_path (str): 图像文件夹路径。
        model (torch.nn.Module): 用于特征提取的模型。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。

    Returns:
        features_np (numpy.ndarray): 所有图像的特征，形状为 (N, D)。
    """
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    model.eval()
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f'Extracting features from {folder_path}'):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
    
    features_np = np.concatenate(all_features, axis=0)
    return features_np

def get_inception_model(device):
    """
    加载并返回预训练的 InceptionV3 模型。

    Args:
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        inception (torch.nn.Module): InceptionV3 模型。
    """
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception = inception.to(device)
    inception.eval()
    return inception

def extract_clip_features(image_paths, model, preprocess, device, batch_size=32):
    """
    批量提取 CLIP 特征。

    Args:
        image_paths (list): 图像路径列表。
        model (CLIPModel): CLIP 模型。
        preprocess (CLIPProcessor): CLIP 预处理器。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。

    Returns:
        features (torch.Tensor): 提取的特征，形状为 (N, D)。
    """
    features = []
    model.eval()
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                # 添加一个全黑图像作为占位符
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        inputs = preprocess(images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            batch_features = model.get_image_features(**inputs)
        features.append(batch_features.cpu())
    
    features = torch.cat(features, dim=0)
    return features

def load_clip_model(device='cuda'):
    """
    加载 CLIP 模型和预处理器。

    Args:
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        model (CLIPModel): CLIP 模型。
        preprocess (CLIPProcessor): CLIP 预处理器。
    """
    model = CLIPModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
    return model, processor

def load_dino_model(device='cuda'):
    """
    加载 DINO 模型和预处理器。

    Args:
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        model (ViTModel): DINO 模型。
        processor (ViTImageProcessor): DINO 预处理器。
    """
    processor = ViTImageProcessor.from_pretrained('/home/hyl/yujia/dino-vits16')
    model = ViTModel.from_pretrained('/home/hyl/yujia/dino-vits16')
    model = model.to(device)
    model.eval()
    return model, processor

def extract_dino_features(image_paths, model, processor, device, batch_size=32):
    """
    批量提取 DINO 特征。

    Args:
        image_paths (list): 图像路径列表。
        model (ViTModel): DINO 模型。
        processor (ViTImageProcessor): DINO 预处理器。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。

    Returns:
        features (torch.Tensor): 提取的特征，形状为 (N, D)。
    """
    features = []
    model.eval()
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINO features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                # 添加一个全黑图像作为占位符
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用 last_hidden_state 的均值作为特征
        batch_features = outputs.last_hidden_state.mean(dim=1)
        features.append(batch_features.cpu())
    
    features = torch.cat(features, dim=0)
    return features

def preprocess_image(image, device):
    """
    图像预处理函数，适用于 InceptionV3 模型。
    
    Args:
        image (PIL.Image): 输入图像。
        device (str): 计算设备 ('cuda' 或 'cpu')。
    
    Returns:
        image (torch.Tensor): 预处理后的图像张量。
    """
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    return image

def extract_features_from_images(images, model, device):
    """
    从一批图像中提取特征。

    Args:
        images (list): 图像对象列表。
        model (torch.nn.Module): 用于特征提取的模型。
        device (str): 计算设备 ('cuda' 或 'cpu')。

    Returns:
        features_np (numpy.ndarray): 提取的特征数组。
    """
    features = []
    with torch.no_grad():
        for image in images:
            image = preprocess_image(image, device)
            feature = model(image)
            features.append(feature.cpu().numpy())
    return np.array(features)

# ===========================
# KID 计算函数
# ===========================

def compute_kid(real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
    """
    计算 KID (Kernel Inception Distance)。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        batch_size (int): 批处理大小。
        gamma (float): RBF 核参数。

    Returns:
        kid_value (float): 计算得到的 KID 值。
    """
    real_images = [
        os.path.join(real_folder, img) for img in os.listdir(real_folder)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    generated_images = [
        os.path.join(generated_folder, img) for img in os.listdir(generated_folder)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    model = get_inception_model(device)
    model.eval()
    
    real_features = extract_features_from_images_from_folder(real_folder, model, device, batch_size)
    generated_features = extract_features_from_images_from_folder(generated_folder, model, device, batch_size)
    
    kid_value = compute_mmd_kid(real_features, generated_features, gamma)
    
    return kid_value

# ===========================
# 保存结果函数
# ===========================

def save_results_to_json(results, file_path):
    """
    将结果保存到 JSON 文件。

    Args:
        results (dict): 结果字典。
        file_path (str): 保存路径。
    """
    # 转换所有值为可序列化类型
    results_serializable = {k: convert_to_float(v) if not isinstance(v, dict) else {sub_k: convert_to_float(sub_v) for sub_k, sub_v in v.items()} for k, v in results.items()}
    
    with open(file_path, 'w') as json_file:
        json.dump(results_serializable, json_file, indent=4)
    print(f"Results saved to {file_path}")

# ===========================
# 主函数
# ===========================

def compute_and_save_metrics(real_folder, generated_folder, device, output_path):
    """
    计算所有指标并保存结果。

    Args:
        real_folder (str): 包含真实图像的文件夹路径。
        generated_folder (str): 包含生成图像的文件夹路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。
        output_path (str): 结果保存路径。
    """
    # 计算 FID
    print("Computing FID...")
    average_fid_score = compute_class_fid(real_folder, generated_folder, device=device)
    
    # 计算 IS
    print("\nComputing Inception Score...")
    mean_score, std_score = compute_inception_score(real_folder, generated_folder, device=device)
    
    # 计算 MMD
    print("\nComputing MMD...")
    average_mmd = compute_class_mmd(real_folder, generated_folder, device=device, batch_size=32, gamma=1.0)
    
    # CLIP similarity
    print("\nComputing CLIP Similarity...")
    clip_model, clip_preprocess = load_clip_model(device)
    mean_similarity = compute_clip_similarity_per_class(real_folder, generated_folder, clip_model, clip_preprocess, device)
    
    # DINO alignment
    print("\nComputing DINO Alignment...")
    dino_model, dino_processor = load_dino_model(device)
    mean_alignment = compute_dino_alignment_per_class(real_folder, generated_folder, dino_model, dino_processor, device)
    
    # IC-LPIPS
    print("\nComputing IC-LPIPS...")
    ic_lpips_score = calculate_ic_lpips_for_subfolders(real_folder, generated_folder)
    
    # KID
    print("\nComputing KID...")
    average_kid = compute_class_kid(real_folder, generated_folder, device=device, batch_size=32, gamma=1.0)
    
    # 组织结果
    results = {
        "FID": average_fid_score,
        "Inception Score (IS)": {
            "mean": mean_score,
            "std": std_score
        },
        "MMD": average_mmd,
        "CLIP Similarity": mean_similarity,
        "DINO Alignment": mean_alignment,
        "IC-LPIPS": ic_lpips_score,
        "KID": average_kid
    }
    
    # 保存结果到 JSON
    save_results_to_json(results, output_path)

# ===========================
# 用法示例
# ===========================

if __name__ == "__main__":
    # 解析命令行参数（可选）
    # parser = argparse.ArgumentParser(description='Compute image quality metrics between real and generated images.')
    # parser.add_argument('--real_folder', type=str, required=True, help='Path to the folder containing real images.')
    # parser.add_argument('--generated_folder', type=str, required=True, help='Path to the folder containing generated images.')
    # parser.add_argument('--output_path', type=str, required=True, help='Path to save the results JSON.')
    # args = parser.parse_args()
    
    # real_folder = args.real_folder
    # generated_folder = args.generated_folder
    # output_path = args.output_path
    
    # 计算并保存指标结果
    # compute_and_save_metrics(real_folder, generated_folder, device=device, output_path=output_path)

    real_folder = '/home/hyl/yujia/A_few_shot/evaluation/test_data/real'
    generated_folder = '/home/hyl/yujia/A_few_shot/evaluation/test_data/fake'
    output_path = '/home/hyl/yujia/A_few_shot/evaluation/results.json'
    # 计算并保存指标结果
    compute_and_save_metrics(real_folder, generated_folder, device=device, output_path=output_path)
