import os
import json
from abc import ABC, abstractmethod
from PIL import Image
from tqdm import tqdm
import torch
# torch.cuda.empty_cache()
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import lpips
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import shutil
import tempfile
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
import itertools
import logging


# 设置日志记录
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Metric(ABC):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32):
        self.real_folder = real_folder
        self.generated_folder = generated_folder
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Dataset that loads images from a folder and its subdirectories.
        
        Args:
            folder_path (str): Path to the root folder containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []

        # Recursively collect all image files from subdirectories
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.jpg', 'jpeg', '.png')):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            # Load the image and convert to RGB
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as a placeholder in case of an error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Returning image and a dummy label (0), as no actual labels are needed
        return image, 0

class FID_Global(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=5, dims=2048):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.dims = dims
        self.score = None

    def flatten_images(self, source_folder, target_folder):
        """
        Flattens the folder structure and copies all images to a target folder.
        Ensures no file name conflicts by renaming if necessary.
        """
        for root, _, files in os.walk(source_folder):
            for f in files:
                if f.lower().endswith(('.jpg', 'jpeg', '.png')):
                    src_path = os.path.join(root, f)
                    dst_path = os.path.join(target_folder, f)
                    # Handle name conflicts
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(f)
                        dst_path = os.path.join(target_folder, f"{base}_{hash(src_path)}{ext}")
                    # Debug: Print the image being copied
                    # print(f"Copying image: {src_path} to {dst_path}")
                    shutil.copy(src_path, dst_path)

    def compute(self):
        # Flatten and collect real and generated images in temp directories
        with tempfile.TemporaryDirectory() as temp_real, tempfile.TemporaryDirectory() as temp_generated:
            print("Flattening real images...")
            self.flatten_images(self.real_folder, temp_real)
            print("Flattening generated images...")
            self.flatten_images(self.generated_folder, temp_generated)

            # Debug: List files in temp directories to ensure correct file copying
            real_images = [f for f in os.listdir(temp_real) if f.lower().endswith(('.jpg', 'jpeg', '.png'))]
            generated_images = [f for f in os.listdir(temp_generated) if f.lower().endswith(('.jpg', 'jpeg', '.png'))]
            
            num_real_images = len(real_images)
            num_generated_images = len(generated_images)

            print(f"Number of real images in temp folder: {num_real_images}")
            print(f"Number of generated images in temp folder: {num_generated_images}")

            if num_real_images == 0 or num_generated_images == 0:
                print("Error: After flattening, one of the temporary folders has no images.")
                self.score = None
                return

            # Adjust batch size
            adjusted_batch_size = min(self.batch_size, num_real_images, num_generated_images)
            if adjusted_batch_size <= 0:
                print("Error: batch_size is set to 0.")
                self.score = None
                return
            else:
                print(f"Using batch_size={adjusted_batch_size} for FID computation.")

            try:
                # Compute FID score
                fid_value = fid_score.calculate_fid_given_paths(
                    [temp_real, temp_generated],
                    batch_size=adjusted_batch_size,
                    device=self.device,
                    dims=self.dims,
                    num_workers=0  # Set to 0 to avoid multi-processing issues
                )
                self.score = fid_value
                print(f"Global FID = {fid_value}")
            except ValueError as e:
                print(f"Error during FID computation: {e}")
                self.score = None

    def get_score(self):
        return self.score if self.score is not None else None

class InceptionScore(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.mean = None
        self.std = None
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def compute(self):
        # 获取排序后的类别文件夹列表
        class_folders = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])

        if not class_folders:
            print("在生成文件夹中未找到任何类别文件夹。")
            return

        per_class_scores = {}

        for class_name in class_folders:
            print(f"正在处理类别: {class_name}")
            generated_class_folder = os.path.join(self.generated_folder, class_name)
            image_files = [
                f for f in os.listdir(generated_class_folder)
                if f.lower().endswith(('.jpg', 'jpeg', '.png'))
            ]

            if not image_files:
                print(f"类别 '{class_name}' 中未找到任何图片。跳过该类别。")
                continue

            generated_images = []
            for image_file in image_files:
                image_path = os.path.join(generated_class_folder, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    generated_images.append(image)
                except Exception as e:
                    print(f"加载图片 {image_path} 时出错: {e}")

            if not generated_images:
                print(f"类别 '{class_name}' 中未找到任何有效图片。跳过该类别。")
                continue

            # 对当前类别进行批量处理
            preds_list = []
            for i in tqdm(range(0, len(generated_images), self.batch_size),
                          desc=f"计算类别 '{class_name}' 的 Inception Score"):
                batch = torch.cat(generated_images[i:i+self.batch_size], dim=0)
                with torch.no_grad():
                    preds = self.model(batch)
                    preds_list.append(preds)

            preds = torch.cat(preds_list, dim=0)
            p_y = F.softmax(preds, dim=1)
            p_y_mean = torch.mean(p_y, dim=0)

            kl_div = p_y * (torch.log(p_y) - torch.log(p_y_mean))
            kl_div = torch.sum(kl_div, dim=1)

            is_score = torch.exp(torch.mean(kl_div)).item()
            is_std = torch.exp(torch.std(kl_div)).item()

            per_class_scores[class_name] = is_score
            print(f"类别 '{class_name}' 的 Inception Score: mean={is_score:.4f}, std={is_std:.4f}")

        if per_class_scores:
            # 计算所有类别的平均 IS 和标准差
            is_values = list(per_class_scores.values())
            mean_is = sum(is_values) / len(is_values)
            std_is = (sum([(x - mean_is) ** 2 for x in is_values]) / len(is_values)) ** 0.5

            self.mean = mean_is
            self.std = std_is
            print(f"\n所有类别的平均 Inception Score: mean={self.mean:.4f}, std={self.std:.4f}")

            # 可选：将每个类别的评分和总体评分保存到文件
            save_path = os.path.join(self.generated_folder, "inception_scores.txt")
            try:
                with open(save_path, 'w') as f:
                    for class_name, score in per_class_scores.items():
                        f.write(f"类别 '{class_name}': IS={score:.4f}\n")
                    f.write(f"\n所有类别的平均 Inception Score: mean={self.mean:.4f}, std={self.std:.4f}\n")
                print(f"Inception scores 已保存到 {save_path}")
            except Exception as e:
                print(f"保存 Inception scores 时出错: {e}")
        else:
            print("未计算任何 Inception scores。")

    def get_score(self):
        return {"mean": self.mean, "std": self.std}

class MMD(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, gamma_list=None):
        """
        Maximum Mean Discrepancy (MMD) metric with multi-scale RBF kernels.

        Args:
            real_folder (str): Path to the folder containing real images.
            generated_folder (str): Path to the folder containing generated images.
            device (str): Device to use ('cuda' or 'cpu').
            batch_size (int): Batch size for DataLoader.
            gamma_list (list of float, optional): List of gamma values for RBF kernels. 
                If None, a default multi-scale list is used.
        """
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.gamma_list = gamma_list if gamma_list is not None else [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        self.score = None
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.scaler = StandardScaler()

    def compute(self):
        # Extract and standardize features
        real_features = self.extract_features(self.real_folder)
        generated_features = self.extract_features(self.generated_folder)

        # if real_features.(0) == 0 or generated_features.size(0) == 0:
        #     print("No features extracted. Skipping MMD computation.")
        #     self.score = None
        #     return

        # Standardize features
        self.scaler.fit(real_features)
        real_features = self.scaler.transform(real_features)
        generated_features = self.scaler.transform(generated_features)

        # Convert to torch tensors
        real_features = torch.tensor(real_features).to(self.device)
        generated_features = torch.tensor(generated_features).to(self.device)

        # Compute MMD with multi-scale RBF kernels
        mmd_squared = 0
        for gamma in self.gamma_list:
            kernel_real = torch.exp(-gamma * torch.cdist(real_features, real_features, p=2) ** 2)
            kernel_generated = torch.exp(-gamma * torch.cdist(generated_features, generated_features, p=2) ** 2)
            kernel_cross = torch.exp(-gamma * torch.cdist(real_features, generated_features, p=2) ** 2)

            mmd_squared += (kernel_real.mean() + kernel_generated.mean() - 2 * kernel_cross.mean())

        self.score = mmd_squared.item()
        print(f"Global MMD = {self.score}")

    def extract_features(self, folder_path):
        dataset = ImageDataset(folder_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_features = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f'Extracting features from {folder_path}'):
                images = images.to(self.device)
                features = self.model(images)
                # 使用 InceptionV3 的最后一层池化层的输出作为特征
                if isinstance(features, torch.Tensor):
                    # For InceptionV3, the output is the logits; to get features from pool3, use 'Mixed_7c' layer or adapt as needed
                    # Alternatively, modify the model to output features from an intermediate layer
                    # Here, as a placeholder, we use the output logits
                    features = features
                all_features.append(features.cpu())

        if all_features:
            all_features = torch.cat(all_features, dim=0)
            return all_features.numpy()
        return np.array([])

    def get_score(self):
        return self.score if self.score is not None else None

class KID(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.gamma = gamma
        self.score = None
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        # InceptionV3 expects 299x299 images
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute(self):
        real_features = self.extract_features(self.real_folder)
        generated_features = self.extract_features(self.generated_folder)

        if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
            print("No features extracted. Skipping KID computation.")
            self.score = None
            return

        kid_value = self.compute_kid(real_features, generated_features)
        self.score = kid_value
        print(f"Global KID = {kid_value}")

    def extract_features(self, folder_path):
        dataset = ImageDataset(folder_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_features = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f'Extracting features from {folder_path}'):
                images = images.to(self.device)
                features = self.model(images)
                all_features.append(features.cpu())

        if all_features:
            all_features = torch.cat(all_features, dim=0).numpy()
            return all_features
        return np.array([])

    def compute_kid(self, real_features, generated_features):
        kernel_real = self.compute_rbf_kernel_matrix(real_features, real_features)
        kernel_generated = self.compute_rbf_kernel_matrix(generated_features, generated_features)
        kernel_cross = self.compute_rbf_kernel_matrix(real_features, generated_features)

        mmd = np.mean(kernel_real) + np.mean(kernel_generated) - 2 * np.mean(kernel_cross)
        return mmd

    def compute_rbf_kernel_matrix(self, X, Y, gamma=1.0):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim == 3:
            Y = Y.reshape(Y.shape[0], -1)

        return rbf_kernel(X, Y, gamma=gamma)

    def get_score(self):
        return self.score if self.score is not None else None

class CLIPSimilarity(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.model, self.preprocess = self.load_clip_model()
        self.model.eval()
        self.scores = {}

    def load_clip_model(self):
        model = CLIPModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to(self.device)
        preprocess = CLIPProcessor.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        return model, preprocess

    def compute(self):
        real_classes = sorted([
            d for d in os.listdir(self.real_folder)
            if os.path.isdir(os.path.join(self.real_folder, d))
        ])
        generated_classes = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])

        assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"

        for class_name in real_classes:
            real_class_folder = os.path.join(self.real_folder, class_name)
            generated_class_folder = os.path.join(self.generated_folder, class_name)

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

            if len(real_image_paths) < 1:
                print(f"No real images found for class '{class_name}'. Skipping.")
                continue

            real_image_path = real_image_paths[0]  # 假设每个类别只有一张真实图像

            similarities = []
            for gen_path in fake_image_paths:
                similarity = self.compute_similarity(real_image_path, gen_path)
                similarities.append(similarity)

            if similarities:
                avg_similarity = np.mean(similarities)
                self.scores[class_name] = avg_similarity
                print(f"Class '{class_name}': CLIP Similarity = {avg_similarity}")

    def compute_similarity(self, real_path, fake_path):
        real_features = self.extract_features([real_path])
        fake_features = self.extract_features([fake_path])
        similarity = F.cosine_similarity(real_features, fake_features, dim=-1).item()
        return similarity

    def extract_features(self, image_paths):
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        inputs = self.preprocess(images=images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features

    def get_score(self):
        if self.scores:
            mean_similarity = np.mean(list(self.scores.values()))
            return mean_similarity
        return None

class DINOAlignment(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.model, self.processor = self.load_dino_model()
        self.model.eval()
        self.scores = {}

    def load_dino_model(self):
        processor = ViTImageProcessor.from_pretrained('/home/hyl/yujia/dino-vits16')
        model = ViTModel.from_pretrained('/home/hyl/yujia/dino-vits16').to(self.device)
        return model, processor

    def compute(self):
        real_classes = sorted([
            d for d in os.listdir(self.real_folder)
            if os.path.isdir(os.path.join(self.real_folder, d))
        ])
        generated_classes = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])

        assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"

        for class_name in real_classes:
            real_class_folder = os.path.join(self.real_folder, class_name)
            generated_class_folder = os.path.join(self.generated_folder, class_name)

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

            if len(real_image_paths) < 1:
                print(f"No real images found for class '{class_name}'. Skipping.")
                continue

            real_image_path = real_image_paths[0]  # 假设每个类别只有一张真实图像

            alignments = []
            for gen_path in fake_image_paths:
                alignment = self.compute_alignment(real_image_path, gen_path)
                alignments.append(alignment)

            if alignments:
                avg_alignment = np.mean(alignments)
                self.scores[class_name] = avg_alignment
                print(f"Class '{class_name}': DINO Alignment = {avg_alignment}")

    def compute_alignment(self, real_path, fake_path):
        real_features = self.extract_features([real_path])
        fake_features = self.extract_features([fake_path])
        alignment = F.cosine_similarity(real_features, fake_features, dim=-1).item()
        return alignment

    def extract_features(self, image_paths):
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 last_hidden_state 的均值作为特征
        features = outputs.last_hidden_state.mean(dim=1)
        return features

    def get_score(self):
        if self.scores:
            mean_alignment = np.mean(list(self.scores.values()))
            return mean_alignment
        return None

class IC_LPIPS_Diversity(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, image_size=256):
        """
        IC_LPIPS_Diversity 类用于评估生成图像的多样性。
        
        Args:
            real_folder (str): Path to the folder containing real images.
            generated_folder (str): Path to the folder containing generated images.
            device (str): Device to use ('cuda' or 'cpu').
            batch_size (int): Batch size for DataLoader.
            image_size (int): Image size for resizing.
        """
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_fn.eval()
        self.total_diversity = 0
        self.total_classes = 0
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def compute(self):
        # 获取所有类别
        real_classes = sorted([
            d for d in os.listdir(self.real_folder)
            if os.path.isdir(os.path.join(self.real_folder, d))
        ])
        generated_classes = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])

        # 确保真实和生成文件夹中的类别一致
        assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"

        for class_name in real_classes:
            real_class_folder = os.path.join(self.real_folder, class_name)
            generated_class_folder = os.path.join(self.generated_folder, class_name)

            if not os.path.isdir(real_class_folder) or not os.path.isdir(generated_class_folder):
                logger.warning(f"Skipping class '{class_name}' as it does not exist in both folders.")
                continue

            generated_images = [
                os.path.join(generated_class_folder, f) for f in os.listdir(generated_class_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if len(generated_images) < 2:
                logger.warning(f"Not enough generated images to compute diversity for class '{class_name}'. Skipping.")
                continue

            # 加载并预处理所有生成图像
            try:
                generated_tensors = []
                for img_path in generated_images:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    generated_tensors.append(img_tensor)
                generated_tensors = torch.cat(generated_tensors, dim=0)  # 形状: [N, C, H, W]
                generated_tensors = generated_tensors * 2 - 1  # 缩放到 [-1, 1]
            except Exception as e:
                logger.error(f"Error loading generated images for class '{class_name}': {e}")
                continue

            # 计算所有唯一对的 LPIPS 距离
            num_images = generated_tensors.size(0)
            diversity_scores = []

            # 生成所有唯一对的索引
            pairs = list(itertools.combinations(range(num_images), 2))
            num_pairs = len(pairs)

            if num_pairs == 0:
                logger.warning(f"No pairs to compute for class '{class_name}'. Skipping.")
                continue

            # 批量处理对
            batch_size = self.batch_size
            for i in tqdm(range(0, num_pairs, batch_size), desc=f"Computing diversity for class '{class_name}'"):
                batch_pairs = pairs[i:i+batch_size]
                img1_indices, img2_indices = zip(*batch_pairs)
                img1 = generated_tensors[list(img1_indices)].to(self.device)
                img2 = generated_tensors[list(img2_indices)].to(self.device)
                
                with torch.no_grad():
                    scores = self.loss_fn(img1, img2)
                diversity_scores.extend(scores.cpu().numpy().flatten())

            # 计算该类别的平均多样性
            class_diversity = np.mean(diversity_scores)
            logger.info(f"Class '{class_name}': Diversity (Average LPIPS) = {class_diversity}")
            self.total_diversity += class_diversity
            self.total_classes += 1

        if self.total_classes > 0:
            global_diversity = self.total_diversity / self.total_classes
            logger.info(f"Global IC-LPIPS Diversity = {global_diversity}")
        else:
            logger.warning("No classes with sufficient generated images to compute diversity.")
            global_diversity = None

        self.global_diversity = global_diversity

    def get_score(self):
        return self.global_diversity if hasattr(self, 'global_diversity') else None
class IC_LPIPS(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, image_size=256):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_fn.eval()
        self.total_score = 0
        self.total_count = 0
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def compute(self):
        real_classes = sorted([
            d for d in os.listdir(self.real_folder)
            if os.path.isdir(os.path.join(self.real_folder, d))
        ])
        generated_classes = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])

        assert real_classes == generated_classes, "Real and Generated folders do not have the same classes!"

        for class_name in real_classes:
            real_class_folder = os.path.join(self.real_folder, class_name)
            generated_class_folder = os.path.join(self.generated_folder, class_name)

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

            if len(real_images) < 1:
                print(f"No real images found for class '{class_name}'. Skipping.")
                continue

            real_image_path = real_images[0]  # 假设每个类别只有一张真实图像

            for gen_img_path in tqdm(generated_images, desc=f"Computing IC-LPIPS for class '{class_name}'"):
                try:
                    real_image = Image.open(real_image_path).convert("RGB")
                    generated_image = Image.open(gen_img_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading images '{real_image_path}' or '{gen_img_path}': {e}")
                    continue

                # 应用转换
                real_image = self.transform(real_image).unsqueeze(0).to(self.device)
                generated_image = self.transform(generated_image).unsqueeze(0).to(self.device)

                # LPIPS 需要图像在 [-1, 1] 范围
                real_image_tensor = real_image * 2 - 1
                generated_image_tensor = generated_image * 2 - 1

                score = self.loss_fn(real_image_tensor, generated_image_tensor)
                self.total_score += score.item()
                self.total_count += 1

        if self.total_count > 0:
            avg_score = self.total_score / self.total_count
            print(f"Global IC-LPIPS = {avg_score}")

    def get_score(self):
        if self.total_count > 0:
            avg_ic_lpips = self.total_score / self.total_count
            return avg_ic_lpips
        return None

class MetricsEvaluator:
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32):
        self.real_folder = real_folder
        self.generated_folder = generated_folder
        self.device = device
        self.batch_size = batch_size
        self.metrics = []

    def add_metric(self, metric_class, **kwargs):
        metric = metric_class(
            real_folder=self.real_folder,
            generated_folder=self.generated_folder,
            device=self.device,
            batch_size=self.batch_size,
            **kwargs
        )
        self.metrics.append(metric)

    def compute_all(self):
        results = {}
        for metric in self.metrics:
            metric.compute()
            score = metric.get_score()
            metric_name = metric.__class__.__name__
            results[metric_name] = score
        return results

def save_results_to_json(results, file_path):
    results_serializable = {k: convert_to_float(v) if not isinstance(v, dict) else {sub_k: convert_to_float(sub_v) for sub_k, sub_v in v.items()} for k, v in results.items()}
    
    with open(file_path, 'w') as json_file:
        json.dump(results_serializable, json_file, indent=4)
    print(f"Results saved to {file_path}")

def convert_to_float(value):
    if isinstance(value, np.float32) or isinstance(value, np.float64):
        return float(value)
    elif isinstance(value, torch.Tensor):
        return float(value.item())
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    return value

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute image quality metrics between real and generated images.')
    parser.add_argument('--real_folder', type=str, required=True, help='Path to the folder containing real images.')
    parser.add_argument('--generated_folder', type=str, required=True, help='Path to the folder containing generated images.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the results JSON.')
    args = parser.parse_args()
    
    real_folder = args.real_folder
    generated_folder = args.generated_folder
    output_path = args.output_path
    
    evaluator = MetricsEvaluator(real_folder, generated_folder, device=device, batch_size=32)
    
    # 添加全局指标和适用于one-shot的指标
    evaluator.add_metric(FID_Global, dims=2048)
    evaluator.add_metric(InceptionScore)
    # evaluator.add_metric(MMD)
    evaluator.add_metric(CLIPSimilarity)
    evaluator.add_metric(DINOAlignment)
    evaluator.add_metric(IC_LPIPS_Diversity)
    # evaluator.add_metric(IC_LPIPS, image_size=256)  # 设置 IC-LPIPS 的图像尺寸为224x224
    # evaluator.add_metric(KID, gamma=1.0)
    
    # 计算所有指标
    results = evaluator.compute_all()
    
    # 保存结果
    save_results_to_json(results, output_path)

if __name__ == "__main__":
    main()
