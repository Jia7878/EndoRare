import os
import json
from abc import ABC, abstractmethod
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
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

import warnings
warnings.filterwarnings("ignore")

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
            # 返回一个全黑图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        return image, 0  # 标签为0，因不需要实际标签

class FID(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=50, dims=2048):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.dims = dims
        self.scores = {}
    
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
            
            fid_value = fid_score.calculate_fid_given_paths(
                [real_class_folder, generated_class_folder],
                batch_size=self.batch_size,
                device=self.device,
                dims=self.dims
            )
            self.scores[class_name] = fid_value
            print(f"Class '{class_name}': FID = {fid_value}")
    
    def get_score(self):
        if self.scores:
            average_fid = sum(self.scores.values()) / len(self.scores)
            return average_fid
        return None

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def compute(self):
        class_folders = sorted([
            d for d in os.listdir(self.generated_folder)
            if os.path.isdir(os.path.join(self.generated_folder, d))
        ])
        
        is_scores = []
        
        for class_name in class_folders:
            generated_class_folder = os.path.join(self.generated_folder, class_name)
            
            if not os.path.isdir(generated_class_folder):
                print(f"Skipping class '{class_name}' as it does not exist in the generated folder.")
                continue
            
            image_files = [
                f for f in os.listdir(generated_class_folder)
                if f.lower().endswith(('.jpg', 'jpeg', '.png'))
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
                image = self.transform(image).unsqueeze(0).to(self.device)
    
                with torch.no_grad():
                    preds = self.model(image)
                    preds_list.append(preds)
            
            if not preds_list:
                print(f"No valid images found for class '{class_name}'. Skipping.")
                continue
            
            preds = torch.cat(preds_list, dim=0)
            p_y = F.softmax(preds, dim=1)
            p_y_mean = torch.mean(p_y, dim=0)
            
            kl_div = p_y * (torch.log(p_y) - torch.log(p_y_mean))
            kl_div = torch.sum(kl_div, dim=1)
            
            is_score = torch.exp(torch.mean(kl_div)).item()
            is_scores.append(is_score)
            print(f"Class '{class_name}': Inception Score = {is_score}")
        
        if is_scores:
            self.mean = np.mean(is_scores)
            self.std = np.std(is_scores)
        else:
            self.mean = None
            self.std = None
    
    def get_score(self):
        return {"mean": self.mean, "std": self.std}

class MMD(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.gamma = gamma
        self.scores = {}
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
            
            real_features = self.extract_features(real_class_folder)
            generated_features = self.extract_features(generated_class_folder)
            
            if real_features.size(0) == 0 or generated_features.size(0) == 0:
                print(f"No features extracted for class '{class_name}'. Skipping.")
                continue
            
            mmd_value = self.compute_mmd(real_features, generated_features)
            self.scores[class_name] = mmd_value
            print(f"Class '{class_name}': MMD = {mmd_value}")
    
    def extract_features(self, folder_path):
        dataset = ImageDataset(folder_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        all_features = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f'Extracting features from {folder_path}'):
                images = images.to(self.device)
                features = self.model(images)
                all_features.append(features.cpu())
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            return all_features
        return torch.Tensor([])
    
    def compute_mmd(self, real_features, generated_features):
        dist_matrix = torch.cdist(real_features, generated_features, p=2)
        kernel_real = torch.exp(-self.gamma * torch.cdist(real_features, real_features, p=2) ** 2).mean()
        kernel_generated = torch.exp(-self.gamma * torch.cdist(generated_features, generated_features, p=2) ** 2).mean()
        kernel_cross = torch.exp(-self.gamma * dist_matrix ** 2).mean()
        mmd_squared = kernel_real + kernel_generated - 2 * kernel_cross
        return mmd_squared.item()
    
    def get_score(self):
        if self.scores:
            average_mmd = sum(self.scores.values()) / len(self.scores)
            return average_mmd
        return None

class KID(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, gamma=1.0):
        super().__init__(real_folder, generated_folder, device, batch_size)
        self.gamma = gamma
        self.scores = {}
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
            
            real_features = self.extract_features(real_class_folder)
            generated_features = self.extract_features(generated_class_folder)
            
            if real_features.shape[0] == 0 or generated_features.shape[0] == 0:
                print(f"No features extracted for class '{class_name}'. Skipping.")
                continue
            
            kid_value = self.compute_kid(real_features, generated_features)
            self.scores[class_name] = kid_value
            print(f"Class '{class_name}': KID = {kid_value}")
    
    def extract_features(self, folder_path):
        dataset = ImageDataset(folder_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
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
        if self.scores:
            average_kid = sum(self.scores.values()) / len(self.scores)
            return average_kid
        return None

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
            
            if len(real_image_paths) != len(fake_image_paths):
                print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
                continue
            
            paired_images = self.pair_images(real_image_paths, fake_image_paths)
            
            if not paired_images:
                print(f"No paired images found for class '{class_name}'. Skipping.")
                continue
            
            similarities = []
            for real_path, fake_path in paired_images:
                real_features = self.extract_features([real_path])
                fake_features = self.extract_features([fake_path])
                similarity = F.cosine_similarity(real_features, fake_features, dim=-1).item()
                similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                self.scores[class_name] = avg_similarity
                print(f"Class '{class_name}': CLIP Similarity = {avg_similarity}")
    
    def pair_images(self, real_images, fake_images):
        real_dict = {os.path.basename(path): path for path in real_images}
        fake_dict = {os.path.basename(path): path for path in fake_images}
        paired = []
        for filename, real_path in real_dict.items():
            if filename in fake_dict:
                paired.append((real_path, fake_dict[filename]))
        return paired
    
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
            
            if len(real_image_paths) != len(fake_image_paths):
                print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
                continue
            
            paired_images = self.pair_images(real_image_paths, fake_image_paths)
            
            if not paired_images:
                print(f"No paired images found for class '{class_name}'. Skipping.")
                continue
            
            alignments = []
            for real_path, fake_path in paired_images:
                real_features = self.extract_features([real_path])
                fake_features = self.extract_features([fake_path])
                alignment = F.cosine_similarity(real_features, fake_features, dim=-1).item()
                alignments.append(alignment)
            
            if alignments:
                avg_alignment = np.mean(alignments)
                self.scores[class_name] = avg_alignment
                print(f"Class '{class_name}': DINO Alignment = {avg_alignment}")
    
    def pair_images(self, real_images, fake_images):
        real_dict = {os.path.basename(path): path for path in real_images}
        fake_dict = {os.path.basename(path): path for path in fake_images}
        paired = []
        for filename, real_path in real_dict.items():
            if filename in fake_dict:
                paired.append((real_path, fake_dict[filename]))
        return paired
    
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

class IC_LPIPS(Metric):
    def __init__(self, real_folder, generated_folder, device='cuda', batch_size=32, image_size=224):
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
            
            if len(real_images) != len(generated_images):
                print(f"Warning: Number of images in class '{class_name}' does not match. Skipping this class.")
                continue
            
            paired_images = self.pair_images(real_images, generated_images)
            
            if not paired_images:
                print(f"No paired images found for class '{class_name}'. Skipping.")
                continue
            
            for real_img_path, gen_img_path in tqdm(paired_images, desc=f"Computing IC-LPIPS for class '{class_name}'"):
                try:
                    real_image = Image.open(real_img_path).convert("RGB")
                    generated_image = Image.open(gen_img_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading images '{real_img_path}' or '{gen_img_path}': {e}")
                    continue
                
                # Apply transformations
                real_image = self.transform(real_image).unsqueeze(0).to(self.device)
                generated_image = self.transform(generated_image).unsqueeze(0).to(self.device)
                
                # LPIPS expects images to be in [-1, 1] range
                real_image_tensor = real_image * 2 - 1
                generated_image_tensor = generated_image * 2 - 1
                
                score = self.loss_fn(real_image_tensor, generated_image_tensor)
                self.total_score += score.item()
                self.total_count += 1
            
            if self.total_count > 0:
                avg_score = self.total_score / self.total_count
                print(f"Class '{class_name}': IC-LPIPS = {avg_score}")
    
    def pair_images(self, real_images, fake_images):
        real_dict = {os.path.basename(path): path for path in real_images}
        fake_dict = {os.path.basename(path): path for path in fake_images}
        paired = []
        for filename, real_path in real_dict.items():
            if filename in fake_dict:
                paired.append((real_path, fake_dict[filename]))
        return paired
    
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
    
    # 添加指标
    evaluator.add_metric(FID)
    evaluator.add_metric(InceptionScore)
    evaluator.add_metric(MMD, gamma=1.0)
    evaluator.add_metric(CLIPSimilarity)
    evaluator.add_metric(DINOAlignment)
    evaluator.add_metric(IC_LPIPS, image_size=224)  # 设置 IC-LPIPS 的图像尺寸为224x224
    evaluator.add_metric(KID, gamma=1.0)
    
    # 计算所有指标
    results = evaluator.compute_all()
    
    # 保存结果
    save_results_to_json(results, output_path)

if __name__ == "__main__":
    main()
