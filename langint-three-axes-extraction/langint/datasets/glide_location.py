import torch
from tu.utils.config import build_from_config
import os
import json
from PIL import Image
from typing import List, Dict
import numpy as np
from torchvision import transforms as TF
import torchvision.transforms.functional as TFF
from langint.utils.dataset import imagenet_templates_small
import logging
from collections import Counter
from transformers import CLIPVisionModel
import kornia
import random
from typing import *
from collections import defaultdict

logger = logging.getLogger(__name__)

# Updated TEMPLATE to include the 'location' dimension
TEMPLATE = """polyp, {} type, {} color, {} and located at {}"""

# Increased the range to accommodate more placeholder tokens (4 groups: type, color, mat, location)
placeholder_words_list = [f'mytoken{i}' for i in range(1000)]

def deepfloyd_sample_prompts(prompts: List[str], num_repeats=4, model=None, processor=None, blip_fruit_q=None, blip_mat_q=None, blip_color_q=None, real_image=None, batch_size=None):
    from polypdiffusion.polyp_diffusion_pipeline import PolypDiffusionPipeline
    pipeline = PolypDiffusionPipeline(
        config_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml',
        ckpt_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/Text_location_215/2025-01-03T12-29-27_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=99-step=2699.ckpt',
        batch_size=batch_size,
        seed=42
    )
    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_mats = []
    blip_common_colors = []
    
    for prompt in prompts:
        assert real_image is not None, "Real image must be provided for inference."
        if real_image is None:
            images: List[Image.Image] = pipeline.get_results(prompt, count=num_repeats)
        else:
            img = Image.open(real_image).convert('RGB')
            images = [img.copy() for _ in range(num_repeats)]
        
        images_all.extend(images)
        
        # Assuming blip_common_* are handled elsewhere or are not needed here
        blip_common_fruits = []
        blip_common_mats = []
        blip_common_colors = []
            
    return torch.stack([TF.ToTensor()(image) * 2 - 1 for image in images_all]), blip_common_fruits, blip_common_mats, blip_common_colors, images_all

class SyntheticBiLevel(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_root: str,
        templates: Dict,
        num_data_per_prompt: int = 8, 
        num_data_copies: int = 1, 
        num_tokens_per_word: int = 1,
        num_placeholder_words: int = 1, 
        num_placeholder_groups: int = 4,  # Updated from 3 to 4
        shared_tokens=0, 
        real_image=None
    ): 
        """
        Initializes the SyntheticBiLevel dataset with an additional 'location' attribute.
        
        Args:
            data_root (str): The root string containing ground truth words separated by "/".
            templates (Dict): Dictionary containing template strings (unused here as TEMPLATE is predefined).
            num_data_per_prompt (int): Number of data samples to generate per prompt.
            num_data_copies (int): Number of copies for each data sample.
            num_tokens_per_word (int): Number of tokens per word (default is 1).
            num_placeholder_words (int): Number of placeholder words (default is 1).
            num_placeholder_groups (int): Number of placeholder groups (updated to 4 for 'location').
            shared_tokens (int): Flag to indicate if tokens are shared (0 or 1).
            real_image (str): Path to a real image for inference (if any).
        """
        
        assert shared_tokens in [0, 1], shared_tokens
        assert num_placeholder_groups == 4, f"num_placeholder_groups must be 4, got {num_placeholder_groups}"
        
        self.templates = TEMPLATE
    
        # Parse ground truth words from data_root
        ground_truth_words = data_root.split("/")
        ground_truth_words = [word.split('_') for word in ground_truth_words]  # e.g., [['type1', 'color1', 'mat1', 'location1'], ...]
        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        
        # Split words into their respective placeholder groups
        ground_truth_prompt_args = [[] for _ in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups, f"Each split word must have {num_placeholder_groups} elements."
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])
        
        self.ground_truth_prompt_args = ground_truth_prompt_args
        self.unique_gt_words = ground_truth_words
        for gt_word in self.unique_gt_words:
            assert len(gt_word) == num_placeholder_groups, f"Each ground truth word must have {num_placeholder_groups} elements."
    
        # Generate unique prompts using the TEMPLATE
        unique_prompts = []
        for ind in range(num_placeholder_words):
            curr_prompt_words = []
            for ground_truth_prompt_arg in ground_truth_prompt_args:
                curr_prompt_words.append(ground_truth_prompt_arg[ind])
            prompt = self.templates.format(*curr_prompt_words) 
            unique_prompts.append(prompt)
        self.gt_prompts: List[str] = unique_prompts
        
        # Generate images using deepfloyd_sample_prompts
        self.images, _, _, _, pil_images = deepfloyd_sample_prompts(
            unique_prompts, 
            num_repeats=num_data_per_prompt, 
            real_image=real_image
        )
    
        # Initialize CLIP Vision Model
        clip_vision = CLIPVisionModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to("cuda").requires_grad_(False)
    
        # Define CLIP preprocessing
        def clip_preprocess(x):
            x = kornia.geometry.resize(
                x, 
                (clip_vision.config.image_size, clip_vision.config.image_size), 
                interpolation='bicubic', 
                align_corners=True, 
                antialias=False
            )
            x = (x + 1.) / 2.
            # Renormalize according to CLIP's requirements
            x = kornia.enhance.normalize(
                x, 
                torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(x.device), 
                torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(x.device)
            )
            return x
    
        # Preprocess images for CLIP
        preprocessed_images = self.images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)
    
        # Extract CLIP features
        self.clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(
                    pixel_values=img.unsqueeze(0).expand(4, -1, -1, -1), 
                    output_hidden_states=True
                )
                # Extract specific hidden states
                result = result.hidden_states[1:][1::2]
                # Apply post_layernorm and move to CPU
                result = [
                    clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") 
                    for hidden_states in result
                ]
                self.clip_features.append(torch.stack(result, dim=1).type(torch.float32).to("cpu"))
        # Validate shapes
        for i in range(len(self.clip_features)):
            assert self.clip_features[i].shape == (4, 12, 1024), f"Expected shape (4, 12, 1024), got {self.clip_features[i].shape}"
        assert len(self.clip_features) == len(self.images), (len(self.clip_features), len(self.images))
    
        # Clean up
        del clip_vision
        torch.cuda.empty_cache()
    
        # Initialize placeholder words
        self.num_placeholder_words = num_placeholder_words * num_tokens_per_word
        # Create placeholder tokens for all groups (4 groups: type, color, mat, location)
        placeholder_words = []
        for i in range(num_placeholder_words):
            for j in range(num_placeholder_groups):
                placeholder_words.append(f'mytoken{j + num_placeholder_groups * i}')  # e.g., mytoken0, mytoken1, mytoken2, mytoken3, mytoken4, ...
    
        # Split and transpose to get prompt args
        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words * num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(
            np.split(np.array(placeholder_words), num_placeholder_words),
            (1, 0, 2)
        )
    
        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, f"Each placeholder words prompt arg must have {num_placeholder_words} words."
        
        # Assign placeholder words to all samples
        self.ph_words_all = []
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)  # Replicate as per num_data_per_prompt
    
        self.placeholder_words_prompt_args = placeholder_words_prompt_args 
        unique_ph_words = []
        num_placeholder_words_actual = len(self.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words_actual):
            curr_ph_words = []
            for placeholder_words_prompt_arg in self.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])
    
        self.num_data_copies = num_data_copies
        self.num_data_per_prompt = num_data_per_prompt
    
    def __len__(self):
        return len(self.images) * self.num_data_copies
    
    def __getitem__(self, item):
        """
        Retrieves a single data sample.
        
        Args:
            item (int): Index of the data sample.
        
        Returns:
            dict: A dictionary containing the image, prompt, ground truth prompt, and CLIP features.
        """
        # Account for multiple copies
        item = item % len(self.images)
        image: torch.Tensor = self.images[item]
        
        # Data augmentation: random horizontal flip
        if np.random.rand() < 0.5:
            image = TFF.hflip(image)
    
        # Get placeholder words for the current sample
        curr_ph_words = self.ph_words_all[item]
    
        # Generate prompt using the placeholder words
        prompt = self.templates.format(*[''.join(word) for word in curr_ph_words])
    
        # Get corresponding CLIP features
        clip_feature = self.clip_features[item]
    
        return {
            'image': image, 
            'prompt': prompt, 
            'gt_prompt': self.gt_prompts[item // self.num_data_per_prompt], 
            'clip_feature': clip_feature
        }

class SyntheticBiLevelEval(SyntheticBiLevel):
    def __init__(
        self, 
        data_root: str, 
        num_placeholder_words: int, 
        templates: Dict, 
        ref_dataset: SyntheticBiLevel
    ):
        """
        Initializes the SyntheticBiLevelEval dataset for evaluation purposes.
        
        Args:
            data_root (str): Not used directly but retained for consistency.
            num_placeholder_words (int): Number of placeholder words.
            templates (Dict): Dictionary containing template strings (unused here as TEMPLATE is predefined).
            ref_dataset (SyntheticBiLevel): Reference dataset to inherit properties from.
        """
        # Define unique placeholder words for evaluation (assuming a single set)
        unique_ph_words = [['mytoken0', 'mytoken1', 'mytoken2', 'mytoken3']]
        
        self.val_batch_size = 4
        assert len(ref_dataset.images) == 1, f"Reference dataset should have exactly one image, got {len(ref_dataset.images)}"
        self.image = ref_dataset.images[0].clone()
        
        self.gt_word_pairs = ref_dataset.unique_gt_words
        self.ph_word_pairs = unique_ph_words
        self.full_template = TEMPLATE
        self.fruit_template = 'polyp, {} type'
        self.mat_template = 'polyp, and {}'
        self.color_template = 'polyp, {} color'
        self.location_template = 'polyp, located at {}'
        
        # Extract all attributes for possible use
        self.all_gt_colors = [word_pair[1] for word_pair in self.gt_word_pairs]
        self.all_ph_colors = [word_pair[1] for word_pair in self.ph_word_pairs]
        self.all_colors = [word_pair[1] for word_pair in ref_dataset.unique_gt_words]
        self.all_gt_mats = [word_pair[2] for word_pair in self.gt_word_pairs]
        self.all_ph_mats = [word_pair[2] for word_pair in self.ph_word_pairs]
        self.all_mats = [word_pair[2] for word_pair in ref_dataset.unique_gt_words]
        self.all_gt_locations = [word_pair[3] for word_pair in self.gt_word_pairs]  # New
        self.all_ph_locations = [word_pair[3] for word_pair in self.ph_word_pairs]  # New
        self.all_locations = [word_pair[3] for word_pair in ref_dataset.unique_gt_words]  # New
    
        # Verify consistency
        assert len(self.all_ph_colors) == len(self.all_gt_colors), (len(self.all_ph_colors), len(self.all_gt_colors))
        assert len(self.all_ph_mats) == len(self.all_gt_mats), (len(self.all_ph_mats), len(self.all_gt_mats))
        assert len(self.all_ph_locations) == len(self.all_gt_locations), (len(self.all_ph_locations), len(self.all_gt_locations))
    
    def __len__(self):
        return len(self.gt_word_pairs) * self.val_batch_size
    
    def __getitem__(self, item):
        """
        Retrieves a single evaluation data sample.
        
        Args:
            item (int): Index of the data sample.
        
        Returns:
            dict: A dictionary containing the image, prompt, ground truth prompt, and individual attributes.
        """
        gt_word_pair = self.gt_word_pairs[item // self.val_batch_size]
        ph_word_pair = self.ph_word_pairs[item // self.val_batch_size]
        gt_prompt = self.full_template.format(*gt_word_pair)
        prompt = self.full_template.format(*ph_word_pair)
    
        random.seed(item)
        assert len(self.all_ph_colors) == len(self.all_gt_colors) == len(self.all_ph_mats) == len(self.all_gt_mats) == len(self.all_ph_locations) == len(self.all_gt_locations), (
            len(self.all_ph_colors),
            len(self.all_gt_colors),
            len(self.all_ph_mats),
            len(self.all_gt_mats),
            len(self.all_ph_locations),
            len(self.all_gt_locations)
        )
        
        # Sample random indices if needed (not used here but retained for structure)
        # indices = random.sample(list(range(len(self.all_gt_colors))), 1)
        # indices2 = random.sample(list(range(len(self.all_gt_colors))), 1)
        
        return {
            'image': self.image,
            'prompt': prompt,
            'gt_prompt': gt_prompt,
            'gt_fruit': gt_word_pair[0],
            'gt_color': gt_word_pair[1],
            'gt_mat': gt_word_pair[2],
            'gt_location': gt_word_pair[3],  # New
            'ph_fruit': ph_word_pair[0],
            'ph_color': ph_word_pair[1],
            'ph_mat': ph_word_pair[2],
            'ph_location': ph_word_pair[3],  # New
            'full_template': self.full_template,
            'fruit_template': self.fruit_template,
            'mat_template': self.mat_template,
            'color_template': self.color_template,
            'location_template': self.location_template,  # New
        }