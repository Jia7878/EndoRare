import json
import torch
from tu.utils.config import build_from_config
import os
from PIL import Image
from typing import *
import numpy as np
import torchvision.transforms.functional as TF
from langint.utils.dataset import imagenet_templates_small
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter, defaultdict
from transformers import CLIPVisionModel
import kornia
import random

logger = logging.getLogger(__name__)

# Increased the range to accommodate the additional 'location' tokens
placeholder_words_list = [f'mytoken{i}' for i in range(1200)]  # ['mytoken0', 'mytoken1', ..., 'mytoken799']

# Updated TEMPLATE to include the 'location' dimension
TEMPLATE = """polyp, {} type, {} color, {} and located at {}"""

def deepfloyd_sample_prompts(prompts: List[str], num_repeats=1, model=None, processor=None, blip_fruit_q=None, blip_mat_q=None, blip_color_q=None, batch_size=None):
    # from langint.utils.deepfloyd_no_diffusers import Pipeline
    from polypdiffusion.polyp_diffusion_pipeline import PolypDiffusionPipeline
    pipeline = PolypDiffusionPipeline(
        config_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml',
        ckpt_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/Text_location_215/2025-01-03T12-29-27_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=11-step=299.ckpt',
        batch_size=batch_size,
        seed=42)
    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_mats = []
    blip_common_colors = []
    for prompt in prompts:
        images: List[Image.Image] = pipeline.get_results(prompt, count=num_repeats)
        images_all.extend(images)

    del pipeline
    processed_images = []
    for image in images_all:
        if isinstance(image, torch.Tensor):
            processed_image = image * 2 - 1
        else:
            processed_image = TF.to_tensor(image) * 2 - 1
        processed_images.append(processed_image)

    return torch.stack(processed_images), blip_common_fruits, blip_common_mats, blip_common_colors, images_all


class SyntheticBiLevel(torch.utils.data.Dataset):
    IMAGE_FILE_ROOT = "/home/hyl/yujia/A_few_shot/data_text_private/image_215_256"
    GT_JSON_FILE = "/home/hyl/yujia/A_few_shot/data_text_private/metadata_location.json"

    def __init__(self, num_data_per_prompt: int = 8, num_data_copies: int = 1, num_tokens_per_word: int = 1,
                 num_placeholder_words: int = 215, num_placeholder_groups: int = 4, shared_tokens=0, *args, **kwargs):
        """
        Initialize the SyntheticBiLevel dataset with an additional 'location' attribute.
        """
        # Load images and ground truth labels, including 'location'
        self.images, self.blip_fruits, self.blip_mats, self.blip_colors, self.blip_locations = self.process_gt_labels()

        # Load CLIP model and process clip features
        self.clip_vision = CLIPVisionModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to("cuda").requires_grad_(False)
        self.clip_features = self.process_clip_features()

        # Inference prompts processing
        inference_input = '0-IIa - white - tubular adenoma - ascending colon'
        inf_data = inference_input.split(",")  # e.g., ['0-IIa - white - tubular adenoma - ascending colon']
        inf_data = [word.replace('_', " ") for word in inf_data]
        inf_data = [word.split(' - ') for word in inf_data]  # Each item should now have 4 elements

        assert all(len(item) == 4 for item in inf_data), "Each attribute must have 4 components (type, color, mat, location)"

        inf_ph_tokens = [
            [f'mytoken{4*i}', f'mytoken{4*i + 1}', f'mytoken{4*i + 2}', f'mytoken{4*i + 3}']
            for i in range(len(inf_data))
        ]

        self.inf_gt_prompts = [TEMPLATE.format(*pair) for pair in inf_data]
        self.inf_prompts = [TEMPLATE.format(*inf_ph_tokens[i]) for i in range(len(inf_data))]

        self.inf_fruit_prompts = ['polyp, {} type'.format(pair[0]) for pair in inf_ph_tokens]
        self.inf_color_prompts = ['polyp, {} color'.format(pair[1]) for pair in inf_ph_tokens]
        self.inf_mat_prompts = ['polyp, {}'.format(pair[2]) for pair in inf_ph_tokens]
        self.inf_location_prompts = ['polyp, located at {}'.format(pair[3]) for pair in inf_ph_tokens]

        self.inf_images, _, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1, batch_size=1)

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = self._clip_preprocess(preprocessed_images)

        run_count = 0
        self.inf_clip_features = []  # List to store features for each image
        for img in preprocessed_images:
            with torch.no_grad():
                result = self.clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [self.clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.inf_clip_features.append(torch.stack(result, dim=1).type(torch.float32).to("cpu"))
                run_count += 1
        self.inf_clip_features = [feat for feat in self.inf_clip_features]

        for i in range(len(self.inf_clip_features)):
            assert self.inf_clip_features[i].shape == (3, 12, 1024)
        assert len(self.inf_clip_features) == len(self.inf_images), (len(self.inf_clip_features), len(self.inf_images))

        self.inf_dict = {
            'image': [img for img in self.inf_images],
            'prompt': self.inf_prompts,
            'gt_prompt': self.inf_gt_prompts,
            'fruit_prompt': self.inf_fruit_prompts,
            'mat_prompt': self.inf_mat_prompts,
            'color_prompt': self.inf_color_prompts,
            'location_prompt': self.inf_location_prompts,  # New
            'clip_feature': [feat for feat in self.inf_clip_features],
        }

        torch.cuda.empty_cache()

        self.num_placeholder_words = num_placeholder_words * num_tokens_per_word  # 215 * 1
        placeholder_words = []  # Store all placeholder tokens

        # Initialize dictionaries for shared tokens
        type_dict = {}
        color_dict = {}
        mat_dict = {}
        location_dict = {}  # New
        type_count = 0
        color_count = 1
        mat_count = 2
        location_count = 3  # New

        ground_truth_words = self.get_ground_truth_words()

        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        ground_truth_prompt_args = [[] for _ in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups, (len(split_word), num_placeholder_groups)
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])

        self.ground_truth_prompt_args = ground_truth_prompt_args
        self.unique_gt_words = ground_truth_words
        for gt_word in self.unique_gt_words:
            assert len(gt_word) == num_placeholder_groups

        unique_prompts = []
        for ind in range(num_placeholder_words):
            curr_prompt_words = []
            for ground_truth_prompt_arg in ground_truth_prompt_args:
                curr_prompt_words.append(ground_truth_prompt_arg[ind])
            prompt = TEMPLATE.format(*curr_prompt_words)
            unique_prompts.append(prompt)
        self.gt_prompts: List[str] = unique_prompts

        for i in range(len(ground_truth_words)):
            # Type
            curr_type = ground_truth_words[i][0].split()[-1]
            if shared_tokens == 1:
                if curr_type not in type_dict:
                    type_dict[curr_type] = f'mytoken{type_count}'
                    type_count += 4
                placeholder_words.append(type_dict[curr_type])
            else:
                placeholder_words.append(f'mytoken{type_count}')
                type_count += 4

            # Color
            curr_color = ground_truth_words[i][1].split()[-1]
            if shared_tokens == 1:
                if curr_color not in color_dict:
                    color_dict[curr_color] = f'mytoken{color_count}'
                    color_count += 4
                placeholder_words.append(color_dict[curr_color])
            else:
                placeholder_words.append(f'mytoken{color_count}')
                color_count += 4

            # Mat
            curr_mat = ground_truth_words[i][2].split()[-1]
            if shared_tokens == 1:
                if curr_mat not in mat_dict:
                    mat_dict[curr_mat] = f'mytoken{mat_count}'
                    mat_count += 4
                placeholder_words.append(mat_dict[curr_mat])
            else:
                placeholder_words.append(f'mytoken{mat_count}')
                mat_count += 4

            # Location
            curr_location = ground_truth_words[i][3].split()[-1]
            if shared_tokens == 1:
                if curr_location not in location_dict:
                    location_dict[curr_location] = f'mytoken{location_count}'
                    location_count += 4
                placeholder_words.append(location_dict[curr_location])
            else:
                placeholder_words.append(f'mytoken{location_count}')
                location_count += 4

        # Split placeholder words into groups
        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words * num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(
            np.split(np.array(placeholder_words), num_placeholder_words),
            (1, 0, 2)
        )

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)

        self.ph_words_all = []
        """
          Example:
          placeholder_words_prompt_args = [ 
            [
                ['mytoken0'], ['mytoken4'], ['mytoken8'], ['mytoken12']
            ], 
            [
                ['mytoken1'], ['mytoken5'], ['mytoken9'], ['mytoken13']
            ], 
            [
                ['mytoken2'], ['mytoken6'], ['mytoken10'], ['mytoken14']
            ],
            [
                ['mytoken3'], ['mytoken7'], ['mytoken11'], ['mytoken15']
            ]
        ] (4 x num_placeholder_words)
          And you have 4 samples
        """
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

        # Process BLIP fruits
        blip_fruit_for_each_ph = defaultdict(list)
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)
        for i in range(len(ground_truth_words)):
            ph_fruit = unique_ph_words[i][0]
            blip_fruit = self.blip_fruits[i]
            blip_fruit_for_each_ph[ph_fruit].append(blip_fruit)

        common_blip_fruit_for_each_ph = {}
        for ph_fruit in blip_fruit_for_each_ph:
            blip_fruit_counter = Counter(blip_fruit_for_each_ph[ph_fruit])
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            common_blip_fruit_for_each_ph[ph_fruit] = blip_common_fruit

        self.blip_fruits = [common_blip_fruit_for_each_ph[ph_pair[0]] for ph_pair in unique_ph_words]
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)

        self.num_data_copies = num_data_copies
        self.num_data_per_prompt = num_data_per_prompt

    def process_gt_labels(self):
        """
        Load and process ground truth labels, including 'location'.
        """
        data = json.load(open(self.GT_JSON_FILE))

        image_pathes = []
        blip_fruits = []
        blip_mats = []
        blip_colors = []
        blip_locations = []  # New

        for image_path, attributes in data.items():
            attributes_str = attributes['Beard_and_Age'].lower()  # Modify if the key changes
            attributes_list = [item.strip() for item in attributes_str.split(' - ')]
            assert len(attributes_list) == 4, "The attributes number is incorrect!"  # Updated to 4

            blip_fruits.append(attributes_list[0])
            blip_colors.append(attributes_list[1])
            blip_mats.append(attributes_list[2])
            blip_locations.append(attributes_list[3])  # New

            image_pathes.append(os.path.join(self.IMAGE_FILE_ROOT, image_path))

        images = self.process_images(image_pathes)

        return images, blip_fruits, blip_mats, blip_colors, blip_locations  # Updated

    def process_images(self, image_pathes: List[str]):
        """
        Load and preprocess images.
        """
        images_all = []

        for image_path in image_pathes:
            image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
            images_all.append(TF.to_tensor(image) * 2 - 1)  # Normalize to [-1, 1]

        return torch.stack(images_all)

    def process_clip_features(self):
        """
        Process images through CLIP to obtain clip features.
        """
        preprocessed_images = self.images.to("cuda")
        preprocessed_images = self._clip_preprocess(preprocessed_images)

        clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = self.clip_vision(pixel_values=img.unsqueeze(0).expand(4, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [self.clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                clip_features.append(torch.stack(result, dim=1).type(torch.float32).to("cpu"))

        # Validate shapes
        for i in range(len(clip_features)):
            assert clip_features[i].shape == (4, 12, 1024)
        assert len(clip_features) == len(self.images), (len(clip_features), len(self.images))

        return clip_features

    def _clip_preprocess(self, x):
        """
        Preprocess images for CLIP.
        """
        x = kornia.geometry.resize(
            x, (self.clip_vision.config.image_size, self.clip_vision.config.image_size),
            interpolation='bicubic', align_corners=True, antialias=False
        )
        x = (x + 1.) / 2.
        # Renormalize according to CLIP
        x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return x

    def process_prompt(self, blip_fruit, blip_color, blip_mat, blip_location):
        """
        Generate ground truth prompt with all four attributes.
        """
        return TEMPLATE.format(blip_fruit, blip_color, blip_mat, blip_location)

    def get_ground_truth_words(self):
        """
        Retrieve ground truth words, including 'location'.
        """
        data = json.load(open(self.GT_JSON_FILE))

        ground_truth_words = []
        for _, attributes in data.items():
            attributes_str = attributes['Beard_and_Age'].lower()  # Modify if key changes
            attributes_list = [item.strip() for item in attributes_str.split(' - ')]
            assert len(attributes_list) == 4, "The attributes number is incorrect!"  # Updated to 4

            ground_truth_words.append([attributes_list[0], attributes_list[1], attributes_list[2], attributes_list[3]])

        return ground_truth_words

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Any:
        """
        Retrieve a single data sample.
        """
        image = self.images[index]
        if np.random.rand() < 0.5:
            image = TF.hflip(image)

        curr_ph_words = self.ph_words_all[index]

        clip_feature = self.clip_features[index]
        blip_color = self.blip_colors[index]
        blip_mat = self.blip_mats[index]
        blip_fruit = self.blip_fruits[index]
        blip_location = self.blip_locations[index]  # New

        prompt = TEMPLATE.format(*[''.join(word) for word in curr_ph_words])
        gt_prompt = self.process_prompt(blip_fruit, blip_color, blip_mat, blip_location)  # Updated to include location

        return {
            'image': image,
            'prompt': prompt,
            'gt_prompt': gt_prompt,
            'clip_feature': clip_feature,
            'blip_color': blip_color,
            'blip_mat': blip_mat,
            'blip_fruit': blip_fruit,
            'blip_location': blip_location  # Include location
        }

class SyntheticBiLevelEval(SyntheticBiLevel):
    def __init__(self, data_root: str, num_placeholder_words: int, templates: Dict, ref_dataset: SyntheticBiLevel):
        """
        Initialize the evaluation dataset, inheriting from SyntheticBiLevel and adding 'location' handling.
        """
        # Extract unique placeholder words from reference dataset
        unique_ph_words = []
        num_placeholder_words_actual = len(ref_dataset.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words_actual):
            curr_ph_words = []
            for placeholder_words_prompt_arg in ref_dataset.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])

        self.image = torch.zeros_like(ref_dataset.images[0])
        self.gt_word_pairs = ref_dataset.unique_gt_words
        self.ph_word_pairs = unique_ph_words
        self.full_template = TEMPLATE
        self.fruit_template = 'polyp, {} type'
        self.mat_template = 'polyp, {}'
        self.color_template0 = 'polyp, {} color'
        self.color_template1 = 'polyp, {} color'
        self.color_template2 = 'polyp, {} color'
        self.location_template = 'polyp, located at {}'  # New template
        self.val_batch_size = 1

        # Extract all attributes
        self.all_gt_colors = [word_pair[1] for word_pair in self.gt_word_pairs]
        self.all_ph_colors = [word_pair[1] for word_pair in self.ph_word_pairs]
        self.all_colors = [word_pair[1] for word_pair in ref_dataset.unique_gt_words]

        self.all_gt_mats = [word_pair[2] for word_pair in self.gt_word_pairs]
        self.all_ph_mats = [word_pair[2] for word_pair in self.ph_word_pairs]
        self.all_mats = [word_pair[2] for word_pair in ref_dataset.unique_gt_words]

        self.all_gt_locations = [word_pair[3] for word_pair in self.gt_word_pairs]  # New
        self.all_ph_locations = [word_pair[3] for word_pair in self.ph_word_pairs]  # New
        self.all_locations = [word_pair[3] for word_pair in ref_dataset.unique_gt_words]  # New

        self.blip_colors = ref_dataset.blip_colors
        self.blip_mats = ref_dataset.blip_mats
        self.blip_fruits = ref_dataset.blip_fruits
        self.blip_locations = ref_dataset.blip_locations  # New

        self.inf_dict = ref_dataset.inf_dict

    def __len__(self):
        return len(self.gt_word_pairs) * self.val_batch_size

    def __getitem__(self, item):
        """
        Retrieve a single evaluation data sample.
        """
        gt_word_pair = self.gt_word_pairs[item // self.val_batch_size]
        ph_word_pair = self.ph_word_pairs[item // self.val_batch_size]
        gt_prompt = self.full_template.format(*gt_word_pair)
        prompt = self.full_template.format(*ph_word_pair)

        assert len(self.all_ph_colors) == len(self.all_gt_colors) == len(self.all_ph_mats) == len(self.all_gt_mats) == len(self.all_ph_locations) == len(self.all_gt_locations), (
            len(self.all_ph_colors),
            len(self.all_gt_colors),
            len(self.all_ph_mats),
            len(self.all_gt_mats),
            len(self.all_ph_locations),
            len(self.all_gt_locations)
        )
        random.seed(item)
        indices = random.sample(list(range(len(self.all_gt_colors))), 1)
        indices2 = random.sample(list(range(len(self.all_gt_colors))), 1)

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
            'color_template0': self.color_template0,
            'color_template1': self.color_template1,
            'color_template2': self.color_template2,
            'location_template': self.location_template,  # New
            'mat_template': self.mat_template,
            'all_gt_colors': [self.all_gt_colors[i] for i in indices],
            'all_ph_colors': [self.all_ph_colors[i] for i in indices],
            'all_colors': self.all_colors,
            'all_gt_mats': [self.all_gt_mats[i] for i in indices2],
            'all_ph_mats': [self.all_ph_mats[i] for i in indices2],
            'all_mats': self.all_mats,
            'all_gt_locations': [self.all_gt_locations[i] for i in indices],  # New
            'all_ph_locations': [self.all_ph_locations[i] for i in indices],  # New
            'all_locations': self.all_locations,  # New
            'blip_color': self.blip_colors[item // self.val_batch_size],
            'blip_mat': self.blip_mats[item // self.val_batch_size],
            'blip_fruit': self.blip_fruits[item // self.val_batch_size],
            'blip_location': self.blip_locations[item // self.val_batch_size],  # New
            'inf': self.inf_dict
        }

    # If there are additional methods that need to be overridden or updated, include them here.
    # For this example, we're assuming the parent class methods handle everything appropriately.

