import json
import torch
from typing import *
from tu.utils.config import build_from_config
import os
from PIL import Image
from typing import List, Dict
import numpy as np
from torchvision import transforms as TF
from langint.utils.dataset import imagenet_templates_small
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
from transformers import CLIPVisionModel
import kornia
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

TEMPLATE = """a {} polyp, the color is {}, and the pathlogy is {}."""

class SyntheticBiLevel(torch.utils.data.Dataset):
    IMAGE_FILE_ROOT = "/home/hyl/yujia/A_few_shot/data_text_private/image_215_256"
    GT_JSON_FILE = "/home/hyl/yujia/A_few_shot/data_text_private/metadata_processed.json"

    def __init__(self):
        # 加载「图片」和「ground truth数据」
        self.images, self.blip_fruits, self.blip_mats, self.blip_colors = self.process_gt_labels()

        # 加载 「clip 模型」并得到「clip_features」
        self.clip_vision = CLIPVisionModel.from_pretrained("/home/hyl/yujia/A_few_shot/models/clip-vit-base-patch32").to("cuda").requires_grad_(False)
        self.clip_features = self.process_clip_features()

    def process_gt_labels(self):
        data = json.load(open(self.GT_JSON_FILE))

        image_pathes = []
        blip_fruits = []
        blip_mats = []
        blip_colors = []

        for image_path, attributes in data.items():
            attributes_str = attributes['Beard_and_Age'].lower() #NOTE: 记得改
            attributes_list = [item.strip() for item in attributes_str.split(' - ')]
            assert len(attributes_list) == 3, "The attributes number is incorrect!"

            blip_fruits.append(attributes_list[0])
            blip_colors.append(attributes_list[1])
            blip_mats.append(attributes_list[2])
            
            image_pathes.append(os.path.join(self.IMAGE_FILE_ROOT, image_path))

        images = self.process_images(image_pathes)

        return images, blip_fruits, blip_mats, blip_colors

    def process_images(self, image_pathes: list[str]):
        images_all = []

        for image_path in image_pathes:
            image = Image.open(image_path).convert('RGB')  # 确保转换为RGB模式
            images_all.append(image)

        return torch.stack([TF.ToTensor()(image) * 2 - 1 for image in images_all])

    def process_clip_features(self):
        preprocessed_images = self.images.to("cuda")
        preprocessed_images = self._clip_preprocess(preprocessed_images)

        clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = self.clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [self.clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                clip_features.append(result)

        clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in clip_features]

        # 判断 clip_features 形状
        for i in range(len(clip_features)):
            assert clip_features[i].shape == (3,6,768)
        assert len(clip_features) == len(self.images), (len(clip_features), len(self.images))

        return clip_features

    def _clip_preprocess(self, x):
        x = kornia.geometry.resize(
            x, (self.clip_vision.config.image_size, self.clip_vision.config.image_size), interpolation='bicubic', align_corners=True, antialias=False
        )
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return x
    
    def process_prompt(self, blip_color, blip_mat, blip_fruit):
        return TEMPLATE.format(blip_fruit, blip_color, blip_mat)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> Any:
        image = self.images[index]
        clip_feature = self.clip_features[index]
        blip_color = self.blip_colors[index]
        blip_mat = self.blip_mats[index]
        blip_fruit = self.blip_fruits[index]

        prompt = self.process_prompt(blip_color, blip_mat, blip_fruit)

        return {'image': image, 'prompt': prompt, 'gt_prompt': prompt, 'clip_feature': clip_feature, 'blip_color': blip_color, 'blip_mat': blip_mat, 'blip_fruit': blip_fruit}



        inference_input = 'ii2-red-low'
        inf_data = inference_input.split(",")
        inf_data = [word.replace('_', " ") for word in inf_data]
        inf_data = [word.split('-') for word in inf_data] # [['apple', 'green'], ['apple', 'red'], ...]

        inf_ph_tokens = [[f'mytoken{3*i}', f'mytoken{3*i + 1}', f'mytoken{3*i + 2}'] for i in range(len(inf_data))]

        assert len(inf_data) == len(inf_ph_tokens), (len(inf_data), len(inf_ph_tokens), inf_data, inf_ph_tokens)

        self.inf_gt_prompts = [self.templates[0].format(*pair) for pair in inf_data]

        self.inf_prompts = [self.templates[0].format(*inf_ph_tokens[i]) for i in range(len(inf_data))]

        self.inf_fruit_prompts = [imagenet_templates_small[0].format(pair[0]) for pair in inf_ph_tokens]
        self.inf_mat_prompts = ['a photo of the {} season'.format(pair[1]) for pair in inf_ph_tokens]
        self.inf_color_prompts = ['a photo of the color {}'.format(pair[2]) for pair in inf_ph_tokens]

        self.inf_images, _, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1)

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.inf_clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.inf_clip_features.append(result)
                run_count += 1
        self.inf_clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.inf_clip_features]

        for i in range(len(self.inf_clip_features)):
            assert self.inf_clip_features[i].shape == (3,6,768)
        assert len(self.inf_clip_features) == len(self.inf_images), (len(self.inf_clip_features), len(self.inf_images))

        self.inf_dict = {
            'image': [img for img in self.inf_images],
            'prompt': self.inf_prompts,
            'gt_prompt': self.inf_gt_prompts,
            'fruit_prompt': self.inf_fruit_prompts,
            'mat_prompt': self.inf_mat_prompts,
            'color_prompt': self.inf_color_prompts,
            'clip_feature': [feat for feat in self.inf_clip_features],
        }        
#TODO
        # image:gt_prompt_generate,tensor,inf_image; 
        # prompt:a photo of a mytoken0 for the mytoken1 season which is mytoken2 in color
        # gt_prompt:a photo of a gloves for the winter season which is red in color'
        # fruit_prompt' =['a photo of a mytoken0']
        # 'mat_prompt' =['a photo of the mytoken1 season']
        # 'color_prompt' =['a photo of the color mytoken2']
        # clip_feature:encode inf_image(gt_generate)
        del clip_vision
        torch.cuda.empty_cache()

        self.num_placeholder_words = num_placeholder_words*num_tokens_per_word
        placeholder_words = [] #['mytoken0', 'mytoken1', 'mytoken0', ...., ]
        fruit_dict = {}
        fruit_count = 0
        mat_count = 1
        color_count = 2
        for i in range(len(ground_truth_words)):
            curr_fruit = ground_truth_words[i][0].split()[-1]
            if shared_tokens == 1:
                # the category token is shared across different instances of the same category
                if curr_fruit not in fruit_dict:
                    fruit_dict[curr_fruit] = f'mytoken{fruit_count}'
                    fruit_count += 3
                placeholder_words.append(fruit_dict[curr_fruit])
            else:
                placeholder_words.append(f'mytoken{fruit_count}')
                fruit_count += 3
            
            placeholder_words.append(f'mytoken{mat_count}')
            mat_count += 3
            placeholder_words.append(f'mytoken{color_count}')
            color_count += 3

        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words*num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(np.split(np.array(placeholder_words), num_placeholder_words), (1,0,2)) 

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)
        

        self.ph_words_all = []
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)
        self.placeholder_words_prompt_args = placeholder_words_prompt_args
        unique_ph_words = []
        num_placeholder_words = len(self.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in self.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])

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

    def __len__(self):
        return len(self.images) * self.num_data_copies

    def __getitem__(self, item):
        item = item % len(self.images)
        image: torch.Tensor = self.images[item]
        if np.random.rand() < .5:
            hflip = TF.RandomHorizontalFlip(p=1.0)
            image = hflip(image)

        curr_ph_words = self.ph_words_all[item]

        template = self.templates[np.random.choice(len(self.templates))]
        prompt = template.format(*[''.join(word) for word in curr_ph_words])

        clip_feature = self.clip_features[item]
        blip_color = self.blip_colors[item//self.num_data_per_prompt]
        blip_mat = self.blip_mats[item//self.num_data_per_prompt]
        blip_fruit = self.blip_fruits[item//self.num_data_per_prompt]

        return {'image': image, 'prompt': prompt, 'gt_prompt': self.gt_prompts[item//self.num_data_per_prompt], 'clip_feature': clip_feature, 'blip_color': blip_color, 'blip_mat': blip_mat, 'blip_fruit': blip_fruit}

