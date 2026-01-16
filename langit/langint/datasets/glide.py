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
from collections import Counter
from transformers import CLIPVisionModel
import kornia
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


placeholder_words_list = [f'mytoken{i}' for i in range(700)]  #* ['mytoken0', 'mytoken1', ...]

# TEMPLATE = """a {} polyp with {} and pathology is {}"""
TEMPLATE = """A polyp, {} type, {} color, and pathology is {}"""


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, data_root: str, num_placeholder_words: int):
        super().__init__()

        self.data_root = data_root.replace('_', " ")
        self.templates = imagenet_templates_small

        pipeline = GLIDEPipeline()
        ground_truth_words = data_root.split(',')
        placeholder_words = placeholder_words_list[:num_placeholder_words]
        assert len(placeholder_words) == num_placeholder_words, (placeholder_words, num_placeholder_words)
        assert len(placeholder_words) == 1 or len(ground_truth_words) == len(placeholder_words), (ground_truth_words, placeholder_words)
        if len(placeholder_words) == 1:
            placeholder_words = placeholder_words * len(ground_truth_words)
        images_all: List[torch.Tensor] = []
        ph_words_all: List[str] = []
        for ind in range(len(ground_truth_words)):
            gt_word = ground_truth_words[ind]
            ph_word = placeholder_words[ind]
            prompt = self.templates[0].format(gt_word)
            images = pipeline.sample(prompt).cpu()
            images_all.append(images)
            ph_words_all.extend([ph_word] * len(images))
        self.images: torch.Tensor = torch.cat(images_all)
        self.placeholder_words: List[str] = ph_words_all
        del pipeline

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # GLIDE expects range [-1, 1]
        image: torch.Tensor = self.images[item]
        if np.random.rand() < .5:
            image = TF.hflip(image)

        ph_word = self.placeholder_words[item]
        prompt = self.templates[np.random.choice(len(self.templates))].format(ph_word)

        return {'image': image, 'prompt': prompt}


def glide_sample_prompt(pipeline, prompt: str, num_repeats=4):
    images: torch.Tensor = pipeline.sample(prompt, batch_size=num_repeats).cpu()
    return images


def glide_sample_prompts(prompts: List[str], num_repeats=4) -> torch.Tensor:
    from langint.utils.glide import GLIDEPipeline
    # return (bs, 3, 64, 64) in pixel range [-1, 1]
    pipeline = GLIDEPipeline()
    images_all: List[torch.Tensor] = []
    for prompt in prompts:
        images = glide_sample_prompt(pipeline, prompt, num_repeats=num_repeats)
        images_all.append(images)
    del pipeline
    return torch.cat(images_all)

import os
from PIL import Image
from collections import Counter
import torch
from torchvision import transforms as TF

def deepfloyd_sample_prompts_from_json(folder_path: str, json_file_path: str):
    # 读取JSON文件内容
    with open(json_file_path, 'r') as f:
        text_file_content = json.load(f)

    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    images_all: List[Image.Image] = []
    blip_fruits = []
    blip_mats = []
    blip_colors = []

    for image_file in image_files:
        # 加载每个图像
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert('RGB')  # 确保转换为RGB模式
        images_all.append(image)

        # 从json文件中获取对应的文本描述
        if image_file in text_file_content:
            description = text_file_content[image_file]["Beard_and_Age"].lower()
            # 按照 " - " 分隔成三部分
            description_parts = description.split(' - ')

            if len(description_parts) >= 3:
                # 确保有三个部分（果实、材料、颜色）
                blip_fruits.append(description_parts[0].strip())
                blip_mats.append(description_parts[1].strip())
                blip_colors.append(description_parts[2].strip())

    # 统计每种类别出现最多的项
    blip_fruit_counter = Counter(blip_fruits)
    blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]

    blip_mat_counter = Counter(blip_mats)
    blip_common_mat = blip_mat_counter.most_common(1)[0][0]

    blip_color_counter = Counter(blip_colors)
    blip_common_color = blip_color_counter.most_common(1)[0][0]

    # 返回图像和最常见的描述
    return torch.stack([TF.ToTensor()(image) * 2 - 1 for image in images_all]), blip_common_fruit, blip_common_mat, blip_common_color, images_all


def deepfloyd_sample_prompts_from_folder(folder_path: str, model=None, processor=None, blip_fruit_q=None, blip_mat_q=None, blip_color_q=None):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]  # 支持的图片格式
    # image_files.sort()  # 可选：按文件名排序，确保顺序一致

    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_mats = []
    blip_common_colors = []

    for image_file in image_files:
        # 加载每个图像
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert('RGB')  # 确保转换为RGB模式
        images_all.append(image)

        if model is not None:
            blip_fruits = []
            blip_mats = []
            blip_colors = []
            # 在这里对每张图像使用BLIP进行推理
            for image_i in range(len(images_all)):
                image = images_all[image_i]

                inputs = processor(image, blip_fruit_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs,max_new_tokens=40)
                blip_fruits.append(processor.batch_decode(blip_out, skip_special_tokens=True)[0].strip())

                inputs = processor(image, blip_mat_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs,max_new_tokens=40)
                blip_mats.append(processor.batch_decode(blip_out, skip_special_tokens=True)[0].strip())

                inputs = processor(image, blip_color_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs,max_new_tokens=40)
                blip_colors.append(processor.batch_decode(blip_out, skip_special_tokens=True)[0].strip())

            # 统计每种类别出现最多的项
            blip_fruit_counter = Counter(blip_fruits)
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            blip_common_fruits.append(blip_common_fruit)

            blip_mat_counter = Counter(blip_mats)
            blip_common_mat = blip_mat_counter.most_common(1)[0][0]
            blip_common_mats.append(blip_common_mat)

            blip_color_counter = Counter(blip_colors)
            blip_common_color = blip_color_counter.most_common(1)[0][0]
            blip_common_colors.append(blip_common_color)

    # 确保所有的数据长度一致
    if model is not None:
        assert len(image_files) == len(blip_common_fruits) == len(blip_common_mats) == len(blip_common_colors), \
            (len(image_files), len(blip_common_fruits), len(blip_common_mats), len(blip_common_colors))

    return torch.stack([TF.ToTensor()(image) * 2 - 1 for image in images_all]), blip_common_fruits, blip_common_mats, blip_common_colors, images_all


def deepfloyd_sample_prompts(prompts: List[str], num_repeats=1, model=None, processor=None, blip_fruit_q=None, blip_mat_q=None, blip_color_q=None,batch_size = None):
    # from langint.utils.deepfloyd_no_diffusers import Pipeline
    from polypdiffusion.polyp_diffusion_pipeline import PolypDiffusionPipeline
    pipeline = PolypDiffusionPipeline(
        config_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml',
        ckpt_path='/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/resume_sd/2024-11-21T00-12-28_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=49-step=699.ckpt',
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
    return torch.stack([image * 2 - 1 for image in images_all]), blip_common_fruits, blip_common_mats, blip_common_colors, images_all


def cache_deepfloyd_samples(prompts: List[str], num_repeats=4) -> torch.Tensor:
    cache_dir = 'cache/deepfloyd'
    image_paths_all = []
    pipeline = None
    for prompt in prompts:
        cache_subdir = prompt.replace(" ", "_")
        os.makedirs(os.path.join(cache_dir, cache_subdir), exist_ok=True)
        image_paths = [os.path.join(cache_dir, cache_subdir, f"{ind:02d}.png") for ind in range(num_repeats)]
        if not all(os.path.exists(path) for path in image_paths):
            if pipeline is None:
                from langint.utils.deepfloyd_no_diffusers import Pipeline
                pipeline = Pipeline()
            images: List[Image.Image] = pipeline.dream(prompt, count=num_repeats)
            for ind in range(num_repeats):
                images[ind].save(image_paths[ind])
        image_paths_all.extend(image_paths)
    if pipeline is not None:
        del pipeline
    return image_paths_all


def load_deepfloyd_samples(prompts: List[str], num_repeats=4):
    logger.info('loading deepfloyd samples from cache...')
    image_paths = cache_deepfloyd_samples(prompts, num_repeats)
    images: List[Image.Image] = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images])


class HookFunction:
    def __init__(self):
        self.layer_outputs = []

    def hook_layers(self, model):
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L206
        layer_counts = 0
        for layer in model.transformer.resblocks:
            layer_counts += 1
            assert layer.__class__.__name__ == 'ResidualAttentionBlock'
            layer.register_forward_hook(self.save_output)
        assert layer_counts > 0

    def save_output(self, module, input, output):
        self.layer_outputs.append(output.detach())

    def clear_outputs(self):
        self.layer_outputs = []


class SyntheticBiLevelOLD(torch.utils.data.Dataset):
    def __init__(self, data_root: str,##gt1-gt2-gt3
                 templates: Dict,
                 num_data_per_prompt: int = 8, num_data_copies: int = 1, num_tokens_per_word: int = 1,
                 num_placeholder_words: int = 35, num_placeholder_groups: int = 3, shared_tokens=0): 
        
        assert shared_tokens in [0, 1], shared_tokens

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        processor = Blip2Processor.from_pretrained("/home/hyl/yujia/A_few_shot/pipeline/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("/home/hyl/yujia/A_few_shot/pipeline/blip2-opt-2.7b", device_map={"": 0})

        # we limit it to words which are known to have corresponding t5 embeddings which are one token long
        blip_fruit_question = "Question: how many cats are there in this image? Answer:"
        blip_material_question = "What is the shape or size of the polyp in the image: small, large, flat, or raised?"
        blip_color_question = "What is the color of the polyp in the image: pink, red, white, or brown?"


        self.templates = build_from_config(templates)

        ground_truth_words = data_root.split(",")
        ground_truth_words = [word.replace('_', " ") for word in ground_truth_words]
#         #TODO:更改划分
        ground_truth_words = [word.split('-') for word in ground_truth_words] # [['apple', 'green'], ['apple', 'red'], ...]

        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        ground_truth_prompt_args = [[] for i in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups, (len(split_word), num_placeholder_groups)
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])
                # [apple, apple, banana, banana] and [green, red, green, yellow]

        self.ground_truth_prompt_args = ground_truth_prompt_args
        
        self.unique_gt_words = ground_truth_words
        for gt_word in self.unique_gt_words:
            assert len(gt_word) == num_placeholder_groups

        unique_prompts = []
        for ind in range(num_placeholder_words):
            curr_prompt_words = []
            for ground_truth_prompt_arg in ground_truth_prompt_args:
                curr_prompt_words.append(ground_truth_prompt_arg[ind])
            prompt = self.templates[0].format(*curr_prompt_words)
            unique_prompts.append(prompt)
        self.gt_prompts: List[str] = unique_prompts 
        # self.images, self.blip_fruits, self.blip_mats, self.blip_colors, pil_images = deepfloyd_sample_prompts_from_json(folder_path='/home/hyl/yujia/A_few_shot/data_text_private/image_215_256', 
        #                                                                                                                    json_file_path='/home/hyl/yujia/A_few_shot/data_text_private/metadata_processed.json')

        self.images, self.blip_fruits, self.blip_mats, self.blip_colors, pil_images = deepfloyd_sample_prompts_from_folder(folder_path='/home/hyl/yujia/A_few_shot/test_image_data', 
                                                                                                                           model=model, processor=processor, 
                                                                                                                           blip_fruit_q=blip_fruit_question, blip_mat_q=blip_material_question, blip_color_q=blip_color_question)
        # we limit it to words which are known to have corresponding t5 embeddings which are one token long
        new_blip_fruits = []
        for i in range(len(self.blip_fruits)):
            to_append = 'clothes'
            for x in ['shirt', 'pants', 'shoes', 'dress', 'cap']:
                if x in self.blip_fruits[i]:
                    to_append = x
            new_blip_fruits.append(to_append)

        new_blip_mats = []
        for i in range(len(self.blip_mats)):
            to_append = 'season'
            for x in ['spring', 'summer', 'fall', 'winter']:
                if x in self.blip_mats[i]:
                    to_append = x
            new_blip_mats.append(to_append)

        new_blip_colors = []
        for i in range(len(self.blip_colors)):
            to_append = 'color'
            for x in ['red', 'yellow', 'green', 'purple', 'white', 'cream']:
                if x in self.blip_colors[i]:
                    to_append = x
            new_blip_colors.append(to_append)

        self.blip_fruits = new_blip_fruits
        self.blip_mats = new_blip_mats
        self.blip_colors = new_blip_colors

        del processor
        del model
        torch.cuda.empty_cache()
        clip_vision = CLIPVisionModel.from_pretrained("/home/hyl/yujia/A_few_shot/models/clip-vit-base-patch32").to("cuda").requires_grad_(False)
        
        def clip_preprocess(x):
            x = kornia.geometry.resize(
                x, (clip_vision.config.image_size, clip_vision.config.image_size), interpolation='bicubic', align_corners=True, antialias=False
            )
            x = (x + 1.) / 2.
            # renormalize according to clip
            x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
            return x

        preprocessed_images = self.images.to("cuda")
        
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.clip_features.append(result)
                run_count += 1
        self.clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.clip_features] 
        
        for i in range(len(self.clip_features)):
            
            assert self.clip_features[i].shape == (3,6,768)
        assert len(self.clip_features) == len(self.images), (len(self.clip_features), len(self.images))
#######################inference######################
        inference_input = 'gloves-winter-red'
        inf_data = inference_input.split(",")  #* inf_data = ['gloves-winter-red']
        inf_data = [word.replace('_', " ") for word in inf_data]  #* inf_data = ['gloves-winter-red']
        inf_data = [word.split('-') for word in inf_data]  #* inf_data = [['gloves', 'winter', 'red'],[...],...]

        inf_ph_tokens = [[f'mytoken{3*i}', f'mytoken{3*i + 1}', f'mytoken{3*i + 2}'] for i in range(len(inf_data))]  #* inf_path_tokens = [['mytoken0', 'mytoken1', 'mytoken2']]

        assert len(inf_data) == len(inf_ph_tokens), (len(inf_data), len(inf_ph_tokens), inf_data, inf_ph_tokens)

        self.inf_gt_prompts = [self.templates[0].format(*pair) for pair in inf_data]  #* self.inf_gt_prompts = a photo of a {'gloves'} for the {'winter'} season which is {'red'} in color

        self.inf_prompts = [self.templates[0].format(*inf_ph_tokens[i]) for i in range(len(inf_data))]  #* self.inf_prompts = a photo of a {'mytoken0'} for the {'mytoken1'} season which is {'mytoken2'} in color

        self.inf_fruit_prompts = [imagenet_templates_small[0].format(pair[0]) for pair in inf_ph_tokens]  #* self.inf_fruit_prompts = a photo of a {'mytoken0'}
        self.inf_mat_prompts = ['a photo of the {} season'.format(pair[1]) for pair in inf_ph_tokens]  #* self.inf_mat_prompts = a photo of the {'mytoken1'} season
        self.inf_color_prompts = ['a photo of the color {}'.format(pair[2]) for pair in inf_ph_tokens]  #* self.inf_color_prompts = a photo of the color {'mytoken2}

        self.inf_images, _, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1)  #* 得到「torch 形式的图片」和「原 PIL 格式的图片」

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)  #* 得到处理后的图片信息

        run_count = 0
        self.inf_clip_features = []  #* 储存每张图片的特征的 list
        for img in preprocessed_images:  #* 遍历每一张图
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

        self.num_placeholder_words = num_placeholder_words*num_tokens_per_word  #* 215 * 1
        placeholder_words = []  #* 储存所有 'ground_truth_words' 里的单词对应的 'mytoken id' 标记: ['mytoken0', 'mytoken1', 'mytoken2', ...., ]
        fruit_dict = {}  # 每一个 '主体' 对应的 'mytoken id'
        fruit_count = 0   
        mat_count = 1
        color_count = 2  
        for i in range(len(ground_truth_words)):  #* 遍历每一个划分后的三元组，item: [shirt, spring, red] 
            curr_fruit = ground_truth_words[i][0].split()[-1]  #* 拿到第一个 word（如 'shirt'）
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

          #* placeholder_words_prompt_args = [ [['mytoken0']，['mytoken3'], ...], [['mytoken1'], ['mytoken4'], ...], [['mytoken2'], ['mytoken5'], ...]]
        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words*num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(np.split(np.array(placeholder_words), num_placeholder_words), (1,0,2)) 

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)
        

        self.ph_words_all = []
        """
          假如我的样本是：
          placeholder_words_prompt_args = [ 
            [
                ['mytoken0'], ['mytoken3'], ['mytoken6'], ['mytoken9']
            ], 
            [
                ['mytoken1'], ['mytoken4'], ['mytoken7'], ['mytoken10']
            ], 
            [
                ['mytoken2'], ['mytoken5'], ['mytoken8'], ['mytoken11']
            ]
        ] (3 x num_placeholder_words)
          并且我有 4 个样本
        """
        for ind in range(num_placeholder_words):  #* 遍历所有样本中的每一项
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:  #* 对于三元组中的每一位
                curr_ph_words.append(placeholder_words_prompt_arg[ind])  #* curr_ph_words = ['mytoken1', 'mytoken2', 'mytoken3']
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)  #* ph_words_all = [['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken1', 'mytoken2', 'mytoken3']](假如 num_data_per_prompt 为3)
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


class SyntheticBiLevel(torch.utils.data.Dataset):
    IMAGE_FILE_ROOT = "/home/hyl/yujia/A_few_shot/data_text_private/image_215_256"
    GT_JSON_FILE = "/home/hyl/yujia/A_few_shot/data_text_private/metadata_processed.json"

    def __init__(self, num_data_per_prompt: int = 8, num_data_copies: int = 1, num_tokens_per_word: int = 1,
                  num_placeholder_words: int = 215, num_placeholder_groups: int = 3, shared_tokens=0,*args, **kwargs):
        # 加载「图片」和「ground truth数据」
        self.images, self.blip_fruits, self.blip_mats, self.blip_colors = self.process_gt_labels()

        # 加载 「clip 模型」并得到「clip_features」
        self.clip_vision = CLIPVisionModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to("cuda").requires_grad_(False)
        self.clip_features = self.process_clip_features()

        inference_input = '0-IIa - white - tubular adenoma with low-grade epithelial dysplasia'
        inf_data = inference_input.split(",")  #* inf_data = ['gloves-winter-red']
        inf_data = [word.replace('_', " ") for word in inf_data]  #* inf_data = ['gloves-winter-red']
        inf_data = [word.split(' - ') for word in inf_data]  #* inf_data = [['gloves', 'winter', 'red'],[...],...]

        inf_ph_tokens = [[f'mytoken{3*i}', f'mytoken{3*i + 1}', f'mytoken{3*i + 2}'] for i in range(len(inf_data))]  #* inf_path_tokens = [['mytoken0', 'mytoken1', 'mytoken2']]

        assert len(inf_data) == len(inf_ph_tokens), (len(inf_data), len(inf_ph_tokens), inf_data, inf_ph_tokens)

        self.inf_gt_prompts = [TEMPLATE.format(*pair) for pair in inf_data]  #* self.inf_gt_prompts = a photo of a {'gloves'} for the {'winter'} season which is {'red'} in color

        self.inf_prompts = [TEMPLATE.format(*inf_ph_tokens[i]) for i in range(len(inf_data))]  #* self.inf_prompts = a photo of a {'mytoken0'} for the {'mytoken1'} season which is {'mytoken2'} in color

        self.inf_fruit_prompts = ['A polyp, {} type'.format(pair[0]) for pair in inf_ph_tokens]  #* self.inf_fruit_prompts = a photo of a {'mytoken0'}
        self.inf_color_prompts = ['A polyp, {} color'.format(pair[1]) for pair in inf_ph_tokens]  #* self.inf_color_prompts = a photo of the color {'mytoken2}
        self.inf_mat_prompts = ['A polyp, and pathology is {}'.format(pair[2]) for pair in inf_ph_tokens]  #* self.inf_mat_prompts = a photo of the {'mytoken1'} season
        

        self.inf_images, _, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1,batch_size=1)  #* 得到「torch 形式的图片」和「原 PIL 格式的图片」

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = self._clip_preprocess(preprocessed_images)  #* 得到处理后的图片信息

        run_count = 0
        self.inf_clip_features = []  #* 储存每张图片的特征的 list
        for img in preprocessed_images:  #* 遍历每一张图
            with torch.no_grad():
                result = self.clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [self.clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.inf_clip_features.append(result)
                run_count += 1
        self.inf_clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.inf_clip_features]

        for i in range(len(self.inf_clip_features)):
            assert self.inf_clip_features[i].shape == (3,12,1024)
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


        torch.cuda.empty_cache()

        self.num_placeholder_words = num_placeholder_words*num_tokens_per_word  #* 215 * 1
        placeholder_words = []  #* 储存所有 'ground_truth_words' 里的单词对应的 'mytoken id' 标记: ['mytoken0', 'mytoken1', 'mytoken2', ...., ]
        fruit_dict = {} 
        color_dict = {}
        mat_dict = {} # 每一个 '主体' 对应的 'mytoken id'
        fruit_count = 0   
        color_count = 1
        mat_count = 2
          
        ground_truth_words = self.get_ground_truth_words()

        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        ground_truth_prompt_args = [[] for i in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups, (len(split_word), num_placeholder_groups)
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])
                # [apple, apple, banana, banana] and [green, red, green, yellow]

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

        # for i in range(len(ground_truth_words)):  #* 遍历每一个划分后的三元组，item: [shirt, spring, red] 
        #     curr_fruit = ground_truth_words[i][0].split()[-1]  #* 拿到第一个 word（如 'shirt'）
        #     if shared_tokens == 1:
        #         # the category token is shared across different instances of the same category
        #         if curr_fruit not in fruit_dict:
        #             fruit_dict[curr_fruit] = f'mytoken{fruit_count}'
        #             fruit_count += 3
        #         placeholder_words.append(fruit_dict[curr_fruit])
        #     else:
        #         placeholder_words.append(f'mytoken{fruit_count}')
        #         fruit_count += 3  

        #     placeholder_words.append(f'mytoken{color_count}')
        #     color_count += 3
            
        #     placeholder_words.append(f'mytoken{mat_count}')
        #     mat_count += 3
        for i in range(len(ground_truth_words)):  # 遍历每一个划分后的三元组，item: [shirt, spring, red]
            # 获取当前水果的名称（例如 'shirt'）
            curr_fruit = ground_truth_words[i][0].split()[-1]  
            
            # 处理水果类别
            if shared_tokens == 1:
                if curr_fruit not in fruit_dict:
                    fruit_dict[curr_fruit] = f'mytoken{fruit_count}'
                    fruit_count += 3
                placeholder_words.append(fruit_dict[curr_fruit])
            else:
                placeholder_words.append(f'mytoken{fruit_count}')
                fruit_count += 3  
            
            # 处理颜色类别
            curr_color = ground_truth_words[i][1].split()[-1]  # 假设颜色类别是每个三元组中的第三个词
            if shared_tokens == 1:
                if curr_color not in color_dict:
                    color_dict[curr_color] = f'mytoken{color_count}'
                    color_count += 3
                placeholder_words.append(color_dict[curr_color])
            else:
                placeholder_words.append(f'mytoken{color_count}')
                color_count += 3  
            
            # 处理物质类别
            curr_mat = ground_truth_words[i][2].split()[-1]  # 假设物质类别是每个三元组中的第二个词
            if shared_tokens == 1:
                if curr_mat not in mat_dict:
                    mat_dict[curr_mat] = f'mytoken{mat_count}'
                    mat_count += 3
                placeholder_words.append(mat_dict[curr_mat])
            else:
                placeholder_words.append(f'mytoken{mat_count}')
                mat_count += 3

            

          #* placeholder_words_prompt_args = [ [['mytoken0']，['mytoken3'], ...], [['mytoken1'], ['mytoken4'], ...], [['mytoken2'], ['mytoken5'], ...]]
        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words*num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(np.split(np.array(placeholder_words), num_placeholder_words), (1,0,2)) 

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)
        

        self.ph_words_all = []
        """
          假如我的样本是：
          placeholder_words_prompt_args = [ 
            [
                ['mytoken0'], ['mytoken3'], ['mytoken6'], ['mytoken9']
            ], 
            [
                ['mytoken1'], ['mytoken4'], ['mytoken7'], ['mytoken10']
            ], 
            [
                ['mytoken2'], ['mytoken5'], ['mytoken8'], ['mytoken11']
            ]
        ] (3 x num_placeholder_words)
          并且我有 4 个样本
        """
        for ind in range(num_placeholder_words):  #* 遍历所有样本中的每一项
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:  #* 对于三元组中的每一位
                curr_ph_words.append(placeholder_words_prompt_arg[ind])  #* curr_ph_words = ['mytoken1', 'mytoken2', 'mytoken3']
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)  #* ph_words_all = [['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken1', 'mytoken2', 'mytoken3'], ['mytoken1', 'mytoken2', 'mytoken3']](假如 num_data_per_prompt 为3)
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
            assert clip_features[i].shape == (3,12,1024)
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
    
    def get_ground_truth_words(self):
        data = json.load(open(self.GT_JSON_FILE))

        ground_truth_words = []
        for _, attributes in data.items():
            attributes_str = attributes['Beard_and_Age'].lower() #NOTE: 记得改
            attributes_list = [item.strip() for item in attributes_str.split(' - ')]
            assert len(attributes_list) == 3, "The attributes number is incorrect!"

            ground_truth_words.append([attributes_list[0], attributes_list[1], attributes_list[2]])

        return ground_truth_words

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> Any:
        image = self.images[index]
        if np.random.rand() < .5:
            hflip = TF.RandomHorizontalFlip(p=1.0)
            image = hflip(image)
        curr_ph_words = self.ph_words_all[index]
        
        clip_feature = self.clip_features[index]
        blip_color = self.blip_colors[index]
        blip_mat = self.blip_mats[index]
        blip_fruit = self.blip_fruits[index]
        prompt = TEMPLATE.format(*[''.join(word) for word in curr_ph_words])
        gt_prompt = self.process_prompt(blip_color, blip_mat, blip_fruit)

        return {'image': image, 'prompt': prompt, 'gt_prompt': gt_prompt, 'clip_feature': clip_feature, 'blip_color': blip_color, 'blip_mat': blip_mat, 'blip_fruit': blip_fruit}

class SyntheticBiLevelEval(SyntheticBiLevel):
    def __init__(self, data_root: str, num_placeholder_words: int, templates: Dict, ref_dataset: SyntheticBiLevel):

        unique_ph_words = []
        num_placeholder_words = len(ref_dataset.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in ref_dataset.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])

        self.image = torch.zeros_like(ref_dataset.images[0])
        self.gt_word_pairs = ref_dataset.unique_gt_words
        self.ph_word_pairs = unique_ph_words
        self.full_template = TEMPLATE
        self.fruit_template = 'A polyp, {} type.'
        self.mat_template = 'A polyp, and pathology is {}.'
        self.color_template0 = 'A polyp, {} color.'
        self.color_template1 = 'A polyp, {} color.' 
        self.color_template2 = 'A polyp, {} color.'
        # a {} polyp with {} and the pathology is {}
        self.val_batch_size = 4
        self.all_gt_colors = [word_pair[2] for word_pair in self.gt_word_pairs]
        self.all_ph_colors = [word_pair[2] for word_pair in self.ph_word_pairs]
        self.all_colors = [word_pair[2] for word_pair in ref_dataset.unique_gt_words]
        self.all_gt_mats = [word_pair[1] for word_pair in self.gt_word_pairs]
        self.all_ph_mats = [word_pair[1] for word_pair in self.ph_word_pairs]
        self.all_mats = [word_pair[1] for word_pair in ref_dataset.unique_gt_words]
        self.blip_colors = ref_dataset.blip_colors
        self.blip_mats = ref_dataset.blip_mats
        self.blip_fruits = ref_dataset.blip_fruits

        self.inf_dict = ref_dataset.inf_dict
        
    def __len__(self):
        return len(self.gt_word_pairs) * self.val_batch_size

    def __getitem__(self, item):
        gt_word_pair = self.gt_word_pairs[item//self.val_batch_size]
        ph_word_pair = self.ph_word_pairs[item//self.val_batch_size]
        gt_prompt = self.full_template.format(*gt_word_pair)
        prompt = self.full_template.format(*ph_word_pair)

        assert len(self.all_ph_colors) == len(self.all_gt_colors) == len(self.all_ph_mats) == len(self.all_gt_mats), (len(self.all_ph_colors), len(self.all_gt_colors), len(self.all_ph_mats), len(self.all_gt_mats))
        random.seed(item)
        indices = random.sample((list(range(len(self.all_gt_colors)))), 1)
        indices2 = random.sample((list(range(len(self.all_gt_colors)))), 1)
        
        return {
            'image': self.image,
            'prompt': prompt,
            'gt_prompt': gt_prompt,
            'gt_fruit': gt_word_pair[0],
            'gt_mat': gt_word_pair[2],
            'gt_color': gt_word_pair[1],
            'ph_fruit': ph_word_pair[0],
            'ph_mat': ph_word_pair[2],
            'ph_color': ph_word_pair[1],
            'full_template': self.full_template,
            'fruit_template': self.fruit_template,
            'color_template0': self.color_template0,
            'color_template1': self.color_template1,
            'color_template2': self.color_template2,
            'mat_template': self.mat_template,
            'all_gt_colors': [self.all_gt_colors[i] for i in indices],
            'all_ph_colors': [self.all_ph_colors[i] for i in indices],
            'all_colors': self.all_colors,
            'all_gt_mats': [self.all_gt_mats[i] for i in indices2],
            'all_ph_mats': [self.all_ph_mats[i] for i in indices2],
            'all_mats': self.all_mats,
            'blip_color': self.blip_colors[item//self.val_batch_size],
            'blip_mat': self.blip_mats[item//self.val_batch_size],
            'blip_fruit': self.blip_fruits[item//self.val_batch_size],
            'inf': self.inf_dict
        }