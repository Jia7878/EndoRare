import torch.nn as nn
import os
from langint.datasets.glide import placeholder_words_list
import torch

from typing import List, Dict, Optional
import logging
from datetime import datetime
from tu.trainers.simple_trainer import BaseTrainer
from tu.utils.training import process_batch
import numpy as np
from tu.utils.visualize import dump_helper, dump_row_helper
from torchvision import transforms
import PIL

# Initialize transformations
to_tensor = transforms.ToTensor()

from tu.train.setup import get_parser

# Updated TEMPLATE to include the 'location' dimension
TEMPLATE = """A polyp, {} type, {} color, and pathology is {}."""

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def _visualize_core(self, data, prefix=None):
        """
        Core visualization method to generate and save images based on different prompt variations.
        
        Args:
            data (dict): Contains 'image', 'prompt', 'ph_fruit', 'ph_mat', 'ph_color', 'ph_location',
                         'fruit_template', 'mat_template', 'color_template', 'location_template'.
            prefix (str): Prefix indicating the phase (e.g., 'val').
        """
        model = self.modules[self.module_key]
        model.eval()
        device = next(model.parameters()).device
        
        parser = get_parser()
        args = parser.parse_args()
        
        with torch.no_grad():
            if 'val' in prefix:
                # Initialize a zero image for visualization purposes
                zero_image = torch.zeros_like(data['image'][0]).unsqueeze(0).to(device)
                images = [data['image'] * 0.5 + 0.5]  # Normalize input image for visualization
    
                # Pass the main prompt through the model
                out = model({'prompt': data['prompt']}, return_all=True)
                fake_data = {
                    'image': zero_image.repeat(len(data['prompt']), 1, 1, 1), 
                    'prompt': data['prompt']
                }
                img = self.loss_modules['textual_inversion'].visualize_loop({'embeddings': out['embeddings']}, fake_data, args.opt_embs_file_path)['image']
                # img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                # # Convert PIL image to Tensor if necessary
                # if isinstance(img, PIL.Image.Image):
                #     img = to_tensor(img).unsqueeze(0).to(device)
                # images.append(img)
    
                # # Initialize lists to hold prompts for each attribute
                # fruit_prompts = []
                # mat_prompts = []
                # color_prompts = []
                # location_prompts = []  # New: List for location prompts
    
                # # Unpack placeholder words and templates, including location
                # for ph_fruit, ph_mat, ph_color, ph_location, fruit_template, mat_template, color_template, location_template in zip(
                #     data['ph_fruit'], 
                #     data['ph_mat'], 
                #     data['ph_color'], 
                #     data['ph_location'],        # New
                #     data['fruit_template'], 
                #     data['mat_template'], 
                #     data['color_template'], 
                #     data['location_template']   # New
                # ):
                #     fruit_prompts.append(fruit_template.format(ph_fruit))
                #     mat_prompts.append(mat_template.format(ph_mat))
                #     color_prompts.append(color_template.format(ph_color))
                #     location_prompts.append(location_template.format(ph_location))  # New
    
                # # Process fruit prompts
                # out = model({'prompt': fruit_prompts}, return_all=True)
                # fake_data = {
                #     'image': zero_image.repeat(len(fruit_prompts), 1, 1, 1), 
                #     'prompt': fruit_prompts
                # }
                # img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                # if isinstance(img, PIL.Image.Image):
                #     img = to_tensor(img).unsqueeze(0).to(device)
                # images.append(img)
    
                # # Process color prompts
                # out = model({'prompt': color_prompts}, return_all=True)
                # fake_data = {
                #     'image': zero_image.repeat(len(color_prompts), 1, 1, 1), 
                #     'prompt': color_prompts
                # }
                # img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                # if isinstance(img, PIL.Image.Image):
                #     img = to_tensor(img).unsqueeze(0).to(device)
                # images.append(img)
    
                # # Process mat prompts
                # out = model({'prompt': mat_prompts}, return_all=True)
                # fake_data = {
                #     'image': zero_image.repeat(len(mat_prompts), 1, 1, 1), 
                #     'prompt': mat_prompts
                # }
                # img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                # if isinstance(img, PIL.Image.Image):
                #     img = to_tensor(img).unsqueeze(0).to(device)
                # images.append(img)
    
                # # Process location prompts (New)
                # out = model({'prompt': location_prompts}, return_all=True)
                # fake_data = {
                #     'image': zero_image.repeat(len(location_prompts), 1, 1, 1), 
                #     'prompt': location_prompts
                # }
                # img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                # if isinstance(img, PIL.Image.Image):
                #     img = to_tensor(img).unsqueeze(0).to(device)
                # images.append(img)
    
                # # Stack all attribute images for visualization
                # nrow = len(images)
                # images = [tensor.unsqueeze(1) for tensor in images]
                # images = torch.cat(images, dim=1)
                # reshape_size = [-1] + list(images[0].shape[1:])
                # images = images.view(*reshape_size)
    
                # # Dump the reconstructed images
                # dump_helper(self, 'Reconstruction', images, prefix='', nrow=nrow)
    
                # # Define random ground truth and placeholder words for permutation
                # rand_gt_fruits = ['0-IIa type', '0-Isp type', '0-Is type', '0-IIa', 'type4', '0-Isp']
                # rand_ph_fruits = [f'mytoken{x}' for x in [6, 3, 27, 6, 69, 3]]
                # rand_gt_mats = [
                #     'adenomatous polyp with low-grade epithelial dysplasia', 
                #     'tubular adenoma with high-grade intraepithelial neoplasia', 
                #     'adenomatous polyp with low-grade epithelial dysplasia', 
                #     'tubular adenoma with high-grade intraepithelial neoplasia', 
                #     'adenomatous polyp with low-grade epithelial dysplasia', 
                #     'tubular adenoma with high-grade intraepithelial neoplasia'
                # ]
                # rand_ph_mats = [f'mytoken{x}' for x in [35, 332, 35, 332, 35, 332]]
                # rand_gt_colors = ['red', 'white', 'pale', 'very pale', 'red', 'similar color to surrounding mucosa']
                # rand_ph_colors = [f'mytoken{x}' for x in [34, 19, 7, 7, 31, 37]]
                # rand_gt_locations = ['location1', 'location2', 'location3', 'location4', 'location5', 'location6']  # New
                # rand_ph_locations = [f'mytoken{x}' for x in [40, 41, 42, 43, 44, 45]]  # New
                # perm_length = 6
    
                # assert perm_length == len(rand_gt_fruits) == len(rand_ph_fruits) == len(rand_gt_mats) == len(rand_ph_mats) \
                #     == len(rand_gt_colors) == len(rand_ph_colors) == len(rand_gt_locations) == len(rand_ph_locations), \
                #     "Permutation lengths do not match."
    
                # # Create new ground truth and placeholder lists
                # new_gt_fruits = [data['gt_fruit'][0]] * perm_length
                # new_ph_fruits = [data['ph_fruit'][0]] * perm_length
                # new_gt_mats = [data['gt_mat'][0]] * perm_length
                # new_ph_mats = [data['ph_mat'][0]] * perm_length
                # new_gt_colors = [data['gt_color'][0]] * perm_length
                # new_ph_colors = [data['ph_color'][0]] * perm_length
                # new_gt_locations = [data['gt_location'][0]] * perm_length  # New
                # new_ph_locations = [data['ph_location'][0]] * perm_length  # New
    
                # # Define permutations including the 'location' attribute
                # perms = [
                #     [rand_gt_fruits, rand_gt_colors, new_gt_mats, rand_gt_locations, 
                #      rand_ph_fruits, rand_ph_colors, new_ph_mats, rand_ph_locations],  # Swap color and mat positions
                #     [rand_gt_fruits, new_gt_colors, rand_gt_mats, rand_gt_locations, 
                #      rand_ph_fruits, new_ph_colors, rand_ph_mats, rand_ph_locations],  # Swap color and mat positions
                #     [new_gt_fruits, rand_gt_colors, rand_gt_mats, rand_gt_locations, 
                #      new_ph_fruits, rand_ph_colors, rand_ph_mats, rand_ph_locations],  # Swap color and mat positions
                # ]
    
                # for perm in perms:
                #     gt_comp_prompts = []
                #     ph_comp_prompts = []
                #     comp_imgs = []
                #     # Unpack the permutation list into corresponding attributes
                #     for gt_fruit, gt_color, gt_mat, gt_location, ph_fruit, ph_color, ph_mat, ph_location in zip(*perm):
                #         gt_comp_prompts.append(TEMPLATE.format(gt_fruit, gt_color, gt_mat, gt_location))  # Updated TEMPLATE with location
                #         ph_comp_prompts.append(TEMPLATE.format(ph_fruit, ph_color, ph_mat, ph_location))  # Updated TEMPLATE with location
    
                #     # Pass placeholder prompts through the model
                #     out = model({'prompt': ph_comp_prompts}, return_all=True)
                #     num_cross_sets = 2
                #     for i in range(num_cross_sets):
                #         fake_data = {
                #             'image': zero_image.repeat(len(ph_comp_prompts), 1, 1, 1), 
                #             'prompt': ph_comp_prompts
                #         }
                #         comp_img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                #         if isinstance(comp_img, PIL.Image.Image):
                #             comp_img = to_tensor(comp_img).unsqueeze(0).to(device)
                #         comp_imgs.append(comp_img)
                #     nrow = len(ph_comp_prompts)
                #     comp_imgs = torch.cat(comp_imgs, dim=0)
                    
                #     # Dump the extrapolated images
                #     dump_helper(self, 'Extrapolation', comp_imgs, prefix='', nrow=nrow)
            else:
                # Handle non-validation visualization
                dump_helper(self, 'Input', data['image'] * 0.5 + 0.5, prefix='')
                out = model(data, return_all=True)
                for k, v in self.loss_modules['textual_inversion'].visualize(out, data).items():
                    if isinstance(v, torch.Tensor) and v.ndim == 4:
                        dump_helper(self, "Output", v * 0.5 + 0.5, prefix='')
                    elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        if len(v) > 10:
                            indices = np.linspace(0, len(v), 10, endpoint=False, dtype=int)
                        else:
                            indices = np.arange(len(v))
                        v = [v[i] * 0.5 + 0.5 for i in indices]
                        dump_row_helper(self, [str(i) for i in indices], v, prefix='')
    
                if self.writer is not None:
                    self.vi_helper.dump_table(self.vi, [[str(out['iteration'])]],
                                              table_name='', col_names=['Epoch #'])
    
    def validate(self, dataloader):
        """
        Validation loop to visualize outputs.
        
        Args:
            dataloader (DataLoader): DataLoader for the validation dataset.
        """
        for ind, batch in enumerate(dataloader):
            data = process_batch(batch)
            with torch.no_grad():
                self._visualize_core(data, prefix="val")
    
                """
                Example data structure:
                'prompt' = ['A polyp, mytoken9 type1, mytoken1 color1, and pathology is mytoken2.']
                'gt_prompt' = ['A polyp, IIa type, red color, and pathology is tubular adenoma with low-grade epithelial dysplasia.']
                'gt_fruit' = ['IIa type']
                'gt_mat' = ['tubular adenoma with low-grade epithelial dysplasia']
                'gt_color' = ['red color']
                'gt_location' = ['location1']  # New
                'ph_fruit' = ['mytoken9']
                'ph_color' = ['mytoken1']
                'ph_mat' = ['mytoken2']
                'ph_location' = ['mytoken3']  # New
                'fruit_template' = ['A polyp, {} type']
                'color_template' = ['A polyp, {} color']
                'mat_template' = ['A polyp, and {}']
                'location_template' = ['A polyp, located at {}']  # New
                """
