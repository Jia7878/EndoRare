import PIL
from tu.trainers.simple_trainer import BaseTrainer
from tu.utils.training import process_batch
import numpy as np
import torch
from tu.utils.visualize import dump_helper, dump_row_helper
from torchvision import transforms

class Trainer(BaseTrainer):
    """训练器，用于计算损失然后优化等等（主要有关训练的部分在 BaseTrainer 中定义）。
    """
    
    def _visualize_core(self, data, prefix=None):
        if 'dataset' in data:
            dataset = data['dataset']
            # dump_helper(self, 'full_dataset', dataset.images * .5 + .5, prefix=prefix)
            # dump_helper(self, 'full_dataset', dataset.images, prefix=prefix)
            if self.writer is not None:
                self.vi_helper.dump_table(self.vi, [[str([sublist for sublist in dataset.ph_words_all[i]]), dataset.gt_prompts[i//dataset.num_data_per_prompt], dataset.blip_fruits[i//dataset.num_data_per_prompt], dataset.blip_colors[i//dataset.num_data_per_prompt], dataset.blip_mats[i//dataset.num_data_per_prompt]] for i in range(len(dataset.ph_words_all))],
                                      table_name='', col_names=[f'{prefix}/ph_words', f'{prefix}/gt_prompt', f'{prefix}/blip_fruit', f'{prefix}/blip_color', f'{prefix}/blip_mat']) 
                
                assert self.title is not None
        model = self.modules[self.module_key]
        model.eval()
        to_tensor = transforms.ToTensor()
        with torch.no_grad():
            if 'val' in prefix:
                zero_image = torch.zeros_like(data['image'][0]).unsqueeze(0)
                images = []

                # Ground Truth Prompt生成
                out = model({'prompt': data['gt_prompt']}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(data['gt_prompt']), 1, 1, 1), 'prompt': data['gt_prompt']}
                img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                
                # 转换PIL图像到Tensor，如果是PIL图像
                if isinstance(img, PIL.Image.Image):
                    img = to_tensor(img).unsqueeze(0)
                images.append(img)

                # 普通 Prompt生成
                out = model({'prompt': data['prompt']}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(data['prompt']), 1, 1, 1), 'prompt': data['prompt']}
                img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                
                # 转换PIL图像到Tensor
                if isinstance(img, PIL.Image.Image):
                    img = to_tensor(img).unsqueeze(0)
                images.append(img)

                # 生成水果、材质、颜色等不同提示词
                fruit_prompts = []
                color_prompts0 = []
                color_prompts1 = []
                color_prompts2 = []
                mat_prompts = []

                
                for ph_fruit, ph_mat, ph_color, fruit_template, mat_template, color_template0, color_template1, color_template2 in zip(
                        data['ph_fruit'], data['ph_mat'], data['ph_color'], data['fruit_template'], data['mat_template'],
                        data['color_template0'], data['color_template1'], data['color_template2']):
                    fruit_prompts.append(fruit_template.format(ph_fruit))
                    mat_prompts.append(mat_template.format(ph_mat))
                    color_prompts0.append(color_template0.format(ph_color))
                    color_prompts1.append(color_template1.format(ph_color))
                    color_prompts2.append(color_template2.format(ph_color))

                # 使用水果提示词生成图像
                out = model({'prompt': fruit_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(fruit_prompts), 1, 1, 1), 'prompt': fruit_prompts}
                img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                if isinstance(img, PIL.Image.Image):
                    img = to_tensor(img).unsqueeze(0)
                images.append(img)

                # 使用材质提示词生成图像
                out = model({'prompt': mat_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(mat_prompts), 1, 1, 1), 'prompt': mat_prompts}
                img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                if isinstance(img, PIL.Image.Image):
                    img = to_tensor(img).unsqueeze(0)
                images.append(img)

                # 使用颜色提示词生成图像
                out = model({'prompt': color_prompts1}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(color_prompts1), 1, 1, 1), 'prompt': color_prompts1}
                img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                if isinstance(img, PIL.Image.Image):
                    img = to_tensor(img).unsqueeze(0)
                images.append(img)

                # 继续组合其他图像生成和拼接
                nrow = len(images)
                images = [tensor.unsqueeze(1) for tensor in images]
                images = torch.cat(images, dim=1)
                reshape_size = [-1] + list(images[0].shape[1:])
                images = images.view(*reshape_size)

                # 打印重建的图像
                # dump_helper(self, 'reconstruction', images * .5 + .5, prefix=prefix, nrow=nrow)
                dump_helper(self, 'reconstruction', images, prefix=prefix, nrow=nrow)

                # 打印文本信息
                if self.writer is not None:
                    self.vi_helper.dump_table(self.vi, [[data['gt_prompt'][0]], [data['prompt'][0] + f' (blip_fruit: {data["blip_fruit"][0]})' + f' (blip_mat: {data["blip_mat"][0]})' + f' (blip_color: {data["blip_color"][0]})'], [fruit_prompts[0]], [mat_prompts[0]], [color_prompts1[0]]],
                                            table_name='', col_names=[f'{prefix}'])
                
                # 生成组合图像（Composition）
                # comp_fruits = [] 
                # comp_imgs = [] 
                # gt_mats = []
                # gt_colors = []
                # gt_comp_prompts = []
                # assert len(data['all_gt_colors']) == len(data['all_gt_mats']), (len(data['all_gt_colors']), len(data['all_gt_mats']))
                # for i in range(len(data['all_gt_colors'])):
                #     curr_gt_mat = data['all_gt_mats'][i][0]
                #     curr_ph_mat = data['all_ph_mats'][i][0]
                #     curr_gt_color = data['all_gt_colors'][i][0]
                #     curr_ph_color = data['all_ph_colors'][i][0]
                #     if curr_ph_color != data['ph_color'][0] and curr_ph_mat != data['ph_mat'][0]:
                #         gt_mats.append(curr_gt_mat)
                #         gt_colors.append(curr_gt_color)
                #         gt_comp_prompts.append(data['full_template'][0].format(data['gt_fruit'][0], curr_gt_mat, curr_gt_color))
                #         comp_fruits.append(data['full_template'][0].format(data['ph_fruit'][0], curr_ph_mat, curr_ph_color))
                        
                # # print(comp_fruits)
                # # 使用组合水果提示词生成图像
                # out = model({'prompt': comp_fruits}, return_all=True)
                # num_cross_sets = 2
                # for i in range(num_cross_sets):
                #     fake_data = {'image': zero_image.repeat(len(comp_fruits), 1, 1, 1), 'prompt': comp_fruits}
                #     img = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                #     if isinstance(img, PIL.Image.Image):
                #         img = to_tensor(img).unsqueeze(0)
                #     comp_imgs.append(img)

                # # 拼接组合图像
                # nrow = len(comp_fruits)
                # comp_imgs = torch.cat(comp_imgs, dim=0)

                # dump_helper(self, 'composition', comp_imgs, prefix=prefix, nrow=nrow)
                # # dump_helper(self, 'composition', comp_imgs * .5 + .5, prefix=prefix, nrow=nrow)

                # # 打印组合图像的文本
                # if self.writer is not None:
                #     self.vi_helper.dump_table(self.vi, [[gt_comp_prompts[i], comp_fruits[i]] for i in range(len(gt_comp_prompts))],
                #                             table_name='', col_names=[f'{prefix}/gt', f'{prefix}/inverted'])
                
            elif 'inference' in prefix:
                data['image'] = torch.stack(data['image'])
                data['clip_feature'] = torch.stack(data['clip_feature'])
                images = []
                
                # images.append(data['image'])
                for _ in range(1):
                    out = model._inference_forward(data)
                    images_pil = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, data)['image']
                    images_ = transforms.ToTensor()(images_pil)
                    images_ = images_.unsqueeze(0).to("cuda") #! embedding有很多维度呢？
                    images.append(images_)

                for _ in range(1):
                    fake_data = {'clip_feature': data['clip_feature'], 'prompt': data['fruit_prompt'], 'image': data['image']}
                    out = model._inference_forward(fake_data)
                    images_pil = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                    images_ = transforms.ToTensor()(images_pil)
                    images_ = images_.unsqueeze(0).to("cuda")
                    images.append(images_)


                for _ in range(1):
                    fake_data = {'clip_feature': data['clip_feature'], 'prompt': data['mat_prompt'], 'image': data['image']}
                    out = model._inference_forward(fake_data)
                    
                    images_pil = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                    images_ = transforms.ToTensor()(images_pil)
                    images_ = images_.unsqueeze(0).to("cuda")
                    images.append(images_)


                for _ in range(1):
                    fake_data = {'clip_feature': data['clip_feature'], 'prompt': data['color_prompt'], 'image': data['image']}
                    out = model._inference_forward(fake_data)
                    images_pil = self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image']
                    images_ = transforms.ToTensor()(images_pil)
                    images_ = images_.unsqueeze(0).to("cuda")
                    images.append(images_)


                images = torch.cat(images, dim=0)

                dump_helper(self, '', images, prefix=prefix, nrow=len(data['gt_prompt']))
                if self.writer is not None:
                    self.vi_helper.dump_table(self.vi, [[data['gt_prompt'][i], data['prompt'][i], data['fruit_prompt'][i], data['mat_prompt'][i], data['color_prompt'][i]] for i in range(len(data['gt_prompt']))],
                                            table_name='', col_names=[f'{prefix}/gt', f'{prefix}/inverted', f'{prefix}/fruit_ph',  f'{prefix}/mat_ph',  f'{prefix}/color_ph'])
            else:
                dump_helper(self, 'input', data['image'] * .5 + .5, prefix=prefix)
                out = model(data, return_all=True)
                for k, v in self.loss_modules['textual_inversion'].visualize(out, data).items():
                    if isinstance(v, torch.Tensor) and v.ndim == 4:
                        dump_helper(self, k, v * .5 + .5, prefix=prefix)
                    elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        if len(v) > 10:
                            indices = np.linspace(0, len(v), 10, endpoint=False, dtype=int)
                        else:
                            indices = np.arange(len(v))
                        v = [v[i] * .5 + .5 for i in indices]
                        dump_row_helper(self, [str(i) for i in indices], v, prefix=prefix)

                if self.writer is not None:
                    self.vi_helper.dump_table(self.vi, [[out['processed_prompts'][ind], data['gt_prompt'][ind], data['blip_fruit'][ind], data['blip_color'][ind], data['blip_mat'][ind]] for ind in range(len(data['prompt']))],
                                            table_name='', col_names=[f'{prefix}/prompt', f'{prefix}/gt_prompt', f'{prefix}/blip_fruit', f'{prefix}/blip_color', f'{prefix}/blip_mat'])

    def validate(self, dataloader):
        # do inference first for easier viewing
        with torch.no_grad():
            batch = next(iter(dataloader))
            old_data = process_batch(batch)
            data = {}
            for k, v in old_data['inf'].items():
                data[k] = [w[0] for w in v]
            self._visualize_core(data, prefix=f"inference")
        """
        'image' =
        [tensor([[[-2.0888, -2.1365, -2.0997,  ..., -1.9758, -1.8827, -1.9552],
                [-2.1....8330, -2.8725]]],
            device='cuda:0')]
        'prompt' =
        ['a mytoken0 polyp with the color mytoken1 and the pathology mytoken2']
        'gt_prompt' =
        ['a gloves polyp with the color winter and the pathology red']
        'fruit_prompt' =
        ['a photo of a mytoken0']
        'mat_prompt' =
        ['a photo of the mytoken1 season']
        'color_prompt' =
        ['a photo of the color mytoken2']
        'clip_feature' =
        [tensor([[[ 0.0559,  0.3421, -0.6166,  ...,  0.5008,  0.0493,  0.2747],
                [-0.1....7087, -0.0225]]],
            device='cuda:0')]
        len() =
        7
        """
        for ind, batch in enumerate(dataloader):
            if ind >= 2:
                break
            data = process_batch(batch)
            with torch.no_grad():
                self._visualize_core(data, prefix=f"val/{data['gt_fruit'][0]}-{data['gt_color'][0]}-{data['gt_mat'][0]}")

        
