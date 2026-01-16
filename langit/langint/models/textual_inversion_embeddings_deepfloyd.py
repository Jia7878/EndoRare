import torch.nn as nn
import os
from langint.datasets.glide import placeholder_words_list
import torch

# from langint.utils.deepfloyd_no_diffusers import CACHE_DIR
# from langint.third_party.deepfloyd.deepfloyd_if.modules.clip_text_encode       r import clip_text_encoderEmbedder
from typing import List, Dict, Optional
import logging
from datetime import datetime
# from langint.test import FrozenCLIPEmbedder
from langint.ldm.modules.encoders.modules import FrozenCLIPEmbedder
from transformers import CLIPTokenizer, CLIPTextModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
class Embeddings(nn.Module):
    def __init__(self, num_placeholder_words: int, initializer_word: Optional[str], num_placeholder_groups: int = 2, shared_tokens=0, gt_init=0, blip_guidance=-1, fruit_blip_coeff=0, mat_blip_coeff=0, color_blip_coeff=0):
        super().__init__()
        self.blip_guidance = blip_guidance
        self.fruit_blip_coeff = fruit_blip_coeff
        self.mat_blip_coeff = mat_blip_coeff
        self.color_blip_coeff = color_blip_coeff
        self.clip_text_encoder=clip_text_encoder=FrozenCLIPEmbedder(device="cuda")
        # tokenizer = clip_text_encoder.tokenizer
        # text_encoder = clip_text_encoder.transformer
        self.tokenizer = tokenizer = CLIPTokenizer.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        self.text_encoder= text_encoder = CLIPTextModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        self.max_length = 77
        text_encoder.to(device)
        text_encoder.eval()

        assert shared_tokens in [0, 1], shared_tokens
        assert gt_init in [0, 1], gt_init
        assert num_placeholder_groups == 3, num_placeholder_groups

        num_placeholder_tokens = num_placeholder_words * num_placeholder_groups
        print('num_placeholder_words, groups, tokens', num_placeholder_words, num_placeholder_groups, num_placeholder_tokens)
        assert initializer_word is None, initializer_word

        # get placeholder token (string) and token id (indices in the embedding)
        placeholder_tokens: List[str] = placeholder_words_list[:num_placeholder_tokens]
        for placeholder_token in placeholder_tokens:
            assert clip_text_encoder.text_preprocessing(placeholder_token) == placeholder_token, (placeholder_token, clip_text_encoder.text_preprocessing(placeholder_token))

        # https://huggingface.co/docs/transformers/v4.29.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_tokens
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            raise ValueError(f'Expected to add {len(placeholder_tokens)} tokens, got {num_added_tokens}')
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        logger.info(f'placeholder tokens: {str(placeholder_tokens)}')
        logger.info(f'placeholder tokens encoded: {str(tokenizer.encode(" ".join(placeholder_tokens)))}')
        logger.info(f'placeholder tokens encoded decoded: {str(tokenizer.decode(tokenizer.encode(" ".join(placeholder_tokens))))}')
        logger.info(f'placeholder token ids: {str(placeholder_token_ids)}')
        logger.info(f'placeholder tokens recon: {str(tokenizer.convert_ids_to_tokens(placeholder_token_ids))}')

        # get initializing token (string) and token id (indices in the embedding)
        assert initializer_word is None
        initializer_embs = []
        
        if initializer_word is None:
            embs_mean = text_encoder.get_input_embeddings().weight.data.mean(0)
            self.embs_mean = embs_mean.clone()

            initializer_embs = [
                embs_mean.clone() + torch.randn_like(embs_mean) * 0.01
                for _ in range(num_placeholder_tokens)
            ]

            for emb in initializer_embs:
                # print(emb.shape)
                assert emb.shape == (768,), emb.shape
            
            assert len(initializer_embs) == num_placeholder_tokens, (num_placeholder_tokens, len(initializer_embs), initializer_embs)
        else:
            initializer_tokens: List[str] = initializer_word.split(',')
            if len(initializer_tokens) == 1:
                initializer_tokens = initializer_tokens * num_placeholder_tokens
            assert len(initializer_tokens) == num_placeholder_tokens, (len(initializer_tokens), num_placeholder_tokens)
            initializer_token_ids: List[List[int]] = [tokenizer.encode(initializer_tokens[ind], add_special_tokens=False) for ind in range(num_placeholder_tokens)]
            assert len(initializer_token_ids) == len(initializer_tokens), (len(initializer_token_ids), len(initializer_tokens))
            logger.info(f'init tokens: {str(initializer_tokens)}')
            logger.info(f'init tokens encoded: {str(tokenizer.encode(" ".join(initializer_tokens)))}')
            logger.info(f'init tokens encoded decoded: {str(tokenizer.decode(tokenizer.encode(" ".join(initializer_tokens))))}')
            logger.info(f'init token ids: {str(initializer_token_ids)}')
            for ind in range(num_placeholder_tokens):
                assert len(initializer_token_ids[ind]) == 1, (initializer_tokens[ind], len(initializer_token_ids[ind]))

            initializer_embs = [
                text_encoder.get_input_embeddings().weight.data[initializer_token_ids[ind]].clone()
                for ind in range(num_placeholder_tokens)
            ]

            
        self.usage_counts = [0]

        text_encoder.requires_grad_(False)

        self.trainable_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind])
            for ind in range(num_placeholder_tokens)
        })
        self.initial_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind].clone()) 
            for ind in range(num_placeholder_tokens)
        })
        self.placeholder_token_to_id = {
            placeholder_tokens[ind]: placeholder_token_ids[ind]
            for ind in range(num_placeholder_tokens)
        }

        self.num_placeholder_groups = num_placeholder_groups
        self.num_placeholder_words = num_placeholder_words

        self.iteration = 0

        self.pos_linear_fruit = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_fruit = nn.Dropout(0.2)
        self.act_fruit = nn.LeakyReLU()
        self.final_linear_fruit = nn.Linear(1024, 768)

        self.pos_linear_mat = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_mat = nn.Dropout(0.2)
        self.act_mat = nn.LeakyReLU()
        self.final_linear_mat = nn.Linear(1024,768)

        self.pos_linear_color = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_color = nn.Dropout(0.2)
        self.act_color = nn.LeakyReLU()
        self.final_linear_color = nn.Linear(1024, 768)

        self.blip_mse_loss = nn.MSELoss()
        self.blip_cos_loss = nn.CosineEmbeddingLoss()



    def _pass_through_layers(self, x, data):
        outputs = []
        for i in range(12):
            outputs.append(self.dropout_fruit(self.pos_linear_fruit[i](x[:, :, i, :]))) #【1，3，6，768」  【
        x_fruit = torch.stack(outputs, dim=2)
        
        x_fruit = x_fruit.mean(dim=1)
        x_fruit = x_fruit.mean(dim=1)

        x_fruit = self.act_fruit(x_fruit)

        x_fruit = self.final_linear_fruit(x_fruit)

        outputs = []
        for i in range(12):
            outputs.append(self.dropout_mat(self.pos_linear_mat[i](x[:, :, i, :])))
        x_mat = torch.stack(outputs, dim=2)

        x_mat = x_mat.mean(dim=1)
        x_mat = x_mat.mean(dim=1)

        x_mat = self.act_mat(x_mat)

        x_mat = self.final_linear_mat(x_mat)

        outputs = []
        for i in range(12):
            outputs.append(self.dropout_color(self.pos_linear_color[i](x[:, :, i, :])))
        x_color = torch.stack(outputs, dim=2)
        
        x_color = x_color.mean(dim=1)
        x_color = x_color.mean(dim=1)

        x_color = self.act_color(x_color)

        x_color = self.final_linear_color(x_color)       #*（1，1024）

        x = torch.stack((x_fruit, x_color, x_mat), dim=1)  #*（1，3，1024）
        return x


    def _process_clip_features(self, data, tokenizer, text_encoder):
        blip_word_dict = {}
        assert len(data['blip_color']) == len(data['prompt']), (data['blip_color'], data['prompt'])
        # gt_prompt是我输入的命令行
        # prompt是mytoken组成的句子
        ph_words_by_prompt = []
        for i in range(len(data['prompt'])):
            prompt = data['prompt'][i]
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None, None]
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    assert ph_words_for_curr_prompt[int(number) % 3] == None, (ph_words_for_curr_prompt[int(number) % 3], prompt)
                    if int(number) % 3 == 0: 
                        ph_words_for_curr_prompt[0] = word
                        blip_fruit_token = tokenizer.encode(data['blip_fruit'][i], add_special_tokens=False)
                        blip_fruit_emb = text_encoder.get_input_embeddings().weight.data[blip_fruit_token].clone()
                        # blip_fruit_emb = self.clip_text_encoder.forward(data['blip_fruit'][i])
                        blip_fruit_emb = blip_fruit_emb # PyTorch
                        # assert len(blip_fruit_emb) == 1, (len(blip_fruit_emb), data['blip_fruit'][i], word)
                        # assert blip_fruit_emb.shape == (768,), (blip_fruit_emb.shape, data['blip_fruit'][i], word)
                        blip_word_dict[word] = blip_fruit_emb#【768
                    elif int(number) % 3 == 1:
                        ph_words_for_curr_prompt[1] = word
                        blip_color_token = tokenizer.encode(data['blip_color'][i], add_special_tokens=False)
                        blip_color_emb = text_encoder.get_input_embeddings().weight.data[blip_color_token].clone()
                        # blip_color_emb = self.clip_text_encoder.forward(data['blip_color'][i])
                        blip_color_emb = blip_color_emb # PyTorch
                        # blip_color_emb = blip_color_emb[0, 0, :]
                        # assert len(blip_color_emb) == 1, (len(blip_color_emb), data['blip_color'][i], word)
                        # assert blip_color_emb.shape == (768,), (blip_color_emb.shape, data['blip_color'][i], word)
                        blip_word_dict[word] = blip_color_emb
                    elif int(number) % 3 == 2:
                        ph_words_for_curr_prompt[2] = word
                        blip_mat_token = tokenizer.encode(data['blip_mat'][i], add_special_tokens=False)
                        blip_mat_emb = text_encoder.get_input_embeddings().weight.data[blip_mat_token].clone()
                        # blip_mat_emb = self.clip_text_encoder.forward(data['blip_mat'][i])
                        blip_mat_emb = blip_mat_emb  # PyTorch
                        # blip_mat_emb = blip_mat_emb[0, 0, :]
                        # assert len(blip_mat_emb) == 1, (len(blip_mat_emb), data['blip_mat'][i], word)
                        # assert blip_mat_emb.shape == (768,), (blip_mat_emb.shape, data['blip_mat'][i], word)
                        blip_word_dict[word] = blip_mat_emb
                    
            assert ph_words_for_curr_prompt[0] != None, prompt
            assert ph_words_for_curr_prompt[1] != None, prompt 
            assert ph_words_for_curr_prompt[2] != None, prompt 
            ph_words_by_prompt.append(ph_words_for_curr_prompt)#['mytoken3', 'mytoken4', 'mytoken5']
        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))

        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))
        x = data['clip_feature'].clone().to('cuda')
        x = self._pass_through_layers(x, data) #*将三个编码器得到的向量堆叠在一起
        
        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))

        temp_ph_word_dict = {}

        for ph_words, emb in zip(ph_words_by_prompt, x):
            assert len(emb) == len(ph_words) == 3, (len(emb), len(ph_words), ph_words)
            ph_fruit, ph_color, ph_mat = ph_words
            emb_fruit, emb_color, emb_mat = emb

            assert len(self.trainable_embeddings[ph_fruit]) == len(emb_fruit), (len(self.trainable_embeddings[ph_fruit]), len(emb_fruit))
            self.trainable_embeddings[ph_fruit] = nn.Parameter(emb_fruit.to(torch.float32).clone())
            temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.float32)

            assert len(self.trainable_embeddings[ph_mat]) == len(emb_mat), (len(self.trainable_embeddings[ph_mat]), len(emb_mat))
            self.trainable_embeddings[ph_mat] = nn.Parameter(emb_mat.to(torch.float32).clone())
            temp_ph_word_dict[ph_mat] = emb_mat.to(torch.float32)
            
            assert len(self.trainable_embeddings[ph_color]) == len(emb_color), (len(self.trainable_embeddings[ph_color]), len(emb_color))
            self.trainable_embeddings[ph_color] = nn.Parameter(emb_color.to(torch.float32).clone())
            temp_ph_word_dict[ph_color] = emb_color.to(torch.float32)

        return temp_ph_word_dict, blip_word_dict

    def _get_text_embeddings(self, texts, clip_text_encoder, tokenizer, model, data):
        # 使用clip_text_encoder的text_preprocessing方法预处理文本
        texts = [clip_text_encoder.text_preprocessing(text) for text in texts]

        # 使用tokenizer将文本转换为input_ids和attention_mask
        text_tokens_and_mask = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length = self.max_length,
                                         return_overflowing_tokens=False,return_length=True)         

        # 将token IDs转移到GPU设备
        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)  

        # 获取CLIP模型的嵌入
        with torch.no_grad():
            # 使用clip_text_encoder获取输入嵌入（embedding）
            # inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = model.get_input_embeddings()(input_ids)

 # 如果你希望直接使用transformer返回的嵌入

        # 如果是训练模式，处理占位符特征
        if self.training:
            temp_ph_word_dict, blip_word_dict = self._process_clip_features(data, tokenizer, model)

        # 初始化BLIP损失
        fruit_blip_loss = []
        mat_blip_loss = []
        color_blip_loss = []

        # 迭代处理占位符token
        for i in range(len(self.placeholder_token_to_id)):
            token = f'mytoken{i}'
            assert token in self.placeholder_token_to_id, (token, self.placeholder_token_to_id)
            token_id = self.placeholder_token_to_id[token]
            
            if token_id in input_ids:
                if self.training:
                    # 保存原始 inputs_embeds 用于后续检查
                    # 获取替换位置的布尔掩码
                    mask = (input_ids == token_id)

                    # 确保 token_id 和 mask 在正确的设备上
                    mask = mask.to(input_ids.device)

                    # 获取嵌入向量，确保与 inputs_embeds 的最后一维匹配
                    embedding = temp_ph_word_dict[token].to(torch.float32)

                    # 扩展嵌入以匹配 inputs_embeds 的形状
                    embedding_expanded = embedding.unsqueeze(0).unsqueeze(0).expand(mask.size(0), mask.size(1), -1)

                    # 使用 torch.where 替换对应位置的嵌入
                    inputs_embeds = torch.where(mask.unsqueeze(-1), embedding_expanded, inputs_embeds)

                    # 计算BLIP损失（根据占位符位置）
                    if self.blip_guidance == 0:
                        if i % 3 == 0:
                            fruit_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[token], blip_word_dict[token]))
                        if i % 3 == 1:
                            color_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[token], blip_word_dict[token]))
                        if i % 3 == 2:
                            mat_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[token], blip_word_dict[token]))
                            
                else:


                    # 获取替换位置的布尔掩码
                    mask = (input_ids == token_id)

                    # 确保 token_id 和 mask 在正确的设备上
                    mask = mask.to(input_ids.device)

                    # 获取嵌入向量，确保与 inputs_embeds 的最后一维匹配
                    embedding = self.trainable_embeddings[token].to(torch.float32)

                    # 扩展嵌入以匹配 inputs_embeds 的形状
                    embedding_expanded = embedding.unsqueeze(0).unsqueeze(0).expand(mask.size(0), mask.size(1), -1)

                    # 使用 torch.where 替换对应位置的嵌入
                    inputs_embeds = torch.where(mask.unsqueeze(-1), embedding_expanded, inputs_embeds)

        inputs_embeds = model.text_model.embeddings(inputs_embeds = inputs_embeds)

        with torch.no_grad():
            # 获取文本嵌入
            outputs = model(
                input_ids = input_ids,                
                inputs_embeds = inputs_embeds
                )
            # 通常使用最后一个隐藏层的平均作为句子的表示
        text_encoder_embs = outputs.last_hidden_state
        # 计算BLIP损失的平均值
        fruit_blip_loss = (sum(fruit_blip_loss) / len(fruit_blip_loss)).float() if fruit_blip_loss else 0.0
        mat_blip_loss = (sum(mat_blip_loss) / len(mat_blip_loss)).float() if mat_blip_loss else 0.0
        color_blip_loss = (sum(color_blip_loss) / len(color_blip_loss)).float() if color_blip_loss else 0.0

        return text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss, mat_blip_loss


    def forward(self, data, return_all=False, return_pre_gumbel=False) -> Dict[str, torch.Tensor]:
        if self.training:   #!你在哪里冒出来的
            self.iteration += 1

            now = datetime.now()
        
        clip_text_encoder = self.clip_text_encoder
        tokenizer = self.tokenizer
        model = self.text_encoder
        # follow clip_text_encoder.get_text_embeddings
        texts: List[str] = data['prompt']
        ret = {}

        text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss, mat_blip_loss = self._get_text_embeddings(texts, clip_text_encoder, tokenizer, model, data) #要改

        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        ret['embeddings'] = text_encoder_embs
        ret['fruit_blip_loss'] = fruit_blip_loss * self.fruit_blip_coeff 
        ret['mat_blip_loss'] = mat_blip_loss * self.mat_blip_coeff 
        ret['color_blip_loss'] = color_blip_loss * self.color_blip_coeff 
        ret['iteration'] = self.iteration
        if return_all:
            input_ids_list = []
            for ind in range(len(data['prompt'])):
                input_ids = text_tokens_and_mask['input_ids'][ind]
                attention_mask = text_tokens_and_mask['attention_mask'][ind]
                input_ids = input_ids[:attention_mask.sum()]
                input_ids_list.append(input_ids)
            ret['input_ids']: List[torch.Tensor] = input_ids_list
            ret['processed_prompts']: List[str] = texts


        return ret
    
    def _inference_forward(self, data):
        # 获取clip_text_encoder（即FrozenCLIPEmbedder）
        clip_text_encoder = self.clip_text_encoder
        tokenizer = self.tokenizer
        model = self.text_encoder
        
        texts: List[str] = data['prompt']
        ret = {}

        # 使用clip_text_encoder的text_preprocessing方法预处理文本
        texts = [clip_text_encoder.text_preprocessing(text) for text in texts]
        
        text_tokens_and_mask = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length = self.max_length,
                                         return_overflowing_tokens=False,return_length=True)      
        # 使用tokenizer将文本转换为input_ids和attention_mask
        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)  
        

        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(input_ids)

        # 获取占位符词汇（mytokenX）与其对应的嵌入向量
        ph_words_by_prompt = []
        for prompt in data['prompt']:
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None, None]
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    assert ph_words_for_curr_prompt[int(number) % 3] == None, (ph_words_for_curr_prompt[int(number) % 3], prompt)
                    ph_words_for_curr_prompt[int(number) % 3] = word
            ph_words_by_prompt.append(ph_words_for_curr_prompt)

        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))
        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))

        # 获取clip_feature并经过必要的处理
        x = data['clip_feature'].clone().to('cuda')
        x = self._pass_through_layers(x, data)  # 通过层进行处理（例如：线性层等）

        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))
        
        # 构建占位符词汇到其对应嵌入的字典
        temp_ph_word_dict = {}
        for ph_word, emb in zip(ph_words_by_prompt, x):
            assert len(ph_word) == len(emb) == 3, (len(ph_word), len(emb), ph_word)
            ph_fruit, ph_color, ph_mat = ph_word
            emb_fruit, emb_color, emb_mat = emb  # clip encode via three linear layers
            if ph_fruit is not None:
                temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.float32)
            if ph_color is not None:
                temp_ph_word_dict[ph_color] = emb_color.to(torch.float32)
            if ph_mat is not None:
                temp_ph_word_dict[ph_mat] = emb_mat.to(torch.float32)
            

        # 更新占位符的嵌入
        for i in range(len(self.placeholder_token_to_id)):
            token = f'mytoken{i}'
            token_id = self.placeholder_token_to_id[token]

            if token_id in input_ids:
                  # 获取原始嵌入
                # inputs_embeds[input_ids == token_id] = temp_ph_word_dict[token]  # 更新占位符的嵌入
                mask = (input_ids == token_id)

                    # 确保 token_id 和 mask 在正确的设备上
                mask = mask.to(input_ids.device)

                # 获取嵌入向量，确保与 inputs_embeds 的最后一维匹配
                embedding = temp_ph_word_dict[token].to(torch.float32)

                # 扩展嵌入以匹配 inputs_embeds 的形状
                embedding_expanded = embedding.unsqueeze(0).unsqueeze(0).expand(mask.size(0), mask.size(1), -1)

                # 使用 torch.where 替换对应位置的嵌入
                inputs_embeds = torch.where(mask.unsqueeze(-1), embedding_expanded, inputs_embeds)
                
                # # are_equal = torch.all(torch.eq(input_embeds_from_embeds.cuda(), inputs_embeds.cuda()))
                 # # print(are_equal)        
        
        # 使用模型获取最终的文本编码（嵌入）
        # with torch.no_grad():
        #     text_encoder_embs = model(
        #         inputs_embeds=inputs_embeds,
        #         input_ids=None,
        #         # attention_mask=text_tokens_and_mask['attention_mask'].cuda(),
        #     )
        inputs_embeds = model.text_model.embeddings(inputs_embeds = inputs_embeds)

        with torch.no_grad():
            # 获取文本嵌入
            outputs = model(
                input_ids = input_ids,                
                inputs_embeds = inputs_embeds
                )
            # 通常使用最后一个隐藏层的平均作为句子的表示
        text_encoder_embs = outputs.last_hidden_state
        
        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        # 返回推理结果
        ret['embeddings'] = text_encoder_embs
        return ret
