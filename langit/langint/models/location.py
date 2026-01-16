import torch
import torch.nn as nn
import os
from langint.datasets.glide_location import placeholder_words_list
import torch
from typing import List, Dict, Optional
import logging
from datetime import datetime
from langint.ldm.modules.encoders.modules import FrozenCLIPEmbedder
from transformers import CLIPTokenizer, CLIPTextModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Embeddings(nn.Module):
    def __init__(
        self, 
        num_placeholder_words: int, 
        initializer_word: Optional[str], 
        num_placeholder_groups: int = 4,  # Updated from 3 to 4
        shared_tokens=0, 
        gt_init=0, 
        blip_guidance=-1, 
        fruit_blip_coeff=0, 
        mat_blip_coeff=0, 
        color_blip_coeff=0,
        location_blip_coeff=0  # New coefficient for location
    ):
        super().__init__()
        self.blip_guidance = blip_guidance
        self.fruit_blip_coeff = fruit_blip_coeff
        self.mat_blip_coeff = mat_blip_coeff
        self.color_blip_coeff = color_blip_coeff
        self.location_blip_coeff = location_blip_coeff  # New
        
        # Initialize CLIP text encoder
        self.clip_text_encoder = FrozenCLIPEmbedder(device=device)
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14").to(device)
        self.text_encoder.eval()
        
        self.max_length = 77

        assert shared_tokens in [0, 1], shared_tokens
        assert gt_init in [0, 1], gt_init
        assert num_placeholder_groups == 4, num_placeholder_groups  # Updated to 4

        num_placeholder_tokens = num_placeholder_words * num_placeholder_groups
        print('num_placeholder_words, groups, tokens', num_placeholder_words, num_placeholder_groups, num_placeholder_tokens)
        assert initializer_word is None, initializer_word

        # Get placeholder tokens and their IDs
        placeholder_tokens: List[str] = placeholder_words_list[:num_placeholder_tokens]
        for placeholder_token in placeholder_tokens:
            assert self.clip_text_encoder.text_preprocessing(placeholder_token) == placeholder_token, (placeholder_token, self.clip_text_encoder.text_preprocessing(placeholder_token))

        # Add placeholder tokens to the tokenizer
        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            raise ValueError(f'Expected to add {len(placeholder_tokens)} tokens, got {num_added_tokens}')
        
        # Resize token embeddings to accommodate new tokens
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        logger.info(f'placeholder tokens: {str(placeholder_tokens)}')
        logger.info(f'placeholder tokens encoded: {str(self.tokenizer.encode(" ".join(placeholder_tokens)))}')
        logger.info(f'placeholder tokens encoded decoded: {str(self.tokenizer.decode(self.tokenizer.encode(" ".join(placeholder_tokens))))}')
        logger.info(f'placeholder token ids: {str(placeholder_token_ids)}')
        logger.info(f'placeholder tokens recon: {str(self.tokenizer.convert_ids_to_tokens(placeholder_token_ids))}')

        # Initialize placeholder embeddings
        initializer_embs = []
        if initializer_word is None:
            embs_mean = self.text_encoder.get_input_embeddings().weight.data.mean(0)
            self.embs_mean = embs_mean.clone()

            initializer_embs = [
                embs_mean.clone() + torch.randn_like(embs_mean) * 0.01
                for _ in range(num_placeholder_tokens)
            ]

            for emb in initializer_embs:
                assert emb.shape == (768,), emb.shape
            assert len(initializer_embs) == num_placeholder_tokens, (num_placeholder_tokens, len(initializer_embs), initializer_embs)
        else:
            # Handle initializer_word if provided (not covered here as initializer_word is None)
            pass

        self.usage_counts = [0]

        # Freeze the CLIP text encoder
        self.text_encoder.requires_grad_(False)

        # Initialize trainable embeddings for placeholders
        self.trainable_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind].clone(), requires_grad=False)  # Set requires_grad=False
            for ind in range(num_placeholder_tokens)
        })
        
        # Store initial embeddings (optional, if needed)
        self.initial_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind].clone(), requires_grad=False)  # Set requires_grad=False
            for ind in range(num_placeholder_tokens)
        })

        self.placeholder_token_to_id = {
            placeholder_tokens[ind]: placeholder_token_ids[ind]
            for ind in range(num_placeholder_tokens)
        }

        self.num_placeholder_groups = num_placeholder_groups
        self.num_placeholder_words = num_placeholder_words

        self.iteration = 0

        # Initialize encoder modules for each attribute
        # Fruit Encoder
        self.pos_linear_fruit = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_fruit = nn.Dropout(0.2)
        self.act_fruit = nn.LeakyReLU()
        self.final_linear_fruit = nn.Linear(1024, 768)

        # Mat Encoder
        self.pos_linear_mat = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_mat = nn.Dropout(0.2)
        self.act_mat = nn.LeakyReLU()
        self.final_linear_mat = nn.Linear(1024, 768)

        # Color Encoder
        self.pos_linear_color = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_color = nn.Dropout(0.2)
        self.act_color = nn.LeakyReLU()
        self.final_linear_color = nn.Linear(1024, 768)

        # Location Encoder (New)
        self.pos_linear_location = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_location = nn.Dropout(0.2)
        self.act_location = nn.LeakyReLU()
        self.final_linear_location = nn.Linear(1024, 768)

        # Loss functions
        self.blip_mse_loss = nn.MSELoss()
        self.blip_cos_loss = nn.CosineEmbeddingLoss()

    def _pass_through_layers(self, x, data):
        """
        Pass the CLIP features through the respective encoder layers for each attribute.
        """
        # Fruit Encoding
        outputs_fruit = []
        for i in range(12):
            outputs_fruit.append(self.dropout_fruit(self.pos_linear_fruit[i](x[:, :, i, :])))
        x_fruit = torch.stack(outputs_fruit, dim=2)
        x_fruit = x_fruit.mean(dim=1).mean(dim=1)
        x_fruit = self.act_fruit(x_fruit)
        x_fruit = self.final_linear_fruit(x_fruit)

        # Mat Encoding
        outputs_mat = []
        for i in range(12):
            outputs_mat.append(self.dropout_mat(self.pos_linear_mat[i](x[:, :, i, :])))
        x_mat = torch.stack(outputs_mat, dim=2)
        x_mat = x_mat.mean(dim=1).mean(dim=1)
        x_mat = self.act_mat(x_mat)
        x_mat = self.final_linear_mat(x_mat)

        # Color Encoding
        outputs_color = []
        for i in range(12):
            outputs_color.append(self.dropout_color(self.pos_linear_color[i](x[:, :, i, :])))
        x_color = torch.stack(outputs_color, dim=2)
        x_color = x_color.mean(dim=1).mean(dim=1)
        x_color = self.act_color(x_color)
        x_color = self.final_linear_color(x_color)

        # Location Encoding (New)
        outputs_location = []
        for i in range(12):
            outputs_location.append(self.dropout_location(self.pos_linear_location[i](x[:, :, i, :])))
        x_location = torch.stack(outputs_location, dim=2)
        x_location = x_location.mean(dim=1).mean(dim=1)
        x_location = self.act_location(x_location)
        x_location = self.final_linear_location(x_location)

        # Stack all attribute encodings
        x = torch.stack((x_fruit, x_color, x_mat, x_location), dim=1)  # Shape: (batch_size, 4, 768)
        return x

    def _process_clip_features(self, data, tokenizer, text_encoder):
        """
        Process the CLIP features to update placeholder embeddings and compute BLIP losses.
        """
        blip_word_dict = {}
        assert len(data['blip_color']) == len(data['prompt']), (data['blip_color'], data['prompt'])
        
        ph_words_by_prompt = []
        for i in range(len(data['prompt'])):
            prompt = data['prompt'][i]
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None, None, None]  # Updated for 4 groups
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    group_index = int(number) % self.num_placeholder_groups  # Updated for 4 groups
                    assert ph_words_for_curr_prompt[group_index] is None, (ph_words_for_curr_prompt[group_index], prompt)
                    ph_words_for_curr_prompt[group_index] = word
                    if group_index == 0:  # Fruit
                        blip_fruit_token = tokenizer.encode(data['blip_fruit'][i], add_special_tokens=False)
                        blip_fruit_emb = text_encoder.get_input_embeddings().weight.data[blip_fruit_token].clone()
                        blip_word_dict[word] = blip_fruit_emb
                    elif group_index == 1:  # Color
                        blip_color_token = tokenizer.encode(data['blip_color'][i], add_special_tokens=False)
                        blip_color_emb = text_encoder.get_input_embeddings().weight.data[blip_color_token].clone()
                        blip_word_dict[word] = blip_color_emb
                    elif group_index == 2:  # Mat
                        blip_mat_token = tokenizer.encode(data['blip_mat'][i], add_special_tokens=False)
                        blip_mat_emb = text_encoder.get_input_embeddings().weight.data[blip_mat_token].clone()
                        blip_word_dict[word] = blip_mat_emb
                    elif group_index == 3:  # Location (New)
                        blip_location_token = tokenizer.encode(data['blip_location'][i], add_special_tokens=False)
                        blip_location_emb = text_encoder.get_input_embeddings().weight.data[blip_location_token].clone()
                        blip_word_dict[word] = blip_location_emb

            assert all(word is not None for word in ph_words_for_curr_prompt), prompt
            ph_words_by_prompt.append(ph_words_for_curr_prompt)
        
        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))
        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))

        x = data['clip_feature'].clone().to(device)
        x = self._pass_through_layers(x, data)  # Pass through encoder layers

        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))

        temp_ph_word_dict = {}

        for ph_words, emb in zip(ph_words_by_prompt, x):
            assert len(emb) == len(ph_words) == 4, (len(emb), len(ph_words), ph_words)  # Updated for 4 groups
            ph_fruit, ph_color, ph_mat, ph_location = ph_words
            emb_fruit, emb_color, emb_mat, emb_location = emb

            # Update embeddings (Set requires_grad=False)
            with torch.no_grad():
                self.trainable_embeddings[ph_fruit].data = emb_fruit.to(torch.float32).clone()
                temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.float32)

                self.trainable_embeddings[ph_color].data = emb_color.to(torch.float32).clone()
                temp_ph_word_dict[ph_color] = emb_color.to(torch.float32)

                self.trainable_embeddings[ph_mat].data = emb_mat.to(torch.float32).clone()
                temp_ph_word_dict[ph_mat] = emb_mat.to(torch.float32)
                
                self.trainable_embeddings[ph_location].data = emb_location.to(torch.float32).clone()
                temp_ph_word_dict[ph_location] = emb_location.to(torch.float32)

        # Compute BLIP losses if required
        fruit_blip_loss = []
        mat_blip_loss = []
        color_blip_loss = []
        location_blip_loss = []  # New

        if self.blip_guidance == 0:
            for word in blip_word_dict:
                group_index = int(word.split("mytoken")[1]) % self.num_placeholder_groups
                if group_index == 0:
                    fruit_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[word], blip_word_dict[word]))
                elif group_index == 1:
                    color_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[word], blip_word_dict[word]))
                elif group_index == 2:
                    mat_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[word], blip_word_dict[word]))
                elif group_index == 3:
                    location_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[word], blip_word_dict[word]))

        # Aggregate losses
        fruit_blip_loss = (sum(fruit_blip_loss) / len(fruit_blip_loss)).float() if fruit_blip_loss else 0.0
        mat_blip_loss = (sum(mat_blip_loss) / len(mat_blip_loss)).float() if mat_blip_loss else 0.0
        color_blip_loss = (sum(color_blip_loss) / len(color_blip_loss)).float() if color_blip_loss else 0.0
        location_blip_loss = (sum(location_blip_loss) / len(location_blip_loss)).float() if location_blip_loss else 0.0

        return temp_ph_word_dict, blip_word_dict, fruit_blip_loss, color_blip_loss, mat_blip_loss, location_blip_loss  # Updated

    def _get_text_embeddings(self, texts, clip_text_encoder, tokenizer, model, data):
        """
        Generate text embeddings with optional BLIP guidance.
        """
        # Preprocess texts
        texts = [clip_text_encoder.text_preprocessing(text) for text in texts]

        # Tokenize texts
        text_tokens_and_mask = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length,
            return_overflowing_tokens=False,
            return_length=True
        )         

        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)  

        # Get input embeddings
        inputs_embeds = model.get_input_embeddings()(input_ids)

        # If training, process CLIP features and compute BLIP losses
        if self.training:
            temp_ph_word_dict, blip_word_dict, fruit_blip_loss, color_blip_loss, mat_blip_loss, location_blip_loss = self._process_clip_features(data, tokenizer, model)
        else:
            fruit_blip_loss = 0.0
            mat_blip_loss = 0.0
            color_blip_loss = 0.0
            location_blip_loss = 0.0

        # Replace placeholder embeddings with trainable ones if not training
        if not self.training:
            for i in range(len(self.placeholder_token_to_id)):
                token = f'mytoken{i}'
                token_id = self.placeholder_token_to_id[token]
                if token_id in input_ids:
                    mask = (input_ids == token_id).to(device)
                    embedding = self.trainable_embeddings[token].to(torch.float32)
                    embedding_expanded = embedding.unsqueeze(0).unsqueeze(0).expand(mask.size(0), mask.size(1), -1)
                    inputs_embeds = torch.where(mask.unsqueeze(-1), embedding_expanded, inputs_embeds)

        # Pass through the text encoder
        outputs = model(
            input_ids=input_ids,                
            inputs_embeds=inputs_embeds
        )

        text_encoder_embs = outputs.last_hidden_state

        # Compute average BLIP losses
        fruit_blip_loss = fruit_blip_loss * self.fruit_blip_coeff 
        mat_blip_loss = mat_blip_loss * self.mat_blip_coeff 
        color_blip_loss = color_blip_loss * self.color_blip_coeff 
        location_blip_loss = location_blip_loss * self.location_blip_coeff  # New

        return text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss, mat_blip_loss, location_blip_loss

    def forward(self, data, return_all=False, return_pre_gumbel=False) -> Dict[str, torch.Tensor]:
        if self.training:
            self.iteration += 1
            now = datetime.now()

        clip_text_encoder = self.clip_text_encoder
        tokenizer = self.tokenizer
        model = self.text_encoder

        texts: List[str] = data['prompt']
        ret = {}

        text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss, mat_blip_loss, location_blip_loss = self._get_text_embeddings(texts, clip_text_encoder, tokenizer, model, data)

        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        ret['embeddings'] = text_encoder_embs
        ret['fruit_blip_loss'] = fruit_blip_loss 
        ret['mat_blip_loss'] = mat_blip_loss 
        ret['color_blip_loss'] = color_blip_loss 
        ret['location_blip_loss'] = location_blip_loss  # New
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
        """
        Inference mode forward pass.
        """
        clip_text_encoder = self.clip_text_encoder
        tokenizer = self.tokenizer
        model = self.text_encoder

        texts: List[str] = data['prompt']
        ret = {}

        # Preprocess texts
        texts = [clip_text_encoder.text_preprocessing(text) for text in texts]
        
        # Tokenize texts
        text_tokens_and_mask = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length,
            return_overflowing_tokens=False,
            return_length=True
        )      

        input_ids = text_tokens_and_mask['input_ids'].to(device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(device)  

        # Get input embeddings
        inputs_embeds = model.get_input_embeddings()(input_ids)

        # Process placeholders
        ph_words_by_prompt = []
        for prompt in data['prompt']:
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None, None, None]  # Updated for 4 groups
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    group_index = int(number) % self.num_placeholder_groups  # Updated for 4 groups
                    ph_words_for_curr_prompt[group_index] = word
            ph_words_by_prompt.append(ph_words_for_curr_prompt)

        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))
        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))

        # Get and process CLIP features
        x = data['clip_feature'].clone().to(device)
        x = self._pass_through_layers(x, data)  # Pass through encoder layers

        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))

        temp_ph_word_dict = {}
        for ph_word, emb in zip(ph_words_by_prompt, x):
            assert len(emb) == len(ph_word) == 4, (len(emb), len(ph_word), ph_word)  # Updated for 4 groups
            ph_fruit, ph_color, ph_mat, ph_location = ph_word
            emb_fruit, emb_color, emb_mat, emb_location = emb

            if ph_fruit is not None:
                temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.float32)
            if ph_color is not None:
                temp_ph_word_dict[ph_color] = emb_color.to(torch.float32)
            if ph_mat is not None:
                temp_ph_word_dict[ph_mat] = emb_mat.to(torch.float32)
            if ph_location is not None:
                temp_ph_word_dict[ph_location] = emb_location.to(torch.float32)

        # Update embeddings for placeholders
        for i in range(len(self.placeholder_token_to_id)):
            token = f'mytoken{i}'
            token_id = self.placeholder_token_to_id[token]

            if token_id in input_ids:
                mask = (input_ids == token_id).to(device)
                embedding = temp_ph_word_dict[token].to(torch.float32)
                embedding_expanded = embedding.unsqueeze(0).unsqueeze(0).expand(mask.size(0), mask.size(1), -1)
                inputs_embeds = torch.where(mask.unsqueeze(-1), embedding_expanded, inputs_embeds)

        # Pass through the text encoder
        outputs = model(
            input_ids=input_ids,                
            inputs_embeds=inputs_embeds
        )

        text_encoder_embs = outputs.last_hidden_state

        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        ret['embeddings'] = text_encoder_embs
        return ret

