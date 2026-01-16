from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings, CLIPEncoder, BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union, List
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
class CustomCLIPTextTransformer(AbstractEncoder):
    def __init__(self, config: CLIPTextConfig, max_length = 77):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.max_length = max_length

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,  # 新增参数
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        replace_token_ids: Optional[List[int]] = None,  # 用于替换的 token IDs
        new_embeddings: Optional[List[torch.Tensor]] = None,  # 新的嵌入，改为列表
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:
            hidden_states = inputs_embeds

        # 替换指定的 token 的嵌入
        if replace_token_ids is not None and new_embeddings is not None:
            for token_id, new_emb in zip(replace_token_ids, new_embeddings):
                mask = input_ids == token_id
                # Ensure new_emb is [hidden_size]
                if new_emb.dim() == 1:
                    hidden_states[mask] = new_emb
                elif new_emb.dim() == 2 and new_emb.size(0) == hidden_states.size(0):
                    hidden_states[mask] = new_emb[mask.nonzero(as_tuple=True)[0]]
                else:
                    raise ValueError("new_emb has incompatible shape")

        # 创建因果注意力掩码
        if input_ids is not None:
            input_shape = input_ids.size()
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # 扩展 attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CustomCLIPTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig, max_length = 77):
        super().__init__(config)
        self.text_model = CustomCLIPTextTransformer(config)
        self.max_length = max_length
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        replace_token_ids=None,
        new_embeddings=None,
        **kwargs,
    ):
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            replace_token_ids=replace_token_ids,
            new_embeddings=new_embeddings,
            **kwargs,
        )

# 使用示例
tokenizer = CLIPTokenizer.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14", max_length=77)
config = CLIPTextConfig.from_pretrained("/home/hyl/yujia/clip_new/clip-vit-large-patch14", max_length=77)
model = CustomCLIPTextModel(config)

text = "This is a sample text with some tokens to replace a a a ."
inputs = tokenizer(text, return_tensors="pt")

# 定义要替换的 token 和新的嵌入
replace_token = tokenizer.encode("sample", add_special_tokens=False)[0]
new_embedding = torch.randn(model.config.hidden_size)  # 生成新的嵌入

# 将新的嵌入调整为列表，每个 replace_token_id 对应一个新的嵌入
new_embeddings = [new_embedding]

# 前向传播时进行替换
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    replace_token_ids=[replace_token],
    new_embeddings=new_embeddings
)

# 获取最后的隐藏状态
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # 应输出 [batch_size, 77, 768]
