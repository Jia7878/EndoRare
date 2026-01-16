import html
import ftfy
import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia
from typing import Type
from transformers import CLIPTokenizer, CLIPTextModel
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
import re
# import html
import urllib.parse as ul

# import ftfy
# import torch
from bs4 import BeautifulSoup

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torch.nn as nn

class CustomCLIPTextModel(CLIPTextModel):
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # 如果传入了 inputs_embeds，直接使用它
        if inputs_embeds is not None:
            return self.text_model.embeddings(inputs_embeds=inputs_embeds, **kwargs)
        
        # 否则使用 input_ids
        return super().forward(input_ids=input_ids, **kwargs)

class FrozenCLIPEmbedder(nn.Module):
    """Frozen CLIP Text Encoder that supports inputs_embeds"""
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')
    def __init__(self, version="/home/hyl/yujia/clip_new/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # 使用我们自定义的 CLIPTextModel
        self.transformer = CustomCLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.transformer.to(self.device)
        self.freeze()
    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)

        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()
    def basic_clean(self,text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
    def text_preprocessing(self, text):

            # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text
        
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, text=None, inputs_embeds=None, input_ids=None):
        # 如果传入了 text，则先将其转换为 input_ids
        if text is not None:
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)  # 将 tokens 移到 device 上
        elif input_ids is not None:
            tokens = input_ids.to(self.device)  # 确保 input_ids 在正确的 device 上
        elif inputs_embeds is not None:
            tokens = None  # 直接使用 inputs_embeds
        else:
            raise ValueError("Either text or input_ids/inputs_embeds must be provided.")

        # 如果使用 inputs_embeds，则直接传递给 transformer
        if inputs_embeds is not None:
            outputs = self.transformer(inputs_embeds=inputs_embeds)
        else:
            # 否则，使用 input_ids 进行编码
            outputs = self.transformer(input_ids=tokens)

        # 获取 CLIP 模型的输出嵌入
        if isinstance(outputs, torch.Tensor):
                # 如果是 Tensor 类型，直接返回
                z = outputs
        else:
                # 否则，访问 last_hidden_state
                z = outputs.last_hidden_state

        return z




def test_frozen_clip_embedder():
    # 创建一个 FrozenCLIPEmbedder 实例
    clip_embedder = FrozenCLIPEmbedder(device="cuda")

    # 输入文本
    texts = ["This is a test sentence.", "Another test sentence!"]

    # 使用 text 参数测试
    embedder_outputs_from_text = clip_embedder(text=texts)
    print("Outputs from text:", embedder_outputs_from_text.shape)

    # 使用随机嵌入测试
    random_embeds = torch.randn(2, 77, 768).to("cuda")  # (batch_size, max_length, embed_dim)
    embedder_outputs_from_embeds = clip_embedder(inputs_embeds=random_embeds)
    print("Outputs from inputs_embeds:", embedder_outputs_from_embeds.shape)

test_frozen_clip_embedder()

