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



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("/home/hyl/yujia/mask_to_image/bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=3,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=2,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        self.linear = nn.Linear(
            in_features=1024, # 1024
            out_features=768, # 640
            bias=True)
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        # x=x.permute(0, 3, 1, 2)
        for stage in range(self.n_stages):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).to('cuda').float()
            x = self.interpolator(x, scale_factor=self.multiplier)
            
            # print("conditions:",x.shape)


        if self.remap_output:
            x = x.to(dtype=torch.float32)
            x = self.channel_mapper(x)
            # print("remap:",x.shape) [32,4,32,32]
        mask_re = rearrange(x, 'b c h w -> b c (h w)')
        seg_mask = self.linear(mask_re)
        return seg_mask

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=100, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize
        self.linear_layer = nn.Linear(768, 1280)
        self.linear_layer.weight.requires_grad = False  # 不训练权重
        self.linear_layer.bias.requires_grad = False

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text,truncate=True).to(self.device)
        z = self.model.encode_text(tokens)
        # print("z:",z.shape)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        # z = self.linear_layer(z)
        # print("z:",z.shape)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        # z = self.linear_layer(z)
        
        return z
class FrozenCLIPEmbedder(AbstractEncoder):
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="/home/hyl/yujia/clip_new/clip-vit-large-patch14", device="cuda", max_length=77,torch_dtype=None):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
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

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        # z = outputs
        # print("z:",z.shape)
        return z

    def encode(self, text):
        return self(text)

class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


class PromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.seg_mask_encoder = SpatialRescaler()
        self.text_encoder = FrozenCLIPEmbedder()

    def forward(self, input):
        if input['segmentation'] is not None:
            seg_mask = input['segmentation']
            mask_emb = self.seg_mask_encoder(seg_mask)
            prompt_encoding = mask_emb
        if input['text_data'] is not None:
            text = input['text_data']
            text_emb = self.text_encoder(text)
            prompt_encoding = text_emb

        return prompt_encoding

    def encode(self, input):
        return self(input)
