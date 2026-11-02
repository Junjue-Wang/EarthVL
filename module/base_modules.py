"""Building blocks shared by the EarthVLNet branches (module/EarthVLNet.py).

Adapted code (see THIRD_PARTY_NOTICES.md): ``MMProjector`` follows the LLaVA multi-modal projector of Hugging Face
transformers and ``rank0_print`` / ``find_all_linear_names`` come from LLaVA (Apache-2.0); ``FeedForward``,
``Attention`` and ``Transformer`` from vit-pytorch (MIT); ``FC``, ``MLP`` and ``AttFlat`` from MCAN-VQA (Apache-2.0);
``LayerNorm2d`` from torchvision (BSD-3-Clause). ``ChannelAttention`` is our implementation of the channel attention
described in the CBAM paper.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_INDEX = -1  # counting target of an answer without a number


class ChannelAttention(nn.Module):
    """Channel attention as in CBAM (Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018), with GELU in
    the bottleneck. The global average and the global maximum of every channel go through the same two 1x1
    convolutions (``fc1`` reduces the channels by ``ratio``, ``fc2`` restores them); the sigmoid of the sum of both
    outputs reweights the channels of the input."""

    def __init__(self, in_planes, ratio=16, dropout=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout)

    def channel_logits(self, pooled):
        return self.fc2(F.gelu(self.fc1(pooled)))

    def forward(self, x):
        # B x C x 1 x 1 descriptors (adaptive_max_pool2d passes the gradient of tied maxima to one position, amax not)
        weights = torch.sigmoid(self.channel_logits(F.adaptive_avg_pool2d(x, 1)) +
                                self.channel_logits(F.adaptive_max_pool2d(x, 1)))
        return self.dropout(x * weights)


class OGA(nn.Module):
    """Object-guided attention: channel attention with a residual connection."""

    def __init__(self, inchannel, cam_ratio=16):
        super(OGA, self).__init__()
        self.pre_cam = ChannelAttention(inchannel, cam_ratio, dropout=0.)

    def forward(self, img_feat):
        img_feat = self.pre_cam(img_feat) + img_feat

        return img_feat


class MMProjector(nn.Module):
    """Multi-modal projector: Linear - GELU - Linear."""

    def __init__(self, vision_hidden_size, text_hidden_size):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def rank0_print(*args):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def find_all_linear_names(model):
    """Names of the nn.Linear layers of the language model that get LoRA adapters (all but lm_head)."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower', 'vision_resampler', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size, flat_glimpses=1, flat_out_size=10, dropout=0.1):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            dropout_r=dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )
        self.flat_glimpses = flat_glimpses

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e4
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class Transformer(nn.Module):
    """Pre-norm Transformer encoder followed by attention pooling (AttFlat) into ``out_num`` classes."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1, out_num=24):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.counting_head = AttFlat(dim, dim, flat_out_size=out_num, dropout=dropout)

    def forward(self, x):
        for attn, ff in self.layers:
            x = self.dropout1(attn(x)) + x
            x = self.dropout2(ff(x)) + x
        x = self.norm(x)
        out_logits = self.counting_head(x)

        return out_logits


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW tensor."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
