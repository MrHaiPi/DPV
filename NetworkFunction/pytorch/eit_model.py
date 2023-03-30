"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=(224, 224), pool_size=4, in_c=3, embed_dim=768, norm_layer=None, kernel_size=3,
                 use_eit_p=True, kernel_strides=1):
        super().__init__()
        pool_size = (pool_size, pool_size)
        self.img_size = img_size
        self.grid_size = (int(np.ceil(img_size[0] / (pool_size[0] * kernel_strides))), int(np.ceil(img_size[1] / (pool_size[1] * kernel_strides))))
        self.patch_size = ((pool_size[0] * kernel_strides), (pool_size[1] * kernel_strides))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.use_eit_p = use_eit_p
        if use_eit_p:
            self.eit_p = EIT_P(in_c=in_c, embed_dim=embed_dim, kernel_size=kernel_size, kernel_strides=kernel_strides, pool_size=pool_size[0])
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        if self.use_eit_p:
            x = self.eit_p(x)
        else:
            x = self.proj(x)

        x = rearrange(x, 'B C H W -> B (H W) C')
        x = self.norm(x)
        return x


class EIT_P(nn.Module):
    def __init__(self, kernel_size=3, pool_size=4, kernel_strides=1, in_c=3, embed_dim=108):
        super().__init__()
        self.conv = nn.Conv2d(in_c, embed_dim, kernel_size=(kernel_size, kernel_size), stride=(kernel_strides, kernel_strides),
                              padding=(kernel_size // 2, kernel_size // 2))
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x


class EIT_T(nn.Module):
    def __init__(self, grid_size=(14, 14), embed_dim=200, kernel_size=3, eit_t_drop_ratio=0.):
        super().__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(kernel_size // 2, kernel_size // 2), bias=False, groups=embed_dim)
        self.drop = nn.Dropout(eit_t_drop_ratio)

    def forward(self, inputs):
        x = inputs[:, 1:, :]
        x = rearrange(x, 'B (H W) C -> B C H W', H=self.grid_size[0], W=self.grid_size[1])

        x = self.conv1(x)
        x = self.drop(x)

        x = rearrange(x, 'B C H W -> B (H W) C')
        x = torch.cat((rearrange(inputs[:, 0, :], 'B C -> B 1 C'), x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 grid_size,
                 dim,
                 num_heads,
                 depth,
                 layer,
                 kernel_size=3,
                 use_eit_t=True,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if use_eit_t:
            num = int(dim // num_heads * (layer + 1) / depth) * num_heads
            self.num_attn_dim = num
            self.num_eit_t_dim = dim - num
            if layer + 1 == depth:
                self.num_attn_dim = dim
                self.num_eit_t_dim = 0
        else:
            self.num_attn_dim = dim
            self.num_eit_t_dim = 0
        print(self.num_eit_t_dim)

        self.attn = Attention(dim=self.num_attn_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio) if self.num_attn_dim != 0 else None

        self.eit_t = EIT_T(grid_size=grid_size, embed_dim=self.num_eit_t_dim, kernel_size=kernel_size,
                           eit_t_drop_ratio=attn_drop_ratio) if self.num_eit_t_dim != 0 else None

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        input = x
        x = self.norm1(x)

        if self.num_eit_t_dim == 0:
            x = self.attn(x)
        elif self.num_attn_dim == 0:
            x = self.eit_t(x)
        else:
            x1 = self.eit_t(x[:, :, :self.num_eit_t_dim])
            x2 = self.attn(x[:, :, -self.num_attn_dim:])
            x = torch.cat((x1, x2), dim=2)

        x = input + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EIT(nn.Module):
    def __init__(self, img_size=32, kernel_size=3, pool_size=4, kernel_strides=1, in_c=3, num_classes=1000,
                 use_eit_p=True, use_eit_t=True, use_pos_em=False,
                 embed_dim=250, depth=5, num_heads=10, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 representation_size=None, drop_ratio=0.2, attn_drop_ratio=0.15, drop_path_ratio=0.2,
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            pool_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, kernel_size=kernel_size, kernel_strides=kernel_strides, pool_size=pool_size, in_c=in_c,
                                       embed_dim=embed_dim, use_eit_p=use_eit_p)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) if use_pos_em else None
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(grid_size=self.patch_embed.grid_size, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  kernel_size=3, norm_layer=norm_layer, act_layer=act_layer, depth=depth, layer=i, use_eit_t=use_eit_t)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_eit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]

        x = x + self.pos_embed if self.pos_embed is not None else x
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_eit_weights(m):
    """
    EIT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base(num_classes: int = 10,
             img_size: tuple = (32, 32),
             in_c: int = 3,
             use_eit_p: bool = False,
             use_eit_t: bool = False,
             use_pos_em: bool = True,
             has_logits: bool = True):

    model = EIT(img_size=img_size,
                in_c=in_c,
                kernel_size=8,
                use_eit_p=use_eit_p,
                use_eit_t=use_eit_t,
                use_pos_em=use_pos_em,
                embed_dim=768,
                depth=3,
                num_heads=12,
                drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0,
                representation_size=768 if has_logits else None,
                num_classes=num_classes)
    return model


def eit_mini(num_classes: int = 10,
             img_size: tuple = (32, 32),
             in_c: int = 3,
             use_eit_p: bool = True,
             use_eit_t: bool = True,
             use_pos_em: bool = False,
             has_logits: bool = True):

    model = EIT(img_size=img_size,
                in_c=in_c,
                kernel_size=16,
                pool_size=1,
                kernel_strides=5,
                use_eit_p=use_eit_p,
                use_eit_t=use_eit_t,
                use_pos_em=use_pos_em,
                embed_dim=250,
                depth=5,
                num_heads=10,
                drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0,
                representation_size=768 if has_logits else None,
                num_classes=num_classes)
    return model

def eit_tiny(num_classes: int = 10,
             img_size: tuple = (32, 32),
             in_c: int = 3,
             use_eit_p: bool = True,
             use_eit_t: bool = True,
             use_pos_em: bool = False,
             has_logits: bool = True):

    model = EIT(img_size=img_size,
                in_c=in_c,
                kernel_size=16,
                pool_size=1,
                kernel_strides=5,
                use_eit_p=use_eit_p,
                use_eit_t=use_eit_t,
                use_pos_em=use_pos_em,
                embed_dim=330,
                depth=8,
                num_heads=10,
                drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                representation_size=768 if has_logits else None,
                num_classes=num_classes)
    return model

def eit_base(num_classes: int = 10,
             img_size: tuple = (32, 32),
             in_c: int = 3,
             use_eit_p: bool = True,
             use_eit_t: bool = True,
             use_pos_em: bool = False,
             has_logits: bool = True):

    model = EIT(img_size=img_size,
                in_c=in_c,
                kernel_size=16,
                pool_size=2,
                kernel_strides=3,
                use_eit_p=use_eit_p,
                use_eit_t=use_eit_t,
                use_pos_em=use_pos_em,
                embed_dim=400,
                depth=10,
                num_heads=16,
                drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                representation_size=768 if has_logits else None,
                num_classes=num_classes)
    return model

