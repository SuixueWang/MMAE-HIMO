# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

from util.options import get_args_parser_pretrain
args = get_args_parser_pretrain()
args = args.parse_args()
num_omics = args.num_omics

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # MAE encoder multi-omics multimodal
        self.encoder_omics = nn.Linear(80, self.embed_dim, bias=True)

        self.pos_embed = nn.Parameter(torch.zeros(1,
                                  (1024 // self.patch_embed.patch_size[0]) ** 2 + 1 + num_omics, self.embed_dim,),
                                    requires_grad=False)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            self.fc_norm1 = norm_layer(embed_dim)
            self.fc_norm2 = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.classifier1 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.classifier2 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        self.classifier = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.output_range = nn.Parameter(torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-4]), requires_grad=False)

    def forward_features(self, samples):
        x, X_mrna0, X_mirna0, X_meth0, X_path0 = samples[:]

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1 + num_omics:, :]

        # append omics token
        X_mrna = self.encoder_omics(X_mrna0).unsqueeze(1)
        X_mirna = self.encoder_omics(X_mirna0).unsqueeze(1)
        X_meth = self.encoder_omics(X_meth0).unsqueeze(1)
        X_path = self.encoder_omics(X_path0).unsqueeze(1)

        X_omics = torch.cat((X_mrna, X_mirna, X_meth, X_path), dim=1)
        X_omics = X_omics + self.pos_embed[:, 1:num_omics+1, :].expand(x.shape[0], -1, -1)
        x = torch.cat((X_omics, x), dim=1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = x[:, 5:, :]

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x1 = x[:,1:num_omics+1,:].mean(dim=1)  # global pool without cls token
            outcome1 = self.fc_norm(x1)

            x2 = x[:, num_omics+1:, :].mean(dim=1)  # global pool without cls token
            outcome2 = self.fc_norm(x2)

            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)

        else:
            x = self.norm(x)
            outcome = x[:, 0]

            outcome1, outcome2 = outcome, outcome

        return outcome1, outcome2, outcome

    def forward(self, x):
            x1, x2, x = self.forward_features(x)

            # x1 = self.classifier(x1)
            # x2 = self.classifier(x2)
            # x = self.classifier(x)

            corr = torch.nn.functional.cosine_similarity(x1, x2)

            return corr


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=1024, patch_size=128, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model