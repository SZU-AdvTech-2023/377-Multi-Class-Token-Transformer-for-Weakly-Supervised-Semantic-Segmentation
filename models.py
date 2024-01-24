import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from torch.autograd import Variable
import math
from ResNetV2 import ResNetV2

__all__ = [
    'deit_small_MCTformerV1_patch16_224', 'deit_small_MCTformerV2_patch16_224'
]


class MCTformerV2(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))
        self.maxpool_conv = nn.MaxPool2d(2)
        self.resnet_num_layers = (3, 4, 9)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)
        self.hybrid_model = ResNetV2(block_units=self.resnet_num_layers, width_factor=1)
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=self.embed_dim,
                                       kernel_size=1,
                                       stride=1)
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes  #216-20
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


    def forward_features(self, x, get_att = False, n=12):
        B, nc, w, h = x.shape
        if get_att == False:
            x, features = self.hybrid_model(x)  # x:torch.Size([64, 1024, 14, 14])  len(features)=3
            x = self.patch_embeddings(x)   #  x: torch.Size([64, 384, 14, 14])
            x = x.flatten(2).transpose(1, 2)
        else:
            x = self.patch_embed(x)   # img_size (224, 224)  patch_size (16, 16)  in_chans 3   embed_dim 384  x: torch.Size([64, 196, 384])

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []
        class_embeddings = []
        patch_embeddings = []
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
            class_embeddings.append(x[:, 0:self.num_classes])
            patch_embeddings.append(x[:, self.num_classes:])
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, class_embeddings, patch_embeddings


    def forward(self, x, return_att=False, n_layers=12, get_att = False, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, class_embeddings, patch_embeddings = self.forward_features(x, get_att=get_att)  #  #x_cls:64,20,384  x_patch:64,196,384
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])   #x_patch:64,14,14,384
        x_patch = x_patch.permute([0, 3, 1, 2])   #x_patch:64,384,14,14
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)  #x_patch:64,20,14,14
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)  #x_patch_logits:64,20

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape
        #p2c = attn_weights[-n_layers:].sum(0)[:, self.num_classes:, 0:self.num_classes].reshape([n, c, h, w])  # [64,196,20].reshape[64,20,14,14]
        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])  #[64,20,196].reshape[64,20,14,14]
        # mtatt1 = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, 0:self.num_classes]  # [64,20,20]
        # print(mtatt1)

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
            cams = torch.sqrt(cams)
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:] #12,64,196,196
        patch_attn = patch_attn + patch_attn.permute(0, 1, 3, 2)
        x_cls_logits = x_cls.mean(-1)

        # w_featmap = w 
        # h_featmap = h 
        # patch_attn = torch.sum(patch_attn, dim=0)
        # refine_cams = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],cams.shape[1], -1, 1)).reshape(cams.shape[0],cams.shape[1], w_featmap, h_featmap)
        c2c = attn_weights[:, :, 0:self.num_classes, 0:self.num_classes]  # 12,64,20,20
        if return_att:
            x_logits = (x_cls_logits + x_patch_logits) / 2
            return x_logits, cams, patch_attn     # patch_attn torch.Size([12, 64, 196, 196])    cams torch.Size([64, 20, 14, 14])
        else:
            return x_cls_logits, x_patch_logits, torch.stack(class_embeddings), torch.stack(patch_embeddings)


@register_model
def deit_small_MCTformerV2_patch16_224(pretrained=False, **kwargs):
    print('load deit small MCTformerV2')
    print(f'args: {kwargs}')
    model = MCTformerV2(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# @register_model
# def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
#     print('load deit small MCTformerV2')
#     print(f'args: {kwargs}')
#     model = MCTformerV1(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])

#     return model

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss


model = MCTformerV2(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=21, drop_rate=0.0, drop_path_rate=0.1)
print(model)
import torch
print(torch.__version__)