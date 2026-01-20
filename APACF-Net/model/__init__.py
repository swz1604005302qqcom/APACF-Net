import torch.nn as nn

def get_fusion_model(model_name, image_size, patch_size, num_class=2):
    if model_name == 'ViTFusion':
        from model.MultiModalViT import MultiModalViT
        net = MultiModalViT( img_size = image_size,
                             patch_size = patch_size,
                             in_chans=3,
                             num_classes=num_class,
                             embed_dim=384,
                             depth=6,
                             num_heads=8,
                             mlp_ratio=4.,
                             qkv_bias=False,
                             qk_scale=None,
                             drop_rate=0.2,
                             attn_drop_rate=0.1,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             init_values=0.,
                             use_learnable_pos_emb=True,
                             init_scale=0.,
                             use_mean_pooling=True,
                             )

    return net
