def masked_adain(content_feat, style_feat, content_mask, style_mask):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style_normalized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return content_feat * (1 - content_mask) + style_normalized_feat * content_mask


def adain(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)

    assert len(size) == 3
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 2
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std


# import torch

# T = torch.Tensor


# def expand_first(
#     feat: T,
#     scale=1.0,
# ) -> T:
#     b = feat.shape[0]
#     feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
#     if scale == 1:
#         feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
#     else:
#         feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
#         feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
#     return feat_style.reshape(*feat.shape)


# def concat_first(feat: T, dim=2, scale=1.0) -> T:
#     feat_style = expand_first(feat, scale=scale)
#     return torch.cat((feat, feat_style), dim=dim)


# def calc_mean_std(feat, eps: float = 1e-5):
#     feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
#     feat_mean = feat.mean(dim=-2, keepdims=True)
#     return feat_mean, feat_std


# def adain(feat: T) -> T:
#     feat_mean, feat_std = calc_mean_std(feat)
#     feat_style_mean = expand_first(feat_mean)
#     feat_style_std = expand_first(feat_std)
#     feat = (feat - feat_mean) / feat_std
#     feat = feat * feat_style_std + feat_style_mean
#     return feat
