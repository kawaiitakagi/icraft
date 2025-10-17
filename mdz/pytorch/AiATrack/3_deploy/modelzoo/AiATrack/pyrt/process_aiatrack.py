import torch
import numpy as np
import math
import cv2
import re
#===== 网络超参数 ======#
NET_NAME = "aiatrack_backbone_boxhead_cat4_revised_BN_boxmm"
MODELFD = "AiATrack"
SEARCH_FACTOR = 5.0
SEARCH_SIZE = 320
FEAT_SIZE = SEARCH_SIZE // 16
CACHE_SIZE = 100
ENSEMBLE = 4
IOU_THRESH = 0.8


def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter=',', fmt='%d')


def save_rawbb(file, data):
    tracked_bb = np.array(data)
    np.savetxt(file, tracked_bb, delimiter=',', fmt="%.12f")

def numeric_sort_key(s):
    return int(re.search(r'\d+', s).group())

def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.tensor(b, dtype=torch.float32)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.tensor(b, dtype=torch.float32)


def map_box_back(state: list, pred_box: list, resize_factor: float):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * SEARCH_SIZE / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor = torch.Tensor([SEARCH_SIZE, SEARCH_SIZE])) -> torch.Tensor:
    """
    Transform the box coordinates from the original image coordinates to the coordinates of the cropped image.

    Args:
        box_in: The box for which the coordinates are to be transformed.
        box_extract: The box about which the image crop has been extracted.
        resize_factor: The ratio between the original image scale and the scale of the image crop.
        crop_sz: Size of the cropped image.

    Returns:
        torch.Tensor: Transformed coordinates of box_in.
    """

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out / crop_sz[0]


def update_ref(pos_emb, search_mem, outputs_coord, refer_cache):
    refer_mem_cache = refer_cache["mem"]
    refer_reg_cache = refer_cache["reg"]
    refer_pos_cache = refer_cache["pos"]
    if len(refer_mem_cache) == CACHE_SIZE:
        _ = refer_mem_cache.pop(1)
        _ = refer_reg_cache.pop(1)
        _ = refer_pos_cache.pop(1)
    feat_size = FEAT_SIZE
    target_region = torch.zeros((feat_size, feat_size))
    x, y, w, h = (outputs_coord * feat_size).round().int()
    target_region[max(y, 0):min(y + h, feat_size), max(x, 0):min(x + w, feat_size)] = 1
    target_region = target_region.view(feat_size * feat_size, -1)
    background_region = 1 - target_region
    refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0)
    refer_mem_cache.append(search_mem)
    refer_reg_cache.append(refer_region)
    refer_pos_cache.append(pos_emb)

    refer_mem_list = [refer_mem_cache[0]]
    refer_reg_list = [refer_reg_cache[0]]
    refer_pos_list = [refer_pos_cache[0]]
    max_idx = len(refer_mem_cache) - 1
    for part in range(ENSEMBLE):
        refer_mem_list.append(refer_mem_cache[max_idx * (part + 1) // ENSEMBLE])
        refer_reg_list.append(refer_reg_cache[max_idx * (part + 1) // ENSEMBLE])
        refer_pos_list.append(refer_pos_cache[max_idx * (part + 1) // ENSEMBLE])
    
    refer_cache["mem"] = refer_mem_cache
    refer_cache["reg"] = refer_reg_cache
    refer_cache["pos"] = refer_pos_cache

    refer_mem1 = torch.cat(refer_mem_list[1:], dim=0)
    refer_reg1 = torch.cat(refer_reg_list[1:], dim=1)
    refer_pos0 = torch.repeat_interleave(refer_pos_list[0], 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)
    refer_pos1 = torch.repeat_interleave(torch.cat(refer_pos_list[1:], dim=0), 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)

    return [refer_mem_list[0].contiguous().numpy().astype(np.float32), refer_mem1.contiguous().numpy().astype(np.float32),
            refer_reg_list[0], refer_reg1,
            refer_pos0.contiguous().numpy().astype(np.float32), refer_pos1.contiguous().numpy().astype(np.float32)], \
           refer_cache



def sample_target(im, target_bb, search_area_factor=SEARCH_FACTOR, output_sz=SEARCH_SIZE):
    """
    Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area.

    Args:
        im: cv image.
        target_bb: Target box [x, y, w, h].
        search_area_factor: Ratio of crop size to target size.
        output_sz (float): Size to which the extracted crop is resized (always square). If None, no resizing is done.

    Returns:
        cv image: Extracted crop.
        float: The factor by which the crop has been resized to make the crop size equal output_size.
    """

    x, y, w, h = target_bb

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    # if crop_sz < 1:
    #     raise Exception('ERROR: too small bounding box')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    # Deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    resize_factor = output_sz / crop_sz
    im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
    return im_crop_padded, resize_factor, att_mask


# def img_preprocess(img_arr: np.ndarray, amask_arr: np.ndarray):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
#     std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
#     # Deal with the image patch
#     img_tensor = torch.tensor(img_arr).float().permute((2, 0, 1)).unsqueeze(dim=0)
#     img_tensor_norm = ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
#     # Deal with the attention mask
#     amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).unsqueeze(dim=0)  # (1,H,W)
#     return img_tensor_norm, amask_tensor


# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """

#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
#         # position_embedding = PositionEmbeddingSine(128, normalize=True)
#         # inner_position_embedding = PositionEmbeddingSine(32, normalize=True)
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError('ERROR: normalize should be True if scale is passed')
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale

def positionEmbeddingSine(num_pos_feats, imgmsk):
    temperature = 10000
    scale = 2 * math.pi
    mask = imgmsk
    # assert mask is not None
    not_mask = ~mask  # (b,h,w)
    # 1 1 1 1... 2 2 2 2... 3 3 3 3...
    y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Cumulative sum along axis 1 (h axis) --> (b,h,w)
    # 1 2 3 4... 1 2 3 4... 1 2 3 4...
    x_embed = not_mask.cumsum(2, dtype=torch.float32)  # Cumulative sum along axis 2 (w axis) --> (b,h,w)
    # if self.normalize:
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale  # 2pi * (y / sigma(y))
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale  # 2pi * (x / sigma(x))

    # num_pos_feats = d/2
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)  # (0,1,2,...,d/2)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
    pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
    return pos  # (b,d,h,w)