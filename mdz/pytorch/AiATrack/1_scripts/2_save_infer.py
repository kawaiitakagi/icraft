import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), './')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import torch
import torch.nn.functional as F
from lib.models.aiatrack.aiatrack import AIATRACK as AIATRACK_M
from lib.models.aiatrack.backbone import *
from lib.models.aiatrack.transformer import *
from lib.models.aiatrack.transformer import _get_activation_fn
from lib.models.aiatrack.head import *
from lib.models.aiatrack.position_encoding import PositionEmbeddingSine
from tracking.test import *
from lib.test.tracker.aiatrack import *
from lib.test.tracker.aiatrack import AIATRACK as AIATRACK_T
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_xywh


#----------------------------
# 1. AIATRACK.backbone相关内容修改
#----------------------------
#----------------------------
# 1.1 BackboneBase中self.body(IntermediateLayerGetter类)的forward函数：
#     返回的变量格式由Dict改为Tensor
#----------------------------
def layer3ft(self, x):
    for name, module in self.items():
        x = module(x)
    return x

IntermediateLayerGetter.forward = layer3ft

#----------------------------
# 1.2 替换BackboneBase的forward函数：
#     输入参数格式由NestedTensor改为Tensor；
#     返回的变量格式由Dict改为Tensor（即上文layer3ft函数）
#----------------------------
def BbBase_forward(self, imgft):  # revision for icraft
    xs = self.body(imgft)
    return xs

BackboneBase.forward = BbBase_forward

#----------------------------
# 1.3 替换Backbone类的__init__函数：
#     backbone-ResNet50中的norm_layer由FrozenBatchNorm2d改为nn.BatchNorm2d
#----------------------------
def Bb_init(self, name: str,
       train_backbone: bool,
       return_interm_layers: bool,
       dilation: bool,
       freeze_bn: bool):
    norm_layer = nn.BatchNorm2d
    # Here is different from the original DETR because we use feature from block3
    backbone = getattr(resnet_module, name)(
        replace_stride_with_dilation=[False, dilation, False],
        pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
    num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
    super(Backbone, self).__init__(backbone, train_backbone, num_channels, return_interm_layers)

Backbone.__init__ = Bb_init

#----------------------------
# 1.3 替换Joiner类的__init__和forward函数：
#     去除原backbone中的embedding函数；
#     输出的变量格式由Dict改为Tensor
#----------------------------
def Joiner_init(self, backbone, position_embedding, learned_embedding):
    super(Joiner, self).__init__(backbone)

def Joiner_forward(self, imgft):
    xsts = self[0](imgft)
    return xsts

Joiner.__init__ = Joiner_init
Joiner.forward = Joiner_forward

#----------------------------
# 1.4 替换AIATRACK类的adjust函数：
#     去除mask, pos_embed, inr_embed相关内容；
#     输入和输出格式统一改为Tensor
#----------------------------
def aiatrack_adjust(self, src_feat):
    # Reduce channel
    feat = self.bottleneck(src_feat)  # (B, C, H, W)
    # Adjust shapes
    feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
    return feat_vec

AIATRACK_M.adjust = aiatrack_adjust

#----------------------------
# 2. AIATRACK.transformer相关内容修改
#----------------------------
#----------------------------
# 2.1 替换TransformerEncoder&TransformerEncoderLayer类的forward函数：
#     去除输入参数中的src_key_padding_mask
#----------------------------
def encoderlayer_forward(self, src,
                         pos: Optional[Tensor] = None,
                         inr: Optional[Tensor] = None):
    q = k = self.with_pos_embed(src, pos)  # Add pos to src
    if self.divide_norm:
        # Encoder divide by norm
        q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
        k = k / torch.norm(k, dim=-1, keepdim=True)
    # src2 = self.self_attn(q, k, value=src)[0]
    src2 = self.self_attn(query=q, key=k, value=src, pos_emb=inr)[0]
    # Add and norm
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    # FFN
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    # Add and Norm
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src

def encoder_forward(self, src,
                    pos: Optional[Tensor] = None,
                    inr: Optional[Tensor] = None):
    output = src  # (HW,B,C)
    for stack, layer in enumerate(self.layers):
        output = layer(output, pos=pos, inr=inr)
    if self.norm is not None:
        output = self.norm(output)
    return output

TransformerEncoderLayer.forward = encoderlayer_forward
TransformerEncoder.forward = encoder_forward

#----------------------------
# 2.2 替换TransformerDecoder&TransformerDecoderLayer类的forward函数：
#     去除输入参数中的refer_msk_list；
#     将TransformerDecoderLayer中对refer_mem_list, refer_emb_list, refer_pos_list的torch.cat操作移至网络外作为前处理计算，并各自拆分为两部分输入TransformerDecoder, TransformerDecoderLayer
#----------------------------
def decoderlayer_forward(self, tgt, 
                         refer_mem_list0, refer_mem_list1,
                         refer_emb_list0, refer_emb_list1,
                         refer_pos_list0, refer_pos_list1):
    # Mutual attention
    mem_ensemble = refer_mem_list0
    emb_ensemble = refer_emb_list0
    refer_pos = refer_pos_list0
    refer_queries = tgt
    refer_keys = mem_ensemble
    refer_values = mem_ensemble + emb_ensemble
    if self.divide_norm:
        refer_queries = refer_queries / torch.norm(refer_queries, dim=-1, keepdim=True) * self.scale_factor
        refer_keys = refer_keys / torch.norm(refer_keys, dim=-1, keepdim=True)
    long_tgt_refer, long_attn_refer = self.long_term_attn(query=refer_queries,
                                                        key=refer_keys,
                                                        value=refer_values,
                                                        pos_emb=refer_pos)
    mem_ensemble = refer_mem_list1
    emb_ensemble = refer_emb_list1
    refer_pos = refer_pos_list1
    refer_queries = tgt
    refer_keys = mem_ensemble
    refer_values = mem_ensemble + emb_ensemble
    if self.divide_norm:
        refer_queries = refer_queries / torch.norm(refer_queries, dim=-1, keepdim=True) * self.scale_factor
        refer_keys = refer_keys / torch.norm(refer_keys, dim=-1, keepdim=True)
    short_tgt_refer, short_attn_refer = self.short_term_attn(query=refer_queries,
                                                            key=refer_keys,
                                                            value=refer_values,
                                                            pos_emb=refer_pos)
    tgt = tgt + self.dropout1_1(long_tgt_refer) + self.dropout1_2(short_tgt_refer)
    tgt = self.norm1(tgt)
    # FFN
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    # Add and Norm
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    return tgt

def decoder_forward(self, tgt, 
                    refer_mem_list0, refer_mem_list1,
                    refer_emb_list0, refer_emb_list1,
                    refer_pos_list0, refer_pos_list1):
    output = tgt
    for stack, layer in enumerate(self.layers):
        output = layer(output,
                       refer_mem_list0, refer_mem_list1,
                       refer_emb_list0, refer_emb_list1,
                       refer_pos_list0, refer_pos_list1)
    if self.norm is not None:
        output = self.norm(output)
    return output.unsqueeze(0)

TransformerDecoderLayer.forward = decoderlayer_forward
TransformerDecoder.forward = decoder_forward

#----------------------------
# 2.3 替换AiAModule及应用该模块的TransformerEncoderLayer和TransformerDecoderLayer类的__init__函数：
#     将CorrAttention类forward函数中对pos_emb的torch.repeat_interleave操作移至网络外作为前处理计算；
#     去除一些未使用的输入参数（对应2.1、2.2中去除参数），精简if/else语句
#----------------------------
from attention_for_save import AiAModule as AiAModuleTRACE

def encoderlayer_init(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                      activation='relu', normalize_before=False, divide_norm=False,
                      use_AiA=True, match_dim=64, feat_size=400):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = AiAModuleTRACE(d_model, nhead, dropout=dropout,
                                    use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.activation = _get_activation_fn(activation)
    self.normalize_before = normalize_before  # First normalization, then add
    self.divide_norm = divide_norm
    self.scale_factor = float(d_model // nhead) ** 0.5

def decoderlayer_init(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                      activation='relu', normalize_before=False, divide_norm=False,
                      use_AiA=True, match_dim=64, feat_size=400):
    super(TransformerDecoderLayer, self).__init__()
    self.long_term_attn = AiAModuleTRACE(d_model, nhead, dropout=dropout,
                                         use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
    self.short_term_attn = AiAModuleTRACE(d_model, nhead, dropout=dropout,
                                          use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1_1 = nn.Dropout(dropout)
    self.dropout1_2 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.activation = _get_activation_fn(activation)
    self.normalize_before = normalize_before
    self.divide_norm = divide_norm
    self.scale_factor = float(d_model // nhead) ** 0.5

TransformerEncoderLayer.__init__ = encoderlayer_init
TransformerDecoderLayer.__init__ = decoderlayer_init

#----------------------------
# 3. AIATRACK.box_head相关内容修改
#----------------------------
#----------------------------
# 3.1 替换Corner—_Predictor类的__init__, forward和soft_argmax函数
#----------------------------
# 3.1.1 在__init__函数中新增一个shape为(feat_sz*feat_sz, 1)、值为1.0的常量tensor
def CP_init(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
    super(Corner_Predictor, self).__init__()
    self.feat_sz = feat_sz
    self.stride = stride
    self.img_sz = self.feat_sz * self.stride
    # Top-left corner
    self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
    self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
    self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
    self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
    self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=(1, 1))
    # Bottom-right corner
    self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
    self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
    self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
    self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
    self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=(1, 1))
    # About coordinates and indexes
    with torch.no_grad():
        self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
        # Generate mesh-grid
        self.coord_x = self.indice.repeat((self.feat_sz, 1)).view((self.feat_sz * self.feat_sz,)).float()  # .cuda()
        self.coord_y = self.indice.repeat((1, self.feat_sz)).view((self.feat_sz * self.feat_sz,)).float()  # .cuda()
    
    # revision for icraft: 新增用于进行torch.matmul操作的常量tensor
    # 可自行决定是否在cuda device下保存模型，确保此处三者与导入模型&输入参数一致即可
    self.exp_e = torch.tensor([[1.0]]*feat_sz*feat_sz).float()  # .cuda()

# 3.1.2 将soft_argmax函数中的torch.sum改为1中新增的常量tensor进行torch.matmul操作
def CP_soft_argmax(self, score_map):
    """
    Get soft-argmax coordinate for a given heatmap.
    """
    prob_vec = nn.functional.softmax(
        score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
    exp_x = self.coord_x * prob_vec
    exp_y = self.coord_y * prob_vec
    exp_x = torch.matmul(exp_x, self.exp_e)
    exp_y = torch.matmul(exp_y, self.exp_e)
    return exp_x, exp_y

# 3.1.3 将forward函数中的torch.stack改为torch.cat和reshape
def CP_forward(self, x):
    """
    Forward pass with input x.
    """
    score_map_tl, score_map_br = self.get_score_map(x)
    coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
    coorx_br, coory_br = self.soft_argmax(score_map_br)
    return torch.cat((coorx_tl, coory_tl, coorx_br, coory_br), dim=0).reshape(1, -1) / self.img_sz

Corner_Predictor.__init__ = CP_init
Corner_Predictor.soft_argmax = CP_soft_argmax
Corner_Predictor.forward = CP_forward

#----------------------------
# 3.2 替换AIATRACK类的forward_box_head函数：
#     将box_xyxy_to_xywh, box_xyxy_to_cxcywh函数计算移至网络外作为后处理计算，输出仅一个tensor变量
#----------------------------
def aiatrack_forward_box_head(self, hs):
    """
    Args:
        hs: Output embeddings (1, HW, B, C).
    """
    # Adjust shape
    opt = hs.permute(2, 0, 3, 1).contiguous()
    bs, Nq, C, HW = opt.size()
    opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
    # Run the corner head
    bbox_coor = self.box_head(opt_feat)
    return bbox_coor

AIATRACK_M.forward_box_head = aiatrack_forward_box_head

#----------------------------
# 4. AIATRACK模型
#----------------------------
#----------------------------
# 4.1 替换AIATRACK类的forward函数：
#     串联模型流程
#----------------------------
def aiatrack_forward(self, imgft, pos_emb, inr_emb,
                     refer_mem0, refer_mem1,
                     refer_emb0, refer_emb1,
                     refer_pos0, refer_pos1):
    
    # Forward the backbone
    feat = self.adjust(self.backbone(imgft))

    # Forward the transformer encoder and decoder
    search_mem = self.transformer.encoder(feat, pos_emb, inr_emb)
    output_embed = self.transformer.decoder(search_mem,
                                            refer_mem0, refer_mem1,
                                            refer_emb0, refer_emb1,
                                            refer_pos0, refer_pos1)
    
    # Forward the corner head and get iou feature
    bbox_coor, iou_feat = self.forward_heads(output_embed)

    return search_mem, bbox_coor, iou_feat

#----------------------------
# 4.2 替换AIATRACK类的forward_heads函数：
#     box_xyxy_to_cxcywh, box_xyxy_to_xywh移至网络外计算；
#     iou_head中仅保留可导出的get_iou_feat部分
#----------------------------
def aiatrack_forward_heads(self, hs):
    """
    Args:
        hs: Output embeddings (1, HW, B, C).
    """

    opt = hs.permute(2, 0, 3, 1).contiguous()
    bs, Nq, C, HW = opt.size()
    opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

    # forward box_head
    bbox_coor = self.box_head(opt_feat)

    # get iou feature
    iou_feat = self.iou_head.get_iou_feat(opt_feat)

    return bbox_coor, iou_feat

AIATRACK_M.forward = aiatrack_forward
AIATRACK_M.forward_heads = aiatrack_forward_heads

#----------------------------
# 5. AIATRACK推理
#----------------------------
#----------------------------
# 5.1 替换PositionEmbeddingSine类的forward函数：
#     去除输入参数中的tensor_list.tensors，仅输入tensor_list.mask
#----------------------------
def posemb_forward(self, mask: Tensor):
    assert mask is not None
    not_mask = ~mask  # (b,h,w)
    # 1 1 1 1... 2 2 2 2... 3 3 3 3...
    y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Cumulative sum along axis 1 (h axis) --> (b,h,w)
    # 1 2 3 4... 1 2 3 4... 1 2 3 4...
    x_embed = not_mask.cumsum(2, dtype=torch.float32)  # Cumulative sum along axis 2 (w axis) --> (b,h,w)
    if self.normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2pi * (y / sigma(y))
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2pi * (x / sigma(x))

    # num_pos_feats = d/2
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)  # (0,1,2,...,d/2)
    dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
    pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
    return pos  # (b,d,h,w)

PositionEmbeddingSine.forward = posemb_forward

#----------------------------
# 5.2 替换AIATRACK类的__init__, initialize和track函数
#----------------------------
# 5.2.1 在__init__函数中新增self.position_embedding, self.inner_embedding
def aiatrack_init(self, params, dataset_name):
    super(AIATRACK, self).__init__(params)
    # network = build_aiatrack(params.cfg)
    # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
    network = torch.jit.load(model_pt)# 加载trace出来的模型
    self.cfg = params.cfg
    self.net = network#cuda.()#在Gpu环境需要取消此处注释
    self.net.eval()
    self.preprocessor = Preprocessor()
    self.state = None
    self.position_embedding, self.inner_embedding = build_position_encoding(params.cfg)
    # 可通过读取1_save.py中保存的embed_bank文件获得embed_bank
    # self.embed_bank = torch.tensor(np.fromfile(EMBED_BANK, dtype=np.float32).reshape(1, 2, 256)).cuda()
    # For debug
    self.debug = False
    self.frame_id = 0
    # Set the hyper-parameters
    DATASET_NAME = dataset_name.upper()
    if hasattr(self.cfg.TEST.HYPER, DATASET_NAME):
        self.cache_siz = self.cfg.TEST.HYPER[DATASET_NAME][0]
        self.refer_cap = 1 + self.cfg.TEST.HYPER[DATASET_NAME][1]
        self.threshold = self.cfg.TEST.HYPER[DATASET_NAME][2]
    else:
        self.cache_siz = self.cfg.TEST.HYPER.DEFAULT[0]
        self.refer_cap = 1 + self.cfg.TEST.HYPER.DEFAULT[1]
        self.threshold = self.cfg.TEST.HYPER.DEFAULT[2]
    if self.debug:
        self.save_dir = 'debug'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    # For save boxes from all queries
    self.save_all_boxes = params.save_all_boxes

#----------------------------
# 5.2.2 在initialize函数中新增原在AIATRACK.backbone中进行的mask, pos, inr相关计算；
#       在initialize函数中新增原在*AiAModule.CorrAttention*中进行的`torch.repeat_interleave`操作；
#       注释后续未使用的refer_msk_cache, refer_msk_list
#----------------------------
def aiatrack_initialize(self, image, info: dict, seq_name: str = None):
    # Forward the long-term reference once
    refer_crop, resize_factor, refer_att_mask = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                              output_sz=self.params.search_size)
    refer_box = transform_image_to_crop(torch.Tensor(info['init_bbox']), torch.Tensor(info['init_bbox']),
                                        resize_factor,
                                        torch.Tensor([self.params.search_size, self.params.search_size]),
                                        normalize=True)
    self.feat_size = self.params.search_size // 16
    refer_img = self.preprocessor.process(refer_crop, refer_att_mask)
    with torch.no_grad():
        mask = F.interpolate(refer_img.mask[None].float(), size=[20, 20]).to(torch.bool)[0]
        pos = self.position_embedding(mask).flatten(2).permute(2, 0, 1)  # HWxBxC
        inr = self.inner_embedding(mask).flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_emb = torch.repeat_interleave(inr, 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)
        # 初始值为0
        ref_mem0 = torch.zeros(400,1,256)
        ref_mem1 = torch.zeros(1600,1,256)
        ref_emb0 = torch.zeros(400,1,256)
        ref_emb1 = torch.zeros(1600,1,256)
        ref_pos0 = torch.zeros(400, 4, 64)
        ref_pos1 = torch.zeros(400, 16, 64)
        # feat = self.net.adjust(self.net.backbone(refer_img.tensors))
        # refer_mem = self.net.transformer.encoder(feat, pos, inr_emb)
        refer_mem, bbox, iou_feat = self.net(refer_img.tensors, pos, inr_emb,ref_mem0,ref_mem1,ref_emb0,ref_emb1,ref_pos0,ref_pos1)

    target_region = torch.zeros((self.feat_size, self.feat_size))
    x, y, w, h = (refer_box * self.feat_size).round().int()
    target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
    target_region = target_region.view(self.feat_size * self.feat_size, -1)
    background_region = 1 - target_region
    refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0)#.cuda()#在Gpu环境需要取消此处注释
    embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                           dim=0).unsqueeze(0)
    # embed_bank可通过读取1_save.py中保存的embed_bank文件获得
    # embed_bank = self.embed_bank
    self.refer_mem_cache = [refer_mem]
    self.refer_emb_cache = [torch.bmm(refer_region, embed_bank).transpose(0, 1)]
    self.refer_pos_cache = [inr]
    self.refer_mem_list = list()
    for _ in range(self.refer_cap):
        self.refer_mem_list.append(self.refer_mem_cache[0])
    self.refer_emb_list = list()
    for _ in range(self.refer_cap):
        self.refer_emb_list.append(self.refer_emb_cache[0])
    self.refer_pos_list = list()
    for _ in range(self.refer_cap):
        self.refer_pos_list.append(self.refer_pos_cache[0])
    # Save states
    self.state = info['init_bbox']
    if self.save_all_boxes:
        # Save all predicted boxes
        all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
        return {'all_boxes': all_boxes_save}

#----------------------------
# 5.2.3 在track函数中新增原在AIATRACK.backbone中进行的mask, pos, pos_emb相关计算
#       在track函数中新增原在AiAModule.CorrAttention中进行的torch.repeat_interleave操作；
#       在track函数中新增原在TransformerDecoderLayer.forward中进行的torch.cat操作；
#       在track函数中新增原在AIATRACK.forward_box_head中进行的box_xyxy_to_xywh, box_xyxy_to_cxcywh计算；
#       在track函数中去除原由AIATRACK.forward_iou_head获得pred_iou及其相关内容，改为每帧均进行“Update state”；
#       注释后续未使用的refer_msk_cache, refer_msk_list
#----------------------------
def aiatrack_track(self, image, info: dict = None, seq_name: str = None):
    H, W, _ = image.shape
    self.frame_id += 1
    # Get the t-th search region
    search_crop, resize_factor, search_att_mask = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
    search_img = self.preprocessor.process(search_crop, search_att_mask)
    with torch.no_grad():
        mask = F.interpolate(search_img.mask[None].float(), size=[20, 20]).to(torch.bool)[0]
        pos = self.position_embedding(mask).flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_emb = self.inner_embedding(mask).flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_emb = torch.repeat_interleave(pos_emb, 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)
        refer_pos0 = torch.repeat_interleave(self.refer_pos_list[0], 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)
        refer_pos1 = torch.repeat_interleave(torch.cat(self.refer_pos_list[1:], dim=0), 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)
        search_mem, bbox_coor, iou_feat = self.net(search_img.tensors, pos, inr_emb,
                                                   self.refer_mem_list[0], torch.cat(self.refer_mem_list[1:], dim=0),
                                                   self.refer_emb_list[0], torch.cat(self.refer_emb_list[1:], dim=0),
                                                   refer_pos0, refer_pos1)

    # Get the final result
    outputs_coord = box_xyxy_to_xywh(bbox_coor)
    out_dict = box_xyxy_to_cxcywh(bbox_coor)
    out_dict = out_dict.view(1, -1, 4)
    pred_boxes = out_dict.view(-1, 4)
    # Baseline: Take the mean of all predicted boxes as the final result
    pred_box = (pred_boxes.mean(
        dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
    # Get the final box result
    self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

    # Update state
    if True:
        if len(self.refer_mem_cache) == self.cache_siz:
            _ = self.refer_mem_cache.pop(1)
            _ = self.refer_emb_cache.pop(1)
            _ = self.refer_pos_cache.pop(1)
        target_region = torch.zeros((self.feat_size, self.feat_size))
        x, y, w, h = (outputs_coord[0] * self.feat_size).round().int()
        target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
        target_region = target_region.view(self.feat_size * self.feat_size, -1)
        background_region = 1 - target_region
        refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0)#.cuda()#在Gpu环境需要取消此处注释
        embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                               dim=0).unsqueeze(0)
        new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)
        # embed_bank可通过读取1_save.py中保存的embed_bank文件获得
        # new_emb = torch.bmm(refer_region, self.embed_bank).transpose(0, 1)
        self.refer_mem_cache.append(search_mem)
        self.refer_emb_cache.append(new_emb)
        self.refer_pos_cache.append(pos_emb)
        self.refer_mem_list = [self.refer_mem_cache[0]]
        self.refer_emb_list = [self.refer_emb_cache[0]]
        self.refer_pos_list = [self.refer_pos_cache[0]]
        max_idx = len(self.refer_mem_cache) - 1
        ensemble = self.refer_cap - 1
        for part in range(ensemble):
            self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
            self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
            self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
    # For debug
    if self.debug:
        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=3)
        save_seq_dir = os.path.join(self.save_dir, seq_name)
        if not os.path.exists(save_seq_dir):
            os.makedirs(save_seq_dir)
        save_path = os.path.join(save_seq_dir, '%04d.jpg' % self.frame_id)
        cv2.imwrite(save_path, image_BGR)
    if self.save_all_boxes:
        # Save all predictions
        all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        all_boxes_save = all_boxes.view(-1).tolist()  # (4N,)
        return {'target_bbox': self.state,
                'all_boxes': all_boxes_save}
    else:
        return {'target_bbox': self.state}

AIATRACK_T.__init__ = aiatrack_init
AIATRACK_T.initialize = aiatrack_initialize
AIATRACK_T.track = aiatrack_track


#----------------------------
# 推理参数配置
#----------------------------
# 网络模型参数文件名
PARAM = "baseline"
# 测试数据集
dataset_name = "lasot"
seq = None  # 测试单一sequence时可配置

# 通过1_save.py保存的模型embedding值文件，亦可无
# EMBED_BANK = "../2_compile/fmodel/embed_bank_1_2_256.ftmp"
# 加载的trace出来的模型
model_pt = "../2_compile/fmodel/AiATrack_ensemble4_320x320_traced.pt"


if __name__ == "__main__":
    run_tracker("aiatrack", PARAM, dataset_name=dataset_name, sequence=seq)

