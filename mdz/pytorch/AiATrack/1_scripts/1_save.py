import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), './')
if prj_path not in sys.path:
    sys.path.append(prj_path)


import torch
import numpy as np
from lib.test.parameter.aiatrack import parameters
from lib.models.aiatrack.aiatrack import *
from lib.models.aiatrack.backbone import *
from lib.models.aiatrack.transformer import *
from lib.models.aiatrack.transformer import _get_activation_fn
from lib.models.aiatrack.head import *


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

AIATRACK.adjust = aiatrack_adjust

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

AIATRACK.forward_box_head = aiatrack_forward_box_head

#----------------------------
# 4. AIATRACK模型保存
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


def aiatrack_forward_box(self, imgft, pos_emb, inr_emb,
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
    
    # Forward the corner head
    bbox_coor = self.forward_box_head(output_embed)

    return search_mem, bbox_coor

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


#----------------------------
# 模型保存配置
#----------------------------
# 是否需要导出模型iou_head中的iou feature
NEED_IOU_FEAT = True
# 输入transformer.decoder中的中间帧数量（对应config.py和{PARAM}.yaml文件中TEST.HYPER相应配置参数list中第2个数值）
ENSEMBLE = 4

if NEED_IOU_FEAT:
    # 需要导出iou_head中的iou feature
    AIATRACK.forward = aiatrack_forward
    AIATRACK.forward_heads = aiatrack_forward_heads
else:
    # 不需要导出iou_head中的iou feature
    AIATRACK.forward = aiatrack_forward_box

#----------------------------
# 路径配置
#----------------------------
# 网络模型参数文件名
PARAM = "baseline"
# trace后模型地址
TRACE_PATH = f"../2_compile/fmodel/AiATrack_ensemble{ENSEMBLE}_320x320_traced.pt"


if __name__ == "__main__":
    # 加载模型
    params = parameters(PARAM)
    model = build_aiatrack(params.cfg)
    weights = torch.load(params.checkpoint, map_location='cpu')
    model.load_state_dict(weights['net'], strict=True)
    model = model.eval()
    # 输入参数
    imgts = torch.randn(1, 3, 320, 320)
    pos = torch.randn(400, 1, 256)
    inr = torch.randn(400, 4, 64)
    refer_mem0 = torch.randn(400, 1, 256)
    refer_mem1 = torch.randn(400*ENSEMBLE, 1, 256)
    refer_emb0 = torch.randn(400, 1, 256)
    refer_emb1 = torch.randn(400*ENSEMBLE, 1, 256)
    refer_pos0 = torch.randn(400, 4, 64)
    refer_pos1 = torch.randn(400, 4*ENSEMBLE, 64)
    # trace模型
    outtrace = torch.jit.trace(model, (imgts, pos, inr,
                                       refer_mem0, refer_mem1,
                                       refer_emb0, refer_emb1,
                                       refer_pos0, refer_pos1))
    # print(outtrace.code)
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    outtrace.save(TRACE_PATH)
    # 保存模型embedding值，用于后处理计算
    embed_bank = torch.cat([model.foreground_embed.weight, model.background_embed.weight], dim=0).unsqueeze(0)
    embed_bank.cpu().detach().contiguous().numpy().astype(np.float32).tofile(os.path.join(os.path.dirname(TRACE_PATH), 'embed_bank_1_2_256.ftmp'))

