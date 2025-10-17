import torch
import os
import sys
import time
from collections import OrderedDict
import types

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation import Tracker
from lib.test.tracker.mixformer2_vit_online import MixFormerOnline
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.test.evaluation.running import _save_tracker_output
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask




def Attention_forward(self, x, t_h, t_w, s_h, s_w):
    """
    x is a concatenated vector of template and search region features.
    """
    # ICRAFT NOTE:
    # 为了消除floor_divide算子，替换为固定数值
    # 为了消除unbind算子，手动将qkv的输出拆分为q, k, v
    B, N, C = x.shape # (1, 456, 768)
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, 64).permute(2, 0, 3, 1, 4)
    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
    k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
    v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w + 4], dim=2)

    # asymmetric mixed attention
    attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

    attn = (q_s @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w + 4, C)

    x = torch.cat([x_mt, x_s], dim=1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def VisionTransformer_forward(self, x_t, x_ot, x_s, pos_embed_s, pos_embed_t, reg_tokens):
    """
    :param x_t: (batch, c, 128, 128)
    :param x_s: (batch, c, 288, 288)
    :return:
    """
    x_t = self.patch_embed(x_t)  # BCHW-->BNC
    x_ot = self.patch_embed(x_ot)
    x_s = self.patch_embed(x_s)
    # B, C = x_t.size(0), x_t.size(-1)
    B, C = 1, 768
    H_s = W_s = self.feat_sz_s
    H_t = W_t = self.feat_sz_t
    # ICRAFT NOTE:
    # 为了减少计算量，将初始化好的常量pos_embedding的作为前向的输入参数
    # x_s = x_s + self.pos_embed_s
    # x_t = x_t + self.pos_embed_t
    # x_ot = x_ot + self.pos_embed_t
    x_s = x_s + pos_embed_s
    x_t = x_t + pos_embed_t
    x_ot = x_ot + pos_embed_t

    # ICRAFT NOTE:
    # 为了消除expand算子，减少计算量，将常量reg_tokens的计算移到外部
    # reg_tokens = self.reg_tokens.expand(B, -1, -1)  # (b, 4, embed_dim)
    # reg_tokens = reg_tokens + self.pos_embed_reg
    # reg_tokens = self.reg_tokens + self.pos_embed_reg
    x = torch.cat([x_t, x_ot, x_s, reg_tokens], dim=1)  # (b, hw+hw+HW+4, embed_dim)
    x = self.pos_drop(x)

    # ICRAFT NOTE:
    # 为了消除无关的计算，将distill_feat_list的计算移除
    # distill_feat_list = []
    for i, blk in enumerate(self.blocks):
        x = blk(x, H_t, W_t, H_s, W_s)
        # distill_feat_list.append(x)
    # [1,298,768] split [49, 49, 196, 4]
    # x_t, x_ot, x_s, reg_tokens = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s, 4], dim=1)
    x_t, x_ot, x_s, reg_tokens = torch.split(x, [49, 49, 196, 4], dim=1)

    x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
    x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
    x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
    # ICRAFT NOTE:
    # 为了配合计算图，将distill_feat_list的返回值移除
    return x_t_2d, x_ot_2d, x_s_2d, reg_tokens #, distill_feat_list

def MixFormer_forward(self, template, online_template, search, pos_embed_s, pos_embed_t, reg_tokens): #, gt_bboxes=None):
    # search: (b, c, h, w)
    # ICRAFT NOTE:
    # 消除推理时的无效检查和squeeze操作
    # if template.dim() == 5:
    #     template = template.squeeze(0)
    # if online_template.dim() == 5:
    #     online_template = online_template.squeeze(0)
    # if search.dim() == 5:
    #     search = search.squeeze(0)

    # ICRAFT NOTE:
    # 因为VisionTransformer的forward方法改动为6输入，4输出，做如下改动
    # template, online_template, search, reg_tokens, distill_feat_list = self.backbone(template, online_template, search)
    template, online_template, search, reg_tokens = self.backbone(template, online_template, search, pos_embed_s, pos_embed_t, reg_tokens) # distill_feat_list removed for icraft parser
    # ICRAFT NOTE:
    # box_head模块已经替换forward，torchscript不支持dict传递feat,做如下改动
    # 固定控制流程，将部分前向计算移到后处理，直接反馈四个坐标tensor和pred_scores_feat
    # out = self.forward_head(search, reg_tokens=reg_tokens, run_score_head=run_score_head, softmax=softmax)
    # out['reg_tokens'] = reg_tokens
    # out['distill_feat_list'] = distill_feat_list
    # b = reg_tokens.size(0) # 1,4,768
    # pred_boxes, prob_l, prob_t, prob_r, prob_b = self.box_head(reg_tokens) # , softmax=softmax)
    coord_l, coord_t, coord_r, coord_b = self.box_head(reg_tokens) # , softmax=softmax)
    # if(pred_boxes.size(0) > 1):
    #     print('multiple tracking boxes')
    # ICRAFT NOTE: Move box transform to postprocess
    # outputs_coord = box_xyxy_to_cxcywh2(pred_boxes)
    # outputs_coord_new = outputs_coord.view(b, 1, 4)
    pred_scores_feat = self.score_head(reg_tokens)
    # if(pred_scores.size(0) > 1):
    #     print('multiple score')
    return coord_l, coord_t, coord_r, coord_b, pred_scores_feat
    # return outputs_coord_new, prob_l, prob_t, prob_b, prob_r, pred_scores, reg_tokens
    # return out

    # add bw_computation
    # record('mixformer2_vit_online.xlsx', 'mixformer2_vit_online', [template, online_template, search])
    return out

# 全局变量
feat_sz = 96
stride = 224 / feat_sz
indice_T = torch.arange(0, feat_sz).unsqueeze(0).transpose(0,-1).contiguous() * stride
def MlpHead_forward(self, reg_tokens):
    """
    reg_tokens shape: (b, 4, embed_dim)
    """
    """
    NOTE: 针对Icraft模型导出的修改如下：
    常量在初始化时候提前转置好
    去掉修改不支持的unbind，直接slice
    用matmul代替*和sum
    用cat代替stack，注释掉非softmax分支
    修改不支持的mean
    调整输入输出，部分小计算移到后处理
    """
    #ICRAFT NOTE:
    # 去掉修改不支持的unbind，直接slice
    # reg_token_l, reg_token_r, reg_token_t, reg_token_b = reg_tokens.unbind(dim=1)   # (b, c)
    reg_token_l = reg_tokens[:,0,:]
    reg_token_r = reg_tokens[:,1,:]
    reg_token_t = reg_tokens[:,2,:]
    reg_token_b = reg_tokens[:,3,:]
    ####
    score_l = self.layers(reg_token_l)
    score_r = self.layers(reg_token_r)
    score_t = self.layers(reg_token_t)
    score_b = self.layers(reg_token_b) # (b, feat_sz)

    prob_l = score_l.softmax(dim=-1)
    prob_r = score_r.softmax(dim=-1)
    prob_t = score_t.softmax(dim=-1)
    prob_b = score_b.softmax(dim=-1)

    # ICRAFT NOTE:
    # 用matmul代替*和sum
    # coord_l_0 = torch.sum((self.indice * prob_l), dim=-1)
    # coord_r_0 = torch.sum((self.indice * prob_r), dim=-1)
    # coord_t_0 = torch.sum((self.indice * prob_t), dim=-1)
    # coord_b_0 = torch.sum((self.indice * prob_b), dim=-1) # (b, ) absolute coordinates
    coord_l = (prob_l @ indice_T)
    coord_r = (prob_r @ indice_T)
    coord_t = (prob_t @ indice_T)
    coord_b = (prob_b @ indice_T)

    # ICRAFT NOTE:
    # 用cat代替stack，注释掉非softmax分支，cat在后处理做
    # return torch.cat((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
    #     prob_l, prob_t, prob_r, prob_b
    return coord_l, coord_t, coord_r, coord_b
    # return xyxy, ltrb
    # if softmax:
    # return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
    #     prob_l, prob_t, prob_r, prob_b
    # else:
    #     return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
    #         score_l, score_t, score_r, score_b

def MlpScoreDecoder_forward(self, reg_tokens):
    """
    reg tokens shape: (b, 4, embed_dim)
    """
    x = self.layers(reg_tokens) # (b, 4, 1)
    # ICRAFT NOTE:
    # 修改不支持的mean，迁移到后处理
    # x = x.mean(dim=1)   # (b, 1)
    # x = (x[:,0,:] + x[:,1,:] + x[:,2,:] + x[:,3,:])/4
    return x

def MixFormerOnline_track(self, image, info: dict = None):
    H, W, _ = image.shape
    self.frame_id += 1
    print(self.frame_id)
    x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                            output_sz=self.params.search_size)  # (x1, y1, w, h)
    search = self.preprocessor.process(x_patch_arr)
    with torch.no_grad():
        # TODO: use forward_test() in test
        # ICRAFT NOTE:
        # 原有常量在计算图中会导致Icraft编译报错，所以提取出来作为输入
        pos_embed_t = self.network.backbone.pos_embed_t
        pos_embed_s = self.network.backbone.pos_embed_s
        reg_tokens = self.network.backbone.reg_tokens + self.network.backbone.pos_embed_reg
        
        # ICRAFT NOTE：
        coord_l, coord_t, coord_r, coord_b, pred_scores_feat = self.network(self.template, self.online_template, search,
                                                    pos_embed_s, pos_embed_t, reg_tokens)#, softmax=True, run_score_head=True)
    # ICRAFT NOTE：
    # 从box_head和score_head提取出来的计算并入后处理
    pred_boxes = torch.cat((coord_l, coord_t, coord_r, coord_b), dim=1) / 224.0
    pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
    pred_score = pred_scores_feat.mean(dim=1).view(1).sigmoid().item()
    # print(pred_boxes)
    # Baseline: Take the mean of all pred boxes as the final result
    pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
    # get the final box result
    self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
    self.max_pred_score = self.max_pred_score * self.max_score_decay
    # update template
    if pred_score > 0.5 and pred_score > self.max_pred_score:
        print(f'On frame {self.frame_id}, pred_score={pred_score}, max_pred_score={self.max_pred_score}, state={self.state}, updated')
        z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                    self.params.template_factor,
                                                    output_sz=self.params.template_size)  # (x1, y1, w, h)
        # cv2.imwrite(f'./debug/{self.frame_id}_online_max_template_{pred_score:.4f}.jpg', z_patch_arr)
        self.online_max_template = self.preprocessor.process(z_patch_arr)
        self.max_pred_score = pred_score
    if self.frame_id % self.update_interval == 0:
        print(f"frame{self.frame_id}: update, pred_score={pred_score}, max_pred_score={self.max_pred_score}, state={self.state}")
        if self.online_size == 1:
            self.online_template = self.online_max_template
        
        self.max_pred_score = -1
        self.online_max_template = self.template

    if self.save_all_boxes:
        '''save all predictions'''
        all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
        return {"target_bbox": self.state,
                "all_boxes": all_boxes_save,
                "conf_score": pred_score}
    else:
        return {"target_bbox": self.state, "conf_score": pred_score}


def MixFormerOnline___init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        # network = build_mixformer2_vit_online(params.cfg,  train=False)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        network = torch.jit.load(model_pt)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        self.cfg = params.cfg
        self.network = network#.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_size'):
            self.online_size = params.online_size
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("Search factor: ", self.params.search_factor)
        print("Update interval is: ", self.update_interval)
        print("Online size is: ", self.online_size)
        print("Max score decay: ", self.max_score_decay)


# 全局变量
model_pt = "../2_compile/fmodel/mixformer2_vit_online_small_1x3x224x224_traced.pt"

if __name__ == '__main__':
    dataset_name = 'lasot'
    tracker_name = 'mixformer2_vit_online'
    tracker_param = '224_depth4_mlp1_score'
    run_id = None
    dataset = get_dataset(dataset_name)
    tracker_params = {'model': '../weights/mixformerv2_small.pth.tar', 'search_area_scale': 4.5, 'max_score_decay': 1.0, 'vis_attn': 0}
    Tester = Tracker(tracker_name, tracker_param, dataset_name, run_id, tracker_params=tracker_params)
    debug_ = getattr(Tester.params, 'debug', 0)
    Tester.params.debug = debug_

    # ICRAFT NOTE: 修改顶层Tracker类MixFormerOnline初始化函数，以加载trace出来的torchscrtpt模型
    MixFormerOnline.__init__ = MixFormerOnline___init__

    # 模型修改点
    # ICRAFT NOTE: 修改顶层Tracker类MixFormerOnline前向
    from lib.test.tracker.mixformer2_vit_online import MixFormerOnline
    MixFormerOnline.track =  MixFormerOnline_track
    # ICRAFT NOTE: 修改MixFormer顶层前向
    from lib.models.mixformer2_vit.mixformer2_vit_online import MixFormer
    MixFormer.forward = MixFormer_forward
    # ICRAFT NOTE: 修改VisionTransformer前向
    from lib.models.mixformer2_vit.mixformer2_vit_online import VisionTransformer
    VisionTransformer.forward = VisionTransformer_forward
    # ICRAFT NOTE: 修改Attention模块
    from lib.models.mixformer2_vit.mixformer2_vit_online import Attention
    Attention.forward = Attention_forward
    # ICRAFT NOTE: 修改BoxHead前向
    from lib.models.mixformer2_vit.head import MlpHead
    MlpHead.forward = MlpHead_forward
    # ICRAFT NOTE: 修改ScoreHead前向
    from lib.models.mixformer2_vit.head import MlpScoreDecoder
    MlpScoreDecoder.forward = MlpScoreDecoder_forward

    tracker = MixFormerOnline(Tester.params, dataset_name=dataset_name)
    # ICRAFT NOTE:
    tracker.online_size = 1 # Fix bug


    for seq in dataset:
        # print(seq)
        params = Tester.params
        init_info = seq.init_info()
        output = {'target_bbox': [],
                    'time': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = Tester._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = Tester._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            # print(out)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)
        
        sys.stdout.flush()

        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        Tester.results_dir = Tester.results_dir + "_save_infer/"
        if not os.path.exists(Tester.results_dir):
                os.makedirs(Tester.results_dir)
        if not params.debug:
            _save_tracker_output(seq, Tester, output)