import numpy  as np
import time
import math
import icraft.xir as ir
import icraft.xrt as rt 

# 网络的数据精度 支持 8 / 16
BIT = 8
# 配置不同精度下的参数
if BIT ==16:
    MAXC = 32
    MINC = 4
elif BIT == 8:
    MAXC = 64
    MINC = 8
# Anchor settings 
STRIDE = [8, 16, 32]
ANCHORS = np.array([[[10,13], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]])
NOC = 80
# ANCHOR_LENGTH 代表icorepost 输出的数据中 每一条的长度 icorepost 输出的数据量等于 目标数*ANCHOR_LENGTH
ANCHOR_LENGTH = math.ceil((5+NOC) / float(MINC)) * MINC + MINC

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def jaccardDistance(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - intersection_area
    
    jaccard_distance = 1.0 - intersection_area / union_area
    
    return jaccard_distance

def soft_nms(box_list, score_list, id_list, conf=0.25, iou=0.45, NOC=80):
    nms_indices = []

    for class_id in range(NOC):
        score_index_vec = []
        for i in range(len(score_list)):
            if score_list[i] > conf and id_list[i] == class_id:
                score_index_vec.append((score_list[i], i))
        
        score_index_vec.sort(key=lambda x: x[0], reverse=True)
        
        # for i in range(len(score_index_vec)):
        #     idx = score_index_vec[i][1]
        #     keep = True
        #     for k in range(len(nms_indices)):
        #         if 1 - jaccardDistance(box_list[idx], box_list[nms_indices[k]]) > iou:
        #             keep = False
        #             break
        #     if keep:
        #         nms_indices.append(idx)
        # 修改为类内NMS
        class_nms_indices = []
        for i in range(len(score_index_vec)):
            idx = score_index_vec[i][1]
            keep = True
            for k in range(len(class_nms_indices)):
                if 1 - jaccardDistance(box_list[idx], box_list[class_nms_indices[k]]) > iou:
                    keep = False
                    break
            if keep:
                class_nms_indices.append(idx)

        nms_indices.extend(class_nms_indices)
    # get box,score,id based on nms_idx 
    box_list = [box_list[obj] for obj in nms_indices]
    nms_box_list = []
    for box in box_list:
        x0 = box[0]
        y0 = box[1]
        x1 = x0+box[2]
        y1 = y0+box[3]
        nms_box_list.append([x0,y0,x1,y1])
    nms_score_list = [score_list[obj] for obj in nms_indices]
    nms_cls_ids = [id_list[obj] for obj in nms_indices]
        
    return nms_indices,nms_box_list,nms_score_list,nms_cls_ids
    return nms_indices,nms_box_list,nms_score_list,nms_cls_ids
def fpga_nms(box_list,score_list,id_list,conf,iou_thresh,NOC,device,nms_reg_base_default = 0x100001C00):
    nms_res = []
    nms_pre_data = []
    # res.sort(key = lambda obj: obj.score,reverse = True)
    score_index_vec = []
    for i in range(len(score_list)):
        if score_list[i] > conf:
            score_index_vec.append((score_list[i],i))
    score_index_vec.sort(key= lambda x:x[0],reverse=True)
    print("score_index_vec=",score_index_vec)
    nms_pre_data = []
    for score,idx in score_index_vec:
        print('idx =',idx)
        x1 = max(box_list[idx][0],0)
        y1 = max(box_list[idx][1],0)
        x2 = max(box_list[idx][2],0)
        y2 = max(box_list[idx][3],0)
        nms_pre_data.append(np.array([x1,y1,x2,y2,idx],dtype=np.uint16))
    print("nms_pre_data=",nms_pre_data)

    box_num = len(nms_pre_data)
    if box_num==0:
        return []
    nms_data = np.array(nms_pre_data, dtype=np.uint16)
    # data to udma buffer
    nms_data_tensor = rt.Tensor(nms_data)
    uregion_ = device.getMemRegion("udma")
    utensor = nms_data_tensor.to(uregion_)
    arbase = utensor.data().addr()
    awbase = arbase
    
    reg_base = nms_reg_base_default
    if device.defaultRegRegion().read(reg_base + 0x008, True) != 0x23110200:
        raise RuntimeError("ERROR :: No NMS Hardware")
    print('arbase =',arbase)
    group_num = int(np.ceil(box_num / 16))
    if group_num == 0:
        raise RuntimeError("ERROR :: group_num == 0")
    print('group_num =',group_num)

    last_araddr = arbase + group_num * 160 - 8
    print('last_araddr =',last_araddr)
    if last_araddr < arbase:
        raise RuntimeError("ERROR :: last_araddr < arbase")
    anchor_hpsize = int(np.ceil(box_num / 64))
    print('anchor_hpsize =',anchor_hpsize)
    if anchor_hpsize == 0:
        raise RuntimeError("ERROR :: anchor_hpsize == 0")

    last_awaddr = awbase + anchor_hpsize * 8 - 8
    print('last_awaddr =',last_awaddr)
    if last_awaddr < awbase:
        raise RuntimeError("ERROR :: last_awaddr < awbase")
    threshold = np.uint16(iou_thresh * 2**15)
    print('threshold =',threshold)
    # set reg
    device.defaultRegRegion().write(reg_base + 0x014, 1, True)
    device.defaultRegRegion().write(reg_base + 0x014, 0, True)
    device.defaultRegRegion().write(reg_base + 0x01C, arbase, True)
    device.defaultRegRegion().write(reg_base + 0x020, awbase, True)
    device.defaultRegRegion().write(reg_base + 0x024, last_araddr, True)
    device.defaultRegRegion().write(reg_base + 0x028, last_awaddr, True)
    device.defaultRegRegion().write(reg_base + 0x02C, group_num, True)
    device.defaultRegRegion().write(reg_base + 0x030, 1, True)
    device.defaultRegRegion().write(reg_base + 0x034, threshold, True)
    device.defaultRegRegion().write(reg_base + 0x038, anchor_hpsize, True)
    # start
    device.defaultRegRegion().write(reg_base + 0x0, 1, True)
    start = time.time()
    while True:
        reg_done = device.defaultRegRegion().read(reg_base + 0x004, True)
        duration = time.time() - start
        if duration > 1:
            raise RuntimeError("NMS Timeout!!!")
        if reg_done != 0:
            break
    
    mask = np.array(utensor.to(rt.HostDevice.MemRegion())).flatten().reshape(-1, 1)

    for i in range(box_num):
        mask_idx = i // 16
        idx = i % 16
        s = np.binary_repr(mask[mask_idx][0], width=16)[15 - idx]
        if s == "1":
            nms_res.append(box_list[i])
    print('nms_res =',nms_res)
    return nms_res

def hard_nms(box_list,score_list,id_list,conf,iou,NOC,device):
    nms_indices = []
    score_index_vec = []
    for i in range(len(score_list)):
        if score_list[i] > conf:
            score_index_vec.append((score_list[i],i))
    score_index_vec.sort(key= lambda x:x[0],reverse=True)

    nms_pre_data = []
    for score,idx in score_index_vec:
        print('idx =',idx)
        x1 = max(box_list[idx][0],0)
        y1 = max(box_list[idx][1],0)
        x2 =box_list[idx][2]
        y2 = box_list[idx][3]
        nms_pre_data.append([np.int16(x1),np.int16(y1),np.int16(x2),np.int16(y2),id_list[idx]])
    
    # nms_data_cptr = np.array(nms_pre_data, dtype=np.int16).data()
    box_num = len(score_index_vec)
    nms_data= np.array(nms_pre_data, dtype=np.int16)
    print(type(nms_data))
    print('#######################test udma#############################')
    uregion_ = device.getMemRegion("udma") #申请udma
    #取udma buffer的地址
    # udma_chunk_ = uregion_.malloc(10000000)
    nms_data_tensor = rt.Tensor(nms_data)
    print('type =',type(nms_data_tensor))
    utensor = nms_data_tensor.to(uregion_) # to 就完成了 将num_data_tensor吐进udma_buffer空间中 不需要udma.write
    # Addr_nms_data = utensor.data().addr()
    # 取udma buffer地址, 赋值给mapped_base
    nms_data_addr = utensor.data().addr()
    print("&"*80)
    print(nms_data_addr)
    print("&"*80)
    mapped_base = nms_data_addr
    print('mapped_base =',mapped_base)
    print('#######################test udma#############################')
    threshold_f = iou
    arbase = mapped_base
    awbase = mapped_base
    reg_base = 0x100001C00
    
    if device.defaultRegRegion().read(reg_base + 0x008, True) != 0x23110200:
        raise RuntimeError("ERROR :: No NMS Hardware")
    group_num = int(np.ceil(box_num / 16))
    if group_num == 0:
        raise RuntimeError("ERROR :: group_num == 0")
    
    last_araddr = arbase + group_num * 160 - 8
    
    if last_araddr < arbase:
        raise RuntimeError("ERROR :: last_araddr < arbase")
    anchor_hpsize = int(np.ceil(box_num / 64))
    
    if anchor_hpsize == 0:
        raise RuntimeError("ERROR :: anchor_hpsize == 0")
    
    last_awaddr = awbase + anchor_hpsize * 8 - 8
    
    if last_awaddr < awbase:
        raise RuntimeError("ERROR :: last_awaddr < awbase")
    
    threshold = np.uint16(threshold_f * 2 ** 15)
    
    device.defaultRegRegion().write(reg_base + 0x014, 1, True)
    device.defaultRegRegion().write(reg_base + 0x014, 0, True)
    device.defaultRegRegion().write(reg_base + 0x01C, arbase, True)
    device.defaultRegRegion().write(reg_base + 0x020, awbase, True)
    device.defaultRegRegion().write(reg_base + 0x024, last_araddr, True)
    device.defaultRegRegion().write(reg_base + 0x028, last_awaddr, True)
    device.defaultRegRegion().write(reg_base + 0x02C, group_num, True)
    device.defaultRegRegion().write(reg_base + 0x030, 1, True)
    device.defaultRegRegion().write(reg_base + 0x034, threshold, True)
    device.defaultRegRegion().write(reg_base + 0x038, anchor_hpsize, True)

    device.defaultRegRegion().write(reg_base + 0x0, 1, True)
    print('region write done')
    start = time.time()
    while True:
        reg_done = device.defaultRegRegion().read(reg_base + 0x004, True)
        duration = time.time() - start
        if duration > 1:
            raise RuntimeError("NMS Timeout!!!")
        if reg_done != 0:
            break
    print('reg done')
    # mask_size = int(np.ceil(box_num / 8))
    # mask = np.empty(64000, dtype=np.uint8)
    # 读取udma buffer的结果 to host进行处理
    mask = []
    mask.append(np.asarray(utensor.to(rt.HostDevice.MemRegion())))
    # print('mask =',mask)
    
    # udma_chunk_.read(str(id(mask)), 0, mask_size)
    for i in range(len(score_index_vec)):
        idx = score_index_vec[i][1]
        mask_index = i // 8
        if i % 8 == 0 and (mask[mask_index] & np.uint8(1)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 1 and (mask[mask_index] & np.uint8(2)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 2 and (mask[mask_index] & np.uint8(4)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 3 and (mask[mask_index] & np.uint8(8)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 4 and (mask[mask_index] & np.uint8(16)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 5 and (mask[mask_index] & np.uint8(32)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 6 and (mask[mask_index] & np.uint8(64)) != 0:
            nms_indices.append(idx)
        elif i % 8 == 7 and (mask[mask_index] & np.uint8(128)) != 0:
            nms_indices.append(idx)
    print('nms_indices =',len(nms_indices))
    return nms_indices

def get_det_results(generated_output,scale_list,ANCHOR_LENGTH=ANCHOR_LENGTH,ANCHORS=ANCHORS,STRIDE = STRIDE,N_CLASS=NOC,BIT=BIT):
    id_list = []
    scores_list = []
    box_list = []
    icore_post_res = []
    # flatten icore_post_result
    for i in range(len(generated_output)):
        output = np.array(generated_output[i]).flatten()#模型中数据排布 e.g [1,1,133,96] ->[133*96]
        icore_post_res.append(output)

    print('INFO: get icore_post flatten results!')

    for i in range(len(icore_post_res)):
        objnum = icore_post_res[i].shape[0] / ANCHOR_LENGTH    
        tensor_data = icore_post_res[i]
        
        for j in range(int(objnum)):
            obj_ptr_start = j * ANCHOR_LENGTH
            obj_ptr_next = obj_ptr_start + ANCHOR_LENGTH
            if BIT==16:
                anchor_index = tensor_data[obj_ptr_next - 1]
                location_y = tensor_data[obj_ptr_next - 2]
                location_x = tensor_data[obj_ptr_next - 3]
            elif BIT==8:
                anchor_index1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 1]
                anchor_index2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 2]
                location_y1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 3]
                location_y2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 4]
                location_x1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 5]
                location_x2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 6]
                anchor_index = (anchor_index1 << 8) + anchor_index2
                location_y = (location_y1 << 8) + location_y2
                location_x = (location_x1 << 8) + location_x2

            _x = sigmoid(tensor_data[obj_ptr_start ]    * scale_list[i])
            _y = sigmoid(tensor_data[obj_ptr_start + 1] * scale_list[i])
            _w = sigmoid(tensor_data[obj_ptr_start + 2] * scale_list[i])
            _h = sigmoid(tensor_data[obj_ptr_start + 3] * scale_list[i])
            _s = sigmoid(tensor_data[obj_ptr_start + 4] * scale_list[i])
            
            class_ptr_start = obj_ptr_start + 5
            class_data_list = tensor_data[obj_ptr_start + 5:obj_ptr_start +5+N_CLASS]
            max_value = max(class_data_list)
            max_idx = list(class_data_list).index(max_value)
            realscore = _s / (1 + np.exp( - max_value * scale_list[i ]))

            x = (2*_x + location_x-0.5) * STRIDE[i]
            y = (2*_y + location_y-0.5) * STRIDE[i]
            w = 4 * (_w)**2  * ANCHORS[i][anchor_index][0]
            h = 4 * (_h)**2  * ANCHORS[i][anchor_index][1]

            scores_list.append(realscore)
            box_list.append(((x - w / 2), (y - h / 2), w, h))
            id_list.append(max_idx)
    return scores_list,box_list,id_list