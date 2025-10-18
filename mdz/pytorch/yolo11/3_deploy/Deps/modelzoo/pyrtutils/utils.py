from typing import List
import platform
import json 
import os
import shutil
from matplotlib import pyplot as plt
import xlsxwriter

import icraft
import icraft.xir as ir
import icraft.xrt as rt 
import icraft.host_backend as hb 
import icraft.buyibackend as bb
import numpy as np 
import pandas as pd

STAGE={
    "p":"parsed",
    "o":"optimized",
    "q":"quantized",
    "a":"adapted",
    "g":"BY",
}
def dmaInit(device, input_tensor, shape, imk):
    if imk:
        h,w,c=shape[0],shape[1],shape[2]
        demo_reg_base = 0x1000C0000
        uregion_=device.getMemRegion("udma")
        utensor = input_tensor.to(uregion_)#data transfer ps->udma + IMK(udma->pl)
        ImageMakeRddrBase = utensor.data().addr()
        ImageMakeRlen = ((w * h - 1) // (24 // c) + 1) * 3
        ImageMakeLastSft = w * h - (ImageMakeRlen - 3) // 3 * (24 // c)
        device.defaultRegRegion().write(demo_reg_base + 0x4, ImageMakeRddrBase, True)
        device.defaultRegRegion().write(demo_reg_base + 0x8, ImageMakeRlen, True)
        device.defaultRegRegion().write(demo_reg_base + 0xC, ImageMakeLastSft, True)
        device.defaultRegRegion().write(demo_reg_base + 0x10, c, True)
        device.defaultRegRegion().write(demo_reg_base + 0x1C, 1, True)
        device.defaultRegRegion().write(demo_reg_base + 0x20, 0, True)
        # imk start
        device.defaultRegRegion().write(demo_reg_base, 1, True)
    return 0
  
def openDevice(sim, ip=0):
    if not sim:
        URL_PATH=""
        if platform.machine() == 'aarch64':
            URL_PATH = R"axi://ql100aiu?npu=0x40000000&dma=0x80000000";
        else:
            URL_PATH = Rf"socket://ql100aiu@192.168.125.{ip}:9981?npu=0x40000000&dma=0x80000000"
        device = rt.Device.Open(URL_PATH)
        return device
    else:
        return 0
    
def ses(sim, network, device,view_idx=0,):
    if sim:
        session = rt.Session.Create([ hb.HostBackend ], network.view(view_idx), [ rt.HostDevice.Default() ])
    else:
        session = rt.Session.Create([ bb.BuyiBackend, hb.HostBackend], network.view(view_idx), [ device,rt.HostDevice.Default()])
    return session

def net(model_path,model_name,stage):
    jpath=model_path+"\\"+model_name+"_"+STAGE[stage]+".json"
    rpath=model_path+"\\"+model_name+"_"+STAGE[stage]+".raw"
    network = ir.Network.CreateFromJsonFile(jpath)
    network.loadParamsFromFile(rpath)
    return network

def fpgaOPlist(network):
    # only used in adapt&by stage
    customop_set = set()
    oplist = network.ops
    for op in oplist:
        if "customop" in op.typeKey():
            customop_set.add(op.typeKey())
    return customop_set

def get_scale(GENERATED_JSON_FILE):
    # 读取json文件获取反量化系数
    with open(GENERATED_JSON_FILE,'r') as f:
        net = json.load(f)
    scale_list = []
    # 从json文件中获取输出的 norm_ratio 用于反量化
    for ftmp in net["ops"][-2]["inputs"]:
        scale_list.append(ftmp["dtype"]["element_dtype"]["normratio"][0]["value"])
    return scale_list
def getOutputNormratio(network):
    # 从xir.network获取输出的norm_ratio用于反量化
    net_out_results = network.outputs()
    scale_list = []
    for value in net_out_results:
        scale = value.dtype.getNormratio().data
        scale_list.append(scale[0])
    return scale_list
# 统计模型时间计时相关
def draw_total_time(op_names,df_sum,filename,respath,resname):
    # 画图
    plt.grid(ls="--", alpha=0.5)

    hard_time = list(df_sum["hard_time"])
    res_time = list(df_sum["res_time"])

    # 绘制堆叠柱状图
    plt.bar(op_names,hard_time,width=0.5,color="blue",label="hard_time")
    plt.bar(op_names,res_time,width=0.5,color="orange",label="res_time", bottom=hard_time)

    plt.xlabel('op_names')
    plt.ylabel('time(ms)')
    plt.legend()

    for i in range(len(op_names)):
        max_y = round(hard_time[i]+res_time[i],2)
        plt.text(op_names[i], max_y+0.2, max_y, va="bottom", ha="center")

    # 保存图片
    plt.savefig(respath+filename+resname+'.png', bbox_inches='tight')

    # plt.show()


def draw_per_op_time(op_id,df,filename,respath,resname):
    #画图
    fig = plt.figure(figsize=(18, 10))
    plt.grid(ls="--", alpha=0.5)

    index = np.arange(len(op_id))
    width=0.5

    hard_time = list(df["hard_time"])
    res_time = list(df["res_time"])

    plt.bar(index,hard_time,width=width,color="blue",label="hard_time")
    plt.bar(index,res_time,width=width,color="orange",label="res_time",bottom=hard_time)#绘制柱状图

    plt.xticks(index, labels=op_id)
    plt.xticks(rotation= 45,fontsize=8)

    plt.xlabel('op_id')
    plt.ylabel('time(ms)')
    plt.legend()



    # 保存图片
    plt.savefig(respath+filename+resname+'.png', bbox_inches='tight')

    # plt.show()