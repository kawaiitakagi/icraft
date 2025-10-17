import icraft
from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from icraft.host_backend import CudaDevice
import os
import platform
from .utils import *

STAGE = {
    "p": "parsed",
    "o": "optimized",
    "q": "quantized",
    "a": "adapted",
    "g": "BY",
}
def getJrPath(folderPath,stage,run_sim):
    jr_path = ["*", "*"]
    if not run_sim: stage = "g"
    for root, dirs, files in os.walk(os.path.abspath(folderPath)):
        dirs[:] = []  # 置空dirs列表，不深入遍历子目录
        for file in files:
            if STAGE[stage] + ".json" in file:
                jr_path[0] = os.path.join(root, file)
            if STAGE[stage] + ".raw" in file:
                jr_path[1] = os.path.join(root, file)

    assert os.path.exists(
        jr_path[0]
    ), "imodel path not right ,please check yaml:imodel:dir"
    mprint("Info:imodel file found at:‌{}".format(jr_path[0]), VERBOSE, 0)        

    return jr_path



def loadNetwork(JSON_PATH, RAW_PATH):
    network = Network.CreateFromJsonFile(JSON_PATH)
    network.lazyLoadParamsFromFile(RAW_PATH)
    return network


def openDevice(run_sim,ip,mmu_Mode = True, cuda_Mode = False,npu_addr = "0x40000000", dma_addr = "0x80000000"):
    current_os = platform.system()
    if current_os == "Windows":
        if run_sim:
            if cuda_Mode:
                return CudaDevice.Default()
            return HostDevice.Default()
        DEVICE_URL = "socket://ql100aiu@" + ip + ":9981?npu=" + npu_addr + "&dma=" + dma_addr

    elif current_os == "Linux":
        DEVICE_URL = "axi://ql100aiu?npu=" + npu_addr + "&dma=" + dma_addr
    device = Device.Open(DEVICE_URL)
    BuyiDevice(device).mmuModeSwitch(mmu_Mode)
    return device


def initSession(run_sim, network, device, mmu, open_speedmode = False,open_compressFtmp = False ):
    if run_sim:
        session = Session.Create( [HostBackend],network, [HostDevice.Default() ])
        return session
    else:
        session = Session.Create([ BuyiBackend, HostBackend], network, [ device,HostDevice.Default()])
        if mmu : return session
        buyi_backend = BuyiBackend(session.backends[0])
        if open_compressFtmp:
            buyi_backend.compressFtmp()
        if open_speedmode:
            buyi_backend.speedMode()
        return session

# 注意使用时候input_array的维度要与实际网络输入位置维度保持一致
def numpy2Tensor(input_array: np.ndarray,message) -> Tensor:
    if isinstance(message, Network):
        network = message
        if "InputNode" in network.ops[0].typeKey():
            input_value = network.ops[0].outputs[0]
        else:
            input_value = network.ops[0].inputs[0]
    elif(isinstance(message, Value)):
        input_value = message
    else:
        raise Exception("Error:输入numpy2Tensor的参数2类型错误,只能是Network类型和Value")
    input_tensortype = input_value.tensorType()
    # input_dtype = input_value.dtype.getStorageType()
    input_dtype = input_tensortype.getStorageType()
    input_tensortype.setShape(list(input_array.shape))
    # print(input_tensortype.shape)
    # print(input_tensortype.shape[0])
    # print(input_tensortype.shape()[1])
    # print(input_tensortype.shape()[2])
    # print(input_tensortype.shape()[3])
    
    if str(input_dtype) == '"@fp(32)"':
        input_array = input_array.astype(np.float32)
    elif str(input_dtype) == '"@fp(16)"':
        input_array = input_array.astype(np.float16)
    elif str(input_dtype) == '"@uint(8)"':
        input_array = input_array.astype(np.uint8)
    elif str(input_dtype) == '"@uint(16)"':
        input_array = input_array.astype(np.uint16)
    elif str(input_dtype) == '"@sint(8)"':
        input_array = input_array.astype(np.int8)
    elif str(input_dtype) == '"@sint(16)"':
        input_array = input_array.astype(np.int16)
    return  Tensor(input_array,input_tensortype)

# 当传入network 或者networkview时候适用于单输入的网络去构造输入tensor 
# 若传入value时候适用于构造网络中任意位置的tensor
# v3.7.0
# def numpy2Tensor(input_array: np.ndarray,message) -> Tensor:
#     if isinstance(message, Network) or isinstance(message, NetworkView) :
#         network = message
#         if "InputNode" in network.ops[0].typeKey():
#             input_value = network.ops[0].outputs[0]
#         else:
#             input_value = network.ops[0].inputs[0]
#     elif(isinstance(message, Value)):
#         input_value = message
#     input_tensortype = input_value.tensorType()
#     # input_dtype = input_value.dtype.getStorageType()
#     input_dtype = input_tensortype.getStorageType()
#     if input_dtype.isFP32():
#         input_array = input_array.astype(np.float32)
#     elif input_dtype.isFP16():
#         input_array = input_array.astype(np.float16)
#     elif input_dtype.isUInt8():
#         input_array = input_array.astype(np.uint8)
#     elif input_dtype.isUInt16():
#         input_array = input_array.astype(np.uint16)
#     elif input_dtype.isSInt8():
#         input_array = input_array.astype(np.int8)
#     elif input_dtype.isSInt16():
#         input_array = input_array.astype(np.int16)
#     return  Tensor(input_array,input_tensortype)