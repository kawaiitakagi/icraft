# drl4vrp
![version](https://img.shields.io/badge/icraft_ver-3.7.1-gold?style=flat&logo=data:image/png;base64,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)  ![author](https://img.shields.io/badge/author-lxm-blue)<br>![metrics](https://img.shields.io/badge/metrics-待测-lightblue?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA8CAYAAAA6/NlyAAAAAXNSR0IArs4c6QAAAAlwSFlzAAAPYQAAD2EBqD+naQAABLZJREFUaEPtmmnIpWMYx39/+75lyy6ELGNohA8GIUvEB6RBtomYmhhDIlskkmUSWcLIyDKU9QOy88FSjCIylmSZsoxl7OPSf7pOPY7zvs8573Oa55x3nqveOr33fT/P/buv5f7f9zliGTMtY7w0wOPd442HGw+PsxVoQrrdoRGxOrBN4f8LJC0YVseXejgidgXeLQBeJ2lmAzwkK9B4uEMONyFdlsMRsTywPrAKsFDSj4MS8X0P6YjYCJgK7A2sCnwPvAA8IOm7usH7ChwRWwNP5DZm79oC+BV4EjhT0sI6ofsGHBGbA48Du40C9DRwvKSf6oLuC3BErAlcDpwNrJQw3wBfALtkLrcY7wBmSPq5Duh+Ae8P3AVslRBfA1cCLwJHARcAa2WbvTvD/SX9s7ShKwNHhHP1UuB8YDngb8BenClpUUpTA18IrJA5/TpwmqQPhxF4U+Clgt52KB8p6c0WTETY83cCB8CSWxYvihdhliR/7tkiwou3WJKLYtfWDw87ZB9J7/rFDwJTJC0uziIipgC3FELb+vxQSQ7/ri33+B2BycDcXg8y/QA24LGFGR8s6dl2gohYA3gUOCjbvCAO69nd0kbE2sAprvTAzsCNki7qdrz7VQLO1bawaBWkz4HtJP3VaRIR4eL2fKFtnqQJZROOCM/THr0W2AlYLcf8Buwj6Z2yZ7TaqwLvAbxVeNlsSSeP9vKIeAOYVOgzWdLLIyyQ89Q14hzgdMBn83bzTuDU+L0b6KrAnsj1hRcdJ+mhEuCTgLsLOX8PcGqn4hMRvni4DDgEWK8wpviKX4Bzc5v7T93oNI+qwI+5IueDvb9OkvRRCbAVmbV16xblE2BfSV92yHunypb5t316eYe2fq7Sr7mtm22uKrDDbU9gIrAucImkH0qArcpmAa3Qt+KaJuneknEOb4e+DyW2RYAXy0ruT+Bi4CZJ/jyiVQLuJmc6eM3vPCHFycrZPicr9h8jPTMivLjO/02yj/f5w1PIuPLb8zeUHU6WOrAnGxG7A85de8f2nrcbSW+PAnwE4ChYJ/vclprcnu7a6gL2nmwRYjFiOeoKa319eyflFRHuc1X2WTFDeLqjpF3glJHXApxePibD2mLC9ipwtKRvO6TBZnk4aYmW+RkRr5QBtrfXCezi5SJUPD+fKOm+4iRTdByWWnzjbPO52irNur0nqw04vWxJej/gOzDbp8CBklx9l1ietq4Bzkpl6PD30fPqsRwv6wb20XJuVtsWo/X2VEmWrAbeLw8kG2aHr6zHJb3fk2uzc63ACWRt/Exhu/HWdGsqLIew78K2LcDdDEwfi3f9jEEA9pXQtAxT33IuWYcM7w0A53rLHPITq1z71g6cXjbYFdbUhTux9oj9LBXZU2MJ5daYgQBO6C3ymuiMvAoqcn2cIf5wmXQsW4yBAU5oC5K9gPPyCGld/lyKlA9GOmeXQRbbBwq4l4mPtW8D3L5yzRfi0PwCYKz5VMe4JoebHP7/r3h8teKvVgbRXF9G/UnVWEJ6EEFbc5ogad5oE2yAu8jhce9hX6X6EnwYbL6/k64U0sNA2cscS3O4l4cNQ98GeBi8VGWOjYerrN4wjG08PAxeqjLHfwG92SNbyY+y8QAAAABJRU5ErkJggg==)  ![speed](https://img.shields.io/badge/speed-OK-green?style=flat&logo=fastapi)<br><a href="../../index.md#drl" target="_blank"><img alt="模型清单" src="https://img.shields.io/badge/drl-模型清单-cornflowerblue?logo=quicklook"></a><br>![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Ubuntu-green)

# 下载

✨ 一键下载开发流程中所需的各种文件，包括编译使用的量化校准集、运行时工程的依赖库，以及输入输出文件。

💡 推荐使用linux版下载脚本，其wget包含断网自动重连功能，不会出现下载文件遗漏情况。

## windows
📌 第一次使用，请在C盘根目录下新建`icraft_auth.txt`，保存下载站账号密码，以换行符分隔

需要事先下载windows版本wget：

（若点击以下链接后未直接下载，请选择 ***1.20.3*** 版本下的对应系统链接进行下载）

[x86系统wget下载](https://eternallybored.org/misc/wget/1.20.3/32/wget.exe)		[x64系统wget下载](https://eternallybored.org/misc/wget/1.20.3/64/wget.exe)

使用时需要将wget.exe的路径作为命令行参数传入，注意不是exe的父文件夹目录，而是包含wget.exe的完整绝对路径：

不下载Deps：`./download.ps1 "PATH_TO_WGET_EXE"`

如果您是第一次使用我们的模型库，请下载包括工程依赖库的所有文件：`./download.ps1 "PATH_TO_WGET_EXE" -d`

💡 下载过程中可能因网络问题出现中断情况，需 **自行重新运行** 下载脚本。

## linux

📌 第一次使用，请在/usr根目录下新建`icraft_auth.txt`，保存下载站账号密码，以换行符分隔

为确保文件格式正确，请在运行脚本前安装格式转换工具`dos2unix`，并执行格式转换命令：
```shell
sudo apt-get install dos2unix
dos2unix /usr/icraft_auth.txt
dos2unix ./download.sh
```

如果您是第一次使用我们的模型库，请下载包括工程依赖库的所有文件：`./download.sh -d`

如果之前已经在使用别的模型时下载过Deps依赖库，可以直接将其中的thirdparty部分复制到路径`3_deploy/Deps`，只需下载量化校准集和输入输出文件即可：`./download.sh`


🌟 Tips：

- 若想要直接获取原始weights和导出保存的模型，可分别前往 [weights](https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/weights/) 和 [fmodels](https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/compile/fmodels/) 网页上根据框架及模型名寻找并下载。


# 0. 文件结构说明

AI部署模型需要以下几部分文件

- 0_drl4vrp    >模型原始工程，需要自行下载
- weights              >存放原始权重，需要自行下载
- 1_scripts            >若干脚本，用于保存部署所需模型、模型导出验证等功能
- 3_deploy            >将模型部署到硬件时需要的c++工程

# 1. python工程准备

## 1. **模型来源：**

- code：https://github.com/mveres01/pytorch-drl4vrp
- branch：master
- commit_id：5b9b86e
- weights：https://drive.google.com/open?id=1wxccGStVglspW-qIpUeMPXAGHh2HsFpF

## 2. **保存模型**

**目的：将模型保存成可部署的形态**

1）根据模型来源中的地址：[https://drive.google.com/open?id=1wxccGStVglspW-qIpUeMPXAGHh2HsFpF ](https://drive.google.com/open?id=1wxccGStVglspW-qIpUeMPXAGHh2HsFpF)，下载原始weights，存放于 `/weights`文件夹中

<div style="background-color: #FFFFCC; color: #000000; padding: 10px; border-left: 5px solid #FFA500;">
注意：

* 有时开源的weights url可能会变更。如果我们提供的weights url失效，请根据原工程相应的branch以及commit版本寻找正确的下载链接
* 若上述weights url永久失效,请联系本模型库相关人员获取权限下载
</div>

2）根据模型来源中的地址，下载指定commit id版本的源代码，文件夹名称要设置为：0_drl4vrp

```shell
# 在此模型根目录
mkdir 0_drl4vrp
git clone -b master https://github.com/mveres01/pytorch-drl4vrp 0_drl4vrp
cd 0_drl4vrp
git checkout 5b9b86e
```

3）进入1_scripts执行保存模型脚本

```shell
# 在此模型根目录
cd 1_scripts
python 1_save.py
```

**1_scripts提供脚本说明：**

- **环境要求：**Icraft编译器对**导出框架模型时**使用的**框架版本**有要求。即以下脚本中所有导出模型的脚本`1_save.py  `，必须在要求的框架版本下执行，其他脚本不限制。要求的版本：
  
  - **pytorch**：支持pytorch1.9.0、pytorch2.0.1两个版本的原生网络模型文件（.pt格式），以及pytorch框架保存为onnx（opset=17）格式的模型文件（.onnx格式）
  - **paddle**：仅支持PaddlePaddle框架保存为onnx（opset=11）格式的模型文件（.onnx格式），不支持框架原生网络模型文件
  - **darknet**：支持Darknet框架原生网络模型[GitHub - pjreddie/darknet: Convolutional Neural Networks](https://github.com/pjreddie/darknet)
  
- 0_infer.py                      	    >可以推理一张图并得到最终结果，模型原始权重会从 `/weights	`中寻找，需要您预先下载

- 1_save.py                              >保存模型，保存好的用于部署的模型，会存放在 `/3_deploy/modelzoo/drl4vrp/imodel`

  <div style="background-color: #FFFFCC; color: #000000; padding: 10px; border-left: 5px solid #FFA500;">
  保存模型时的修改点：   

  1. 将模型由3输入修改为5输入<br>
  2. 导出迭代一次的结果(max_steps=1)<br>
  3. 将ptr计算之后的操作去掉，并添加last_hh作为输出算子<br>
  </div>
  
- 2_save_infer.py                    >用修改后保存的模型做前向推理，验证保存的模型与原模型是否一致


# 2. 部署模型

目的：编译c/c++可执行程序，在硬件上调用onnxruntime进行前向推理

模型库以ubuntu操作系统为例：

1. **编译环境准备**
   - os: ubuntu20.04
   - cmake>=3.10
   - compiler: aarch64-linux-gnu-g++/aarch64-linux-gnu-gcc

2. **版本依赖下载**

   请至[modelzoo_pub/deploy/Deps/onnxruntime.zip](https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/Deps/onnxruntime.zip)下载主要版本依赖，解压后存放在`\3_deploy\modelzoo\drl4vrp\onnxruntime`。<br>
   下载后文件结构为：
   ```shell
   ├── include
   │   ├── cpu_provider_factory.h
   │   ├── onnxruntime_c_api.h
   │   ├── onnxruntime_cxx_api.h
   │   ├── onnxruntime_cxx_inline.h
   │   ├── onnxruntime_float16.h
   │   ├── onnxruntime_run_options_config_keys.h
   │   ├── onnxruntime_session_options_config_keys.h
   │   ├── onnxruntime_training_c_api.h
   │   ├── onnxruntime_training_cxx_api.h
   │   ├── onnxruntime_training_cxx_inline.h
   │   └── provider_options.h
   └── lib
      ├── aarch64
      │   ├── libonnxruntime.so
      │   └── libonnxruntime.so.1.17.1
      └── x64
         ├── libonnxruntime.so
         └── libonnxruntime.so.1.17.1
   
   ```
   
3. **编译c++程序**
  目前只支持linux_x64和linux_aarch64环境的Release编译，需要提前安装好aarch64交叉编译器(apt install g++-aarch64-linux-gnu)
  * 交叉编译 aarch64可执行文件: 
   ```shell
   #在3.1所需的linux编译环境中
   cd 3_deploy/modelzoo/drl4vrp/build_arm
   cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
   make -j12
   ```
   * 运行前需要手动把libonnxruntime.so, libonnxruntime.so.1.17.1复制到运行环境中，例如 usr/lib下   
   * 将编译得到的的可执行文件`drl_run`复制至片上系统`/home/fmsh/ModelZoo/drl4vrp/`即可


   模型输入均在 `3_deploy/modelzoo/drl4vrp/io`中，可根据需要进行替换，生成方式如下：
  ```shell
   # input
   static =torch.rand((1, 2, 20))
   dynamic = torch.zeros((1,1,20))
   decoder_input = torch.zeros((1,2,1))
   last_hh = torch.zeros((1,1,128))
   mask = torch.ones((1,20))
  ```
  最后手动放入对应`3_deploy/modelzoo/drl4vrp/io`中

5. **部署环境检查**

   * 以root账户登录片上系统terminal（ssh或串口皆可），模型库默认的模型存放路径为以下目录，如果没有请预先创建：

   ```
   /home/fmsh/ModelZoo/
   ```

   * 将3_deploy中所有文件夹复制到以上目录中（如果**Deps**中已经存在**相同**版本的依赖则可以不必再复制）

   * 3_deploy/modelzoo/drl4vrp工程结构如下   
    ├── build   
    ├── build_arm   
    ├── CMakeLists.txt   
    ├── CMakePresets.json   
    ├── cmake   
    ├── onnxruntime   
    ├── imodel   
    ├── io   
    ├── drl_run   
    └── source   
   * 3_deploy/modelzoo/drl4vrp工程文件说明   
     * build: linux_x64下的运行示例，drl_run是source工程的编译结果   
     * build_arm: linux_aarch64下的运行示例，drl_run是source工程的编译结果，运行前需要手动把libonnxruntime.so, libonnxruntime.so.1.17.1复制到运行环境中，例如 usr/lib下   
     * drl_run: 模型前向推理工程   
     * CMakeLists.txt: CMake配置文件   
     * CMakePresets.json: CMake配置文件   
     * cmake: CMake配置文件   
     * onnxruntime: include和lib依赖文件，include文件是共享的，lib文件分别对应aarch64交叉编译和x64编译环境，cmake编译会自动选择依赖；   
     * source: 工程前向代码main.cpp   
     * io: 输入，可根据需求手动生成替换
6. **执行程序**

   运行前请确保已经手动将3_deploy\modelzoo\drl4vrp\onnxruntime\lib\aarch64下的libonnxruntime.so, libonnxruntime.so.1.17.1复制到运行环境中，例如 usr/lib下, 然后执行：
   ```
   cd /home/fmsh/ModelZoo/modelzoo/drl4vrp
   chmod 777 *
   ./drl_run
   ```

   在终端可查看程序运行结果，显示最终迭代的输出及耗时
   
   



# 3. 模型性能记录

| drl4vrp | input shape     | hard time      |
| -------------- | --------------- | -------------- | 
| float          | [1,2,20],[1,1,20],[1,2,1],[1,1,128],[1,20] | 26ms              | 

