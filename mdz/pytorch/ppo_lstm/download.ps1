
$txtFilePath = "c:\icraft_auth.txt"

$wget_path     = $args[0]
$download_deps = $args[1]

# 检查必需的参数是否为空
if (-not $wget_path){
    Write-Error "Necessary param missing: Please give path to wget.exe as the first command-line argument!"
    exit 1
}

# test
# $wget_path = "E:\EdgeDownload\wget-1.21.4-win64\wget.exe"

if(Test-Path $txtFilePath){
    # 逐行读取文本文件内容
    $content  = Get-Content $txtFilePath -TotalCount 2

    $usrname  = $content[0].Trim()
    $password = $content[1].Trim()

} else {
    Write-Output "Please create file c:\icraft_auth.txt, containing usrname and password."
}

$oriModelPath = $PWD

# io
Set-Location ./3_deploy/modelzoo/ppo_lstm/io/
$url_io = "https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/ppo_lstm/io/"
& $wget_path $url_io --user $usrname --password $password -r -np -nH -c --reject="index.html*" -P './' --cut-dirs=9
Set-Location $oriModelPath

# onnxruntime
Set-Location ./3_deploy/modelzoo/ppo_lstm/onnxruntime/
$url_onnxruntime = "https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/onnxruntime/"
& $wget_path $url_onnxruntime --user $usrname --password $password -r -np -nH -c --reject="index.html*" -P './' --cut-dirs=8
Set-Location $oriModelPath

