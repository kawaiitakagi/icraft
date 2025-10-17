
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

# qtset
Set-Location ./2_compile/qtset/
$url_qtset = "https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/compile/qtsets/AiATrack/"
& $wget_path $url_qtset --user $usrname --password $password -r -np -nH -c --reject="index.html*" -P './' --cut-dirs=8
Set-Location $oriModelPath

# io
Set-Location ./3_deploy/modelzoo/AiATrack/io/
$url_io = "https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/AiATrack/io/"
& $wget_path $url_io --user $usrname --password $password -r -np -nH -c --reject="index.html*" -P './' --cut-dirs=9
Set-Location $oriModelPath

# Deps
if ($download_deps){
    Set-Location ./3_deploy/Deps
    $url_deps="https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/Deps/thirdparty/"
    & $wget_path $url_deps --user $usrname --password $password -r -np -nH -c --reject="index.html*" -P './' --cut-dirs=7 --level=10
    Set-Location $oriModelPath
}

