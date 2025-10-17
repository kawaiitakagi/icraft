#!/bin/bash
# dir path must have '/' at the end

lines=()
while IFS= read -r line; do
    lines+=("$line")
done < /usr/icraft_auth.txt
name="${lines[0]}"
key="${lines[1]}"

ori_model_path=$PWD

# download_deps=$(1:-true)

# io
cd ./3_deploy/modelzoo/sac/io/
url_io=https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/sac/io/
c1=$(grep -oF "/" <<< $url_io | wc -l)
c2=$(expr $c1 - 3)
c3=$(expr $c1 - 4)
# 下载文件里的内容到当前文件夹用c2;带最后一层文件夹结构选c3
level=$c2
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -np -nH -c --reject="index.html*" --user $name --password $key -P './' -r $url_io --cut-dirs=$level
cd $ori_model_path

# onnxruntime
cd ./3_deploy/modelzoo/sac/onnxruntime/
url_onnxruntime=https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/onnxruntime/
c1=$(grep -oF "/" <<< $url_onnxruntime | wc -l)
c2=$(expr $c1 - 3)
c3=$(expr $c1 - 4)
# 下载文件里的内容到当前文件夹用c2;带最后一层文件夹结构选c3
level=$c2
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -np -nH -c --reject="index.html*" --user $name --password $key -P './' -r $url_onnxruntime --cut-dirs=$level
cd $ori_model_path

