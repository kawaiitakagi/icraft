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

# qtset
cd ./2_compile/qtset/
url_qtset=https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/compile/qtsets/coco/
c1=$(grep -oF "/" <<< $url_qtset | wc -l)
c2=$(expr $c1 - 3)
c3=$(expr $c1 - 4)
# 下载文件里的内容到当前文件夹用c2;带最后一层文件夹结构选c3
level=$c2
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -np -nH -c --reject="index.html*" --user $name --password $key -P './' -r $url_qtset --cut-dirs=$level
cd $ori_model_path

# io
cd ./3_deploy/modelzoo/yolov5s6//io/
url_io=https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/modelzoo/yolov5s6/io/
c1=$(grep -oF "/" <<< $url_io | wc -l)
c2=$(expr $c1 - 3)
c3=$(expr $c1 - 4)
level=$c2
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -np -nH -c --reject="index.html*" --user $name --password $key -P './' -r $url_io --cut-dirs=$level
cd $ori_model_path

# Deps
# if [ "$download_deps" = true ]; then
if [ "$1" = "-d" ]; then
    cd ./3_deploy/Deps
    url_deps="https://download.fdwxhb.com/data/04_FMSH-100AI/100AI/04_modelzoo/modelzoo_pub/deploy/Deps/thirdparty/"
    c1=$(grep -oF "/" <<< $url_deps | wc -l)
    c2=$(expr $c1 - 3)
    c3=$(expr $c1 - 4)
    level=$c3
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -np -nH -c --reject="index.html*" --user $name --password $key -P './' -r $url_deps --cut-dirs=$level  --level=10

fi