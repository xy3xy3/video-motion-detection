# 环境配置

快捷写requirements.txt
```
pipreqs . --force
```

Python 3.12

服务端环境创建
```shell
conda create --name server_dc python=3.12 && conda activate server_dc
```


先安装cuda的pytorch
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

检查
```shell
import torch
print(torch.cuda.is_available())
```
服务端pip
```shell
cd server && pip install -r requirements.txt
```

客户环境创建
```shell
conda create --name client_dc python=3.12 && conda activate client_dc
```

客户端pip
```shell
cd client && pip install -r requirements.txt
```


# http版本

## 服务端

python服务端启动`httpserver.py`，后台挂着

```shell
conda activate server_dc
cd server
python httpserver.py
```

## 客户端-网页版

python启动客户端`app.py`，按照提示打开网页，点击按钮即可打开摄像头匹配或者调用`client`文件夹的视频匹配

```shell
conda activate client_dc
cd client
python app.py
```

## 客户端-命令行版

python启动客户端`httpclient.py`，结果存在`client/restmp`文件夹
