# 环境配置

服务端pip
```shell
pip install websockets ultralytics opencv-python torch torchvision torchaudio pillow numpy fastapi
```

客户端pip
```shell
pip install websockets opencv-python numpy pillow flask
```

# http版本

## 服务端

python服务端启动`httpserver.py`，后台挂着

## 客户端-网页版

python启动客户端`app.py`，按照提示打开网页，点击按钮即可打开摄像头匹配或者调用`client`文件夹的视频匹配

## 客户端-命令行版

python启动客户端`httpclient.py`，结果存在`client/restmp`文件夹
