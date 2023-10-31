[系統架構圖](https://lucid.app/lucidchart/40a689cc-98a8-4ca0-9b61-b78bf0cbfcd9/edit?beaconFlowId=6489020DC29E4D52&invitationId=inv_53b45015-4c0b-4648-85af-01a130d3591e&page=0_0#)

# 注意事項
.keras是手語辨識模型
.h5是語序辨識模型
EngToChinese.txt是word embedding的資料集

# 安裝須知
python version==3.8.10
若是mac平台，mediapipe需要額外進行設定
使用
```
pip install -r requirements.txt
或
pip3 install -r requirements.txt
```
進行安裝


## 更換模型注意事項
手語辨識模型在VideoRecognition.py中進行更改，需要更改的地方有:
1. 模型名稱
2. actions的list

語序辨識模型在main2.py與templates\index.html中進行更改
1. 在main2.py的self.available_models新增語序模型dict
2. 在templates\index.html的select name="model" id="model"後面新增下拉式選項，value為新增dict的key，讓管理者能夠在前端選擇並切換語序模型。

