---

title: README
image: https://i3.read01.com/SIG=1n8sj6l/304f70713475676f4759.jpg

---

# README
- ref: https://github.com/Esun-DF/ai_competition_api_sharedoc/tree/features/2022-share-doc/model-spec

---


|組員|信箱|職責|
|:--|:--|:--|
|林育辰 Yu-Chen Lin|i98565412@gmail.com|API Server|
|梁俊彥 Chon-In Leong|  ivan000105@gmail.com| NLP Model|

 
 
## 程式架構 (TBD)
* 請將程式碼有依據的分類，列出2層內的檔案架構
* 以下架構為"**必需**"要有的檔案及資料夾，請自行整理程式架構成以下格式
* 層數或內容可視情況補充調整
    
    ```
    $tree -L 2
    
    .
    ├── api.py
    ├── data_preprocess
    │   └── preprocess.py
    ├── inference
    │   ├── config.py
    │   ├── dataset_inference.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── perplexity.py
    │   ├── __pycache__
    │   ├── search_method.py
    │   └── word_converter.py
    ├── model
    │   ├── config.py
    │   ├── data.py
    │   ├── evaluation.py
    │   ├── model.py
    │   ├── perplexity.py
    │   ├── __pycache__
    │   └── train.py
    ├── README.md
    └── requirements.txt
    ```

## 程式環境
### 系統平台
在諸多系統平台考量下，如 TWCC 雲端服務、Google Cloud Platform 等，在費用以及其他因素考量之下，決定模型訓練以及 API Server 架設，均使用 **國立臺灣大學 多媒體資訊檢索實驗室** 之實驗室主機配備。

因此模型訓練、程式撰寫以遠端連線(SSH) 主機為主，API Server 之 架設/技術 均為隊伍自行完成。

主機配備規格如下：


|設備 | 型號 / 版本 |Note|
|:--|:--|:--|
|CPU| Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz|20 cores|
|GPU| GeForce GTX 1080 Ti| 2 cores|
|RAM| 128 GB||
|OS| Ubuntu 21.04  (Hirsute Hippo)||

### 程式語言
無論是 AI model 或是 API Server 方面，最終均以 **Python 3.9.5** 版本為主撰寫。

有考慮過 後端 方面以 Node.js 或 Go 等語言撰寫，而後認為後端與 AI model 語言相同較方便處理，並且以主辦方提供之 `api.py` 著手較不容易有疏漏之處，故以 Python 為主要語言。

### 函式庫
**requirements.txt** 請參考如下：
```
flask==1.1.2
numpy==1.22.2
gunicorn==20.1.0
ckiptagger==0.2.1
tensorflow==2.8.0
transformers==4.5.0
torch==1.11.0+cu113
argparse==1.4.0
pandas==1.4.1
einops==0.4.1
service_streamer==0.1.2
nltk==3.7
tqdm==4.62.3
evaluate==0.1.1
```

值得注意的有以下幾點：
- api 架設以 `flask` 為主
- 自行訓練模型時，有使用斷詞工具 `ckiptagger`，相較 `jieba`，其在繁體中文斷詞方面表現較優一些
- [service streamer](https://github.com/ShannonAI/service-streamer) 是 Github 上的開源專案，可協助 api 導入 batch 功能以增進效能


## 執行方式及原始碼 (TBD)
* 請說明從資料源頭至答案產出執行過程，範例可以參考下方**執行範例** 
* 請於重點設計程式上加上註解
* 在資料處理的部分請仔細說明所使用的工具，完整處理邏輯，如果有用到額外資料也請附上資料來源以及處理邏輯
* 在推論的部分請仔細說明推論邏輯以及各個函數的作用

## 執行範例
範例執行功能為最低描述需求，參賽者可依照團隊需求增述

* 虛擬環境
    ```
    # 安裝虛擬環境
    $ python3 -m venv venv
    
    # 啟動虛擬環境 
    $ source venv/bin/activate
    
    # 安裝所需套件
    $ pip install -r requirements.txt 
    ```

* 資料前處理

    在資料前處理的階段中，我們針對了部分大寫數字進行了轉換，而其中一些大寫數字（參肆伍陸拾）因爲出現了不少同音異義 / 異音異義的詞，不好處理故選擇跳過

    在前處理的同時，我們也把Ground Truth的60000句文本提取出來，結合來自CLMAD Corpus的Finance/Stock Topic文本，利用NLTK套件建立了1/2/3-Gram的Language Model，用於後續的困惑度計算
    
    ```
    # 執行前處理
    $ cd data_preprocess
    $ python data_preprocess.py
    ```

* 模型訓練
    本次參賽用的模型採用了基本的Transformer Encoder-Decoder模型，採用了6層Encoder/Decoder的結構

    ```
    # 執行模型訓練
    $ python training.py
    ```
* 模型推論
    * 說明是否進行額外的後處理
    * 說明是否優化API執行時間
    ```
    # 執行模型推論
    $ python inference.py
    ```
