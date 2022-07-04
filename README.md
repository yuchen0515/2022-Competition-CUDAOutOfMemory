---

title: README
image: https://i3.read01.com/SIG=1n8sj6l/304f70713475676f4759.jpg

---



# README
> Reference example: https://github.com/Esun-DF/ai_competition_api_sharedoc/tree/features/2022-share-doc/model-spec

|組員|信箱|職責|
|:--|:--|:--|
|林育辰 Yu-Chen Lin|i98565412@gmail.com|API Server|
|梁俊彥 Chon-In Leong|  ivan000105@gmail.com| NLP Model|

 
 
## 程式架構
* 程式碼之結構，已依循 [官方要求](https://github.com/Esun-DF/ai_competition_api_sharedoc/blob/features/2022-share-doc/model-spec/README.md)，**必須** 檢附之資料夾以及檔案均有附上。(例：`data_preprocess`, `inference`, `model`, etc.)
    ```
    $tree -L 2
    
    .
	├── data_preprocess
	│   ├── data_preprocess.py
	│   ├── external_data_preprocess.py
	│   └── __init__.py
	├── Dockerfile
	├── inference
	│   ├── config.py
	│   ├── dataset_inference.py
	│   ├── inference.py
	│   ├── __init__.py
	│   ├── model.py
	│   ├── perplexity.py
	│   ├── search_method.py
	│   └── word_converter.py
	├── model
	│   ├── config.py
	│   ├── data.py
	│   ├── evaluation.py
	│   ├── __init__.py
	│   ├── model.py
	│   ├── perplexity.py
	│   └── train.py
	├── nlp_fluency
	│   ├── example.py
	│   ├── __init__.py
	│   ├── LICENSE
	│   ├── models.py
	│   ├── __pycache__
	│   ├── README.md
	│   └── train_ngramslm.py
	├── __pycache__
	│   └── api.cpython-39.pyc
	├── README.md
	└── requirements.txt
    ```
* 補充
    * [nlp_fluency](https://github.com/baojunshan/nlp-fluency) 為 Github 上開源專案，用以協助評估語句之「流暢度」所用。
    * 可直接用 Docker 架設基於本隊程式碼的 API Server。

## 程式環境
### 系統平台
在諸多系統平台考量下，如 TWCC 雲端服務、Google Cloud Platform 等，在費用以及其他因素考量之下，決定模型訓練以及 API Server 架設，均使用 **國立臺灣大學 多媒體資訊檢索實驗室** 之主機配備。

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
jinja2<3.1.0
markupsafe==2.0.1
protobuf==3.20.1
flask==1.1.4
numpy==1.22.2
gunicorn==20.1.0
ckiptagger==0.2.1
tensorflow==2.8.0
transformers==4.5.0
torch==1.11.0
argparse==1.4.0
pandas==1.4.1
einops==0.4.1
service_streamer==0.1.2
nltk==3.7
tqdm==4.62.3
evaluate==0.1.1
python-dateutil==2.8.2
opencc-python-reimplemented==0.1.6
```

值得注意的有以下幾點：
- api 架設以 `flask` 為主
- 自行訓練模型時，有使用斷詞工具 `ckiptagger`，相較 `jieba`，其在繁體中文斷詞方面表現較優一些
- [service streamer](https://github.com/ShannonAI/service-streamer) 是 Github 上的開源專案，可協助 api 導入 batch 功能以增進效能。


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

    在前處理的同時，我們也把Ground Truth的60000句文本提取出來，結合來自CLMAD Corpus Dataset的Finance/Stock Topic文本，利用NLTK套件建立了1/2/3-Gram的Language Model，用於後續的困惑度計算
    
    ```
    # 執行前處理
    $ python -m data_preprocess.data_preprocess
    # 下載/轉換/生成額外的噪音文本
    $ python -m data_preprocess.external_data_preprocess
    
    ```
    

* 模型訓練
    本次參賽用的模型採用了基本的Transformer Encoder-Decoder模型進行訓練，採用了6層Encoder/Decoder的結構

    模型在執行面上是以文本翻譯之概念爲基礎進行文本修正，旨在嘗試將“錯誤文本”翻譯成“正確文本”
    
    ```
    # 一些模型訓練參數
    Optimizer=AdamW
    Activation Function=Relu
    Learing Rate=0.0001
    Dropout=0.2
    Epoch=6
    ```

    ```
    # 執行模型訓練
    $ python -m model.train
    ```
* 模型推論
    用Input提供之文本輸入至Model中，利用Greedy Search Inference出每句Input之修正句
    
    其後計算各句（含Input句）之3-Gram困惑度，取最低者作最終Output
    
    ```
    # 執行模型推論
    $ python -m inference.inference --input_path INPUT_PATH --model_path MODEL_PATH --ngram_dict_dir NGRAM_DICT_DIR --nlp_fluency_dir NLP_FLUENCY_DIR
    ```
* API Server
    * 本隊提供 **Docker** 之 API 伺服器架設方式
    * 您只需安裝好 docker ([Tutorial](https://docs.docker.com/engine/install/ubuntu/))，接著在本專案目錄下依循下列指令操作即可：
    ```bash
    sudo docker build . -t mathlin/cudaoutofmemory
    sudo docker run -p 614:614 -d mathlin/cudaoutofmemory
    ```
    * 附註
        * `mathlin/cudaoutofmemory` 為建立之容器名稱，您可以按照自己的喜好命名。
        * build 階段時，會按照 Dockerfile 指定安裝、執行檔案，最終會在後台持續運行 `api.py`
        * run 階段則是啟動容器 (Image)，此時即可按您的 ip，port:614 傳送資料並取得結果
        * 此啟動形式一般只可用 CPU，本隊於正式賽時是使用 nvidia-docker (較為麻煩) 啟動 GPU 架設 API Server，故若要使用 GPU，請在主機上安裝好 **nvidia-docker** ([ref](https://github.com/NVIDIA/nvidia-docker))，並按照以下指令操作(device 需看自己主機上的 GPU 編號)：
        ```bash
        sudo nvidia-docker build . -t mathlin/cudaoutofmemory
        sudo nvidia-docker run --gpus='"device=0,1"' -p 614:614 -d mathlin/cudaoutofmemory
        ```
        * 若需要查詢執行狀況，建議按以下方式操作：
        ```bash
        docker ps
        docker logs [CONTAINER_ID]
        ```
        * 先前 pre_trained 的模型是 **bin** 檔，而後 inference 訓練改用 state file，因此若您按照我們的方式訓練模型，而後讀取模型時就確保 load **parameter 一致**，且 `api.py` 讀取時請將對應的程式碼更改如下：
        ```python3
        # model = torch.load(model_path, map_location=device)
        model = GecTransformer(max_len=200,
                num_of_vocab=tokenizer.vocab_size,
                d_model=512,
		        nhead=8,
		        num_encoder_layers=6,
		        num_decoder_layers=6,
		        dim_feedforward=2048,
		        dropout=0.2,
		        activation="relu")
       model.load_state_dict(torch.load(model_path, map_location=device))
        ```
