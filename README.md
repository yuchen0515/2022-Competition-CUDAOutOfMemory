# 【看見你的聲音 - 語音辨識後修正】API Server 架構


## （一）簡介
本文件為「**最佳 API 服務獎**」之說明文件，參加玉山銀行舉辦之「看見你的聲音—語音辨識後修正」競賽，由林育辰、梁俊彥組成的 **CUDAOutOfMemory**，前者負責 API Server、後者則主要負責 AI model。

API Server 以 [JMeter](https://jmeter.apache.org/index.html) 作為壓力測試工具評測，在以下條件下進行：
- 同時有 **10** 個 clients
- 每個 clients 皆輪流發送 10 個不同的 request
- 每個 clients 收到 response 後才傳送下個 request

最終在上述條件，以 Trasformer 模型達到 **14.5 / sec (Throughput)** 的水準，大多數的 Request 能在 **1** 秒內回傳推論結果。

### 附註：機器規格
本隊曾考慮過 GCP, AWS 等雲端服務平台，然因費用及其他因素，決定均使用校內 **國立臺灣大學 多媒體資訊檢索實驗室** 的主機設備：
|| 型號 / 版本 |Note|
|:--|:--|:--|
|CPU| Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz|20 cores|
|GPU| GeForce GTX 1080 Ti| 2 cores|
|RAM| 128 GB||
|OS| Ubuntu 21.04  (Hirsute Hippo)||

本文件之所有結果與成效，**均**在使用 GPU (GeForce GTX 1080 Ti) 情況下達成。

## （二）緣起
起初，我們著重於「**模型精準性**」上，並決定以重量級模型 Transformer 實行，並不斷優化模型的精確度，這也使得推論時間越拉越長，僅作一次 inference 就需高達 2300 ms 的時間，也因此開始不斷精進與研究提高 API 服務效能的方法。


## （三）問題解決與目標
### 解決之道之一
每筆高達 2300 ms 的推論時間，起初思考：

- TOP10 中，真的有必要每個都兼顧嗎？
- model level 是否應提升，亦可避免 overfitting？

經過反覆嘗試，前三語句修正、前二語句修正, ...，甚至使用 RNN 模型輔助，但其成效都遜於 baseline。最後**只針對 TOP1 糾錯**才是結果最佳的，除此以外也將 **level 調降**，使得每筆 inference 降低到 700 ms。

### 解決之道之二
> 注：後續主辦方更改為 **2** 秒時限

然而，仔細觀覽賽事要求：
- 最多需**同時**處理 10 筆 requests
- 每筆 request 需於 1 秒內回傳

隨著同時處理的 request 增加，所需時間也會**線性**、甚至指數成長。一旦超時，該筆 request 還會再度回傳兩次，容易產生連鎖反應，使得整體服務壅塞而全盤崩壞。

因此，接著我有幾個目標想完成：
- **Concurrent**, API 服務需能異步處理
- **non-timeout**, 無法處理的就隨它去吧！
- **GPU** speed up

#### Concurrent
考慮 WSGI, Nginx 等正式環境部署工具，然實測結果比 flask 內建的 Thread 還慢 30% 左右，因此最後以 flask 內建之 Thread 達到 concurrent。

#### non-timeout
timeout 對於本次競賽損傷是最嚴重的，直接沒有分數。因此對於無法計算、超出本 API 服務能力所及的，直接回傳 TOP1 獲得 baseline 是最實際的。

承襲 concurrent 提到的，flask thread 應用於正式環境部署，對於例外處理較弱，應替接受資料先行篩選，例如：
1. 編碼總長度超過 **45** 者不予計算，直接給 TOP1，反之則經過 inference 推算。
2. retry 的語句就直接**不**予糾正，避免陷入惡性循環。

原因為 transformer 是**字與字間**的推算，**字的長度也影響著推論速度**，一旦遭惡意傳送很長的字串，將會導致服務停滯。


#### GPU Speed up
善用實驗室主機的 GPU，然而使用 docker 要跑 GPU 的話有很多繁瑣的前置作業，因此使用 **nvidia-docker** 架設服務最為實際。


概括而論，引用 GPU 為我們提高 **40%** 的效能 (800 ms --> 480 ms)，non-timeout 選擇性糾正使我們得以控制能夠在 主機能力所及範圍內**提供滿足競賽需求之服務**。Concurrent 使我們能同時承受十個 request 傳送之。

然而，這樣糾正編碼總長度為 25 的語句已是極限，如下圖即為長度 25 以下才糾正的 JMeter 壓力測試結果：
![](https://i.imgur.com/x4OPJZl.png)

可以注意到部分的 request 甚至**超過 2 秒**的限制，因此這樣的效能對於我們而言尚且不足，故有了下個目標：*Batch* 技術的引入。

### 解決之道之三 —— Batch 批次處理
先提結論：本方法之導入，為我從 throughput 5.7 / sec 提升至 **14.5** / sec，**效率為原本的 254 % 之多**。

我認為 API 在接收 request 時，可以給定一個 "延遲時間"，去累積 request，等時間一到 或 超過某一數量限制，就**批次處理**，如此還可利用 GPU 的平行處理的效能。

因此，找到了 **[ShannonAI/service-streamer](https://github.com/ShannonAI/service-streamer)** 這套工具，能達到這個需求，實測後 Response time graph 如下：
![](https://i.imgur.com/hLxn4eJ.png)

可注意到多數 request 皆能在遠超乎主辦方之條件，於 1 秒內回傳結果，而下圖更可看出 throughput 穩定的程度：
![](https://i.imgur.com/DHux2sQ.png)

以下則為關於 request 的各項處理數據：
|Label|Samples|Average|Min|Max|Error|Thorughput|
|:--|:--|:--|:--|:--|:--|:--|
|HTTP Request|1017|668|263|1248|0.00%|14.5/sec|




## （四）系統架構
在前三節次中，我們從競賽開始談及，以及接連遇到問題的應對與制定對策，一步步改善我們的 API Server，而在 **系統架構** 將清楚敘述「整體架構」以及 本 API Server 優勢。

### API Server 架構圖
![](https://i.imgur.com/KosszE5.png)
1. **[POST]** \
由主辦方傳送 Requests 至 Server。
2. **[Thread]** \
Flask server 為**每**筆 request 建立一個 thread 處理之。
3. **[Queue]** \
 延遲 **150 ms** 以收集 request，收集到 64 筆 或超出時間就批次處理。
4. **[Batch & model]** \
同時批次處理每個 request，每個 request 都檢查是否「有誤」，確認後依序前處理、推論 (AI model)、檢查後得到結果。
5. **[Return]** \
計算出結果後，將結果給予對應的 Thread。
6. **[Response]** \
由 Thread 回傳給對應的 Request。


### 優點
本架構以 Docker 架置，並且輔以 batch 和例外處理等方式，目標是希望達到：
1. **便於建置** —— 僅需安裝 Docker，即可輕鬆建置 API Server。
2. **回應速率一致** —— 批次處理技術，使得同時傳一個、兩個或多個 request 時，每筆 request 花費時間會接近。這能讓使用者不會有服務「忽快忽慢」不穩定的感覺。
3. **例外處理** —— 避免過長或失敗的 request 使服務壅塞。


## （五）應用技術

本隊 API Server 應用技術之概述：
- **[pre-load]** \
API Server 啟動時**預**先載入 pre-trained model。
- **[batch]** \
批次處理，使得效率大幅提升，並使所有 request 回應時間接近。
- **[例外處理]** \
**避免**異常的 request 使服務壅塞而中斷。
- **[concurrent]** \
使用異步處理，使得其能平行化增進效能，避免 sequential 形式一來一往使效率低落。
- **[cache]** \
docker 建置時運用 cache 機制，使得 **re-build** 時建置速度從 15 分鐘 降低至 5 分鐘。
- **[CUDA]** \
使用 nvidia-docker 工具，使得 docker 能同時搭配 GPU，以提高效能。
- **[測試]**
    - [API 測試] 使用 [postman](https://www.postman.com/) 檢測 API 運作、推論結果
    - [異步測試] 使用 Python 套件 [grequests](https://pypi.org/project/grequests/) 檢測同時發送多筆 request 所需時間
    - [壓力測試] 使用 [JMeter](https://jmeter.apache.org/) 作為正式賽事規格乘載力之檢測方式
 
 
## （六）程式架構維運技巧
1. **[Docker]**\
 選擇使用 Docker 建置，除了易於建置外，Docker 容器於運行時皆會留下專屬於該 [CONTAINER ID] 的 log 檔案，方便追蹤 api 回應狀況。
2. **[ArgumentParser]**\
 能彈性指定輸入的參數，便於調整與測試。
3. **[Module]**\
 檔案執行形式以 `python3 -m ooxx` 的模組化形式為主，輔以架構化的檔案結構，和資料夾專屬之 `__init__.py`，便於維護不同功能之程式碼。
4. **[PEP8]**\
 遵循 Python **[PEP8](https://peps.python.org/pep-0008/)** code style guideline，維持全體程式碼風格一致，提升可讀與可維護性。


## （七）手把手之使用教學
### (1) 打包

1. 下載本專案，並且載入 submodule (其他開源專案程式碼)，切換到 `api-spec` 分支。
```
git clone git@github.com:yuchen0515/2022-Competition-CUDAOutOfMemory.git
cd 2022-Competition-CUDAOutOfMemory
git submodule init
git submodule update
git checkout api-spec
```
2. 下載模型檔案
- `ngram_lm`: 下載[資料夾](https://drive.google.com/drive/folders/196FIAexcXiARKcU2NdkRAMpphOxDeajF?usp=sharing)後，放置專案主目錄
- `trained_model`：下載 [檔案](https://drive.google.com/drive/folders/1SWbF1ZW-H3NIk-W8Uvi2cr72MRzZDATk?usp=sharing)，並在專案主目錄建立資料夾 `trained_model`，並將該檔案放在裡面。
- (optional) `data_preprocess/train_all.json`：此為主辦方提供之訓練資料，架設 API 並不需要，若有 AI model 推論一系列需求的話，請自行準備檔案放置相對應之位置。
 
### (2) Docker
本隊於正式賽時，Docker 建置使用到 GPU，因此使用 nvidia-docker 建置。您可以評估自己是否要使用 GPU，再選擇下列對應的方式架設 API。

#### 不使用 GPU
請安裝 docker，安裝方式可參考 [tutorial](https://docs.docker.com/engine/install/)，在此同樣附上 Ubuntu 的 Docker 安裝方式：
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

接著，神奇的時間到了，按照下列指定建立 Docker image 並啟動他：
```
sudo docker build . -t mathlin/cudaoutofmemory
sudo docker run -p 614:614 -d mathlin/cudaoutofmemory
```

當然，上述 image 名稱可以自行更換，第一行指令需待 Docker 環境建置，約需 10~15 分鐘。而我們使用了 **cache** 機制，因此您將 image 刪掉後立即重建，能縮減至 5 分鐘建置完成。

第二行執行後，即是指定 image 運行，此時會建立 "container" 運行之，為避免途中錯誤滋生以致服務中斷，因此要確認程序是否建立成功：

```
sudo docker ps
```

如下圖：
![](https://i.imgur.com/oAu1VZz.png)

接著在以 postman 測試之：
![](https://i.imgur.com/8MzeHoL.png)


倘若發現 API Server 無法正確回應要求，請先確認 flask 是否已經運行，因此可查詢 Docker log，以上圖為例：

```
sudo docker logs f4125f2ff3f2
```

若能看到 `listening ooxx:614/`，那就已經完成了，反之若甚至沒看到 Container id，代表出現錯誤而自行中斷，這時請往上翻第二行指令完成時輸出的一串 id，那就是 container id，可以找到具體的錯誤訊息。


#### 使用 GPU
請安裝 nvidia-docker，而在安裝之前還必須確認：
- 電腦是否有 GPU？
- CUDA 是否有安裝？
- 驅動(Driver)程式是否安裝？

安裝過程可參考 [Medium - Ubuntu 安裝 CUDA cuDNN pytorch tensorflow mxnet by 林塔恩](https://mikethreeacer.medium.com/ubuntu-18-04-%E5%AE%89%E8%A3%9D-cuda-cudnn-anaconda-pytorch-1f170b3326a4)

docker 想架設有 GPU 的環境相對繁雜許多，請先上網查詢並確認安裝完畢後，確認 `nvidia-docker` 指令是否能使用，若能使用，即可按照以下方式架設 API Server：
```
sudo nvidia-docker build . -t mathlin/cudaoutofmemory
sudo nvidia-docker run --gpus='"device=0,1"' -p 614:614 -d mathlin/cudaoutofmemory
```

同樣地，上述 images 命名可自行更換。格外要注意的是 gpu 設備編號請看自己電腦對應的編號更改之。而若建置上需要排查，請參考上方 **[不使用 GPU]** 篇，只需將指令 `docker` 改為 `nvidia-docker` 就可使用。

### (3) 小測試
本隊測試時有使用三種工具：
- JMeter - 壓力測試
- Postman - API 基本測試
- grequests - 異步處理測試

前兩者是非常重要與好用的測試方式，網路上有相當詳細的教學步驟，可自行查閱。而這邊想簡單介紹 grequests 如何簡單測試 API Server 異步處理之耗時。

首先，`testing/example_concurrentTesting.py` 是範例程式檔，可編輯該檔案的 URL 為你的 API Server 位置，接著：
```
time python3 -m testing/example_concurrentTesting.py
```

即可看到在同時發送十筆 requests 時，API Server 回傳總耗時為何。而你可從 User, System 以及 total 彼此間的花費時間關係看出系統是 I/O bound，還是 CPU bound。

若要做其他的測試，可參考這份 python 檔更改，或著也可搜尋到更多 grequests 的使用方式。


## （八）未來展望
本隊 CUDAOutOfMemory 在正式賽期間第三天，主機於早上發生意外，因而在第三、第四天無法正常提供服務，因此未來有機會的話，希望能盡量將 API 服務「雲端化」，以避免意外狀況，不需用到那麼多較高的後端技術成本，就可以得到遠遠勝出的效能。

然倘若考量成本，以及那些意外大多可避免的話，未來選擇自行架設主機有以下項目是未來能夠發展與改進的方向。


### 效能提升
- **[nvidia-docker]**\
 根據網路上文章指出，nvidia-docker 在運行過程中會不斷查詢網路上特定文件，以至於效率拖沓，因此若能 code tracing 抓出執行一系列步驟，將那些文件進到 local 端維護並定期更新，能提高約 30% 之效能。
- **[multi-GPUs]**\
 Transformer 模型要以 multi-Worker 形式執行有一定的困難。而當時發現能以 parallel data pipeline 的方式，拆分 transformer encoder, decoder 等步驟，並於不同 GPU 間同步分開執行。但此法是最困難與麻煩的，可考慮未來嘗試之。
- **[Redis]**\
 若服務有超時狀況發生，將會浪費先前的運算時間。因此可利用 Redis 將計算到一半的結果對應於 pid (任務代號)，遇到相同任務代號持續計算，以節省不必要之浪費、提高效率。此外，[service-streamer](https://github.com/ShannonAI/service-streamer) 亦有針對 Redis 的 batch 加速技術。

 
### 程式碼維護
- **[Config]**\
 在本專案中，大多資料夾皆配置有 `Config.py`，然而當時因 `api.py` 與各模型推論資料夾設定相互衝突，故並未設定 api 的 config，未來可處理這部分，能有效屏蔽敏感內容、消弭雜亂的程式碼片段，並提高可維護性。
- **[ArgumentParser]**\
 目前 ArgumentParser 參數寫的不夠完全，常常使用時仍有些麻煩，這點需要改進。
- **[Coding style]**\
 現今本專案遵循 PEP8 形式，然而在類別命名等方面尚無一致性標準，也並沒有善用 class 的結構寫出 "clean code"。較多程式碼片段顯得拖沓稍不嚴謹些，未來可在這方面持續努力。
- **[Git]**\
 專案因使用實驗室主機進行，而在 AI model 方面以 jupyter 較為方便。不過這也導致了 負責 model 的人員與 API Server 的人員難以一致性的使用 git 控管之。而若可以改善這問題，並按功能繳交 commit，遵循一般的 git flow，整體維護上會更為簡單明瞭。
 
### 穩定提供服務
- **[Celery]**\
  Celery 是 Python 工作排程的套件，搭配此可以更有效管理 request 的「進出型態」，此外能強迫逾時的 request 不再計算，先將結果儲存至 redis。也能避免過於異常的 request 把整個 API 伺服器搞爛的狀況，但因相對麻煩，在競賽中並未使用。
- **[Exception]**\
  配合上述 Celery 套件，我們在正式賽時例外處理並不夠周全，只是因為效能改善比預期要好太多，對於競賽需求綽綽有餘，才沒有遇到狀況。在異常資料或壅塞狀況時的例外處理之完善顯得是未來必須要改善的項目。
- **[Nginx]**\
  正式賽有試過 Nginx, WSGI 等正式部署 API 的工具，雖在例外處理上格外優秀，例如：JMeter 測試戛然停止時，末段的 request 並不會跳錯誤。\
  但實際評測過三者 (包含 flask Thread) 之效能後，發現 flask thread 效能高出不少，為了效能才鋌而走險使用。未來可研究「正式」部署 API 服務的工具，提高穩定性，也需要找出如何提高正式部署工具的效能。
- **[Request]**\
 目前是針對每筆 request 都開 Thread，但遇到 **過多** 的狀況是很容易出事的，因此配合上述 Nginx 正式部署工具之機制，可改為架設 5~10 台 虛擬的伺服器 server，並作 balance，才能穩定提供服務。
 
 
## （九）心得

因五月中才加入競賽，以致事前評估不夠周全。綜觀本次競賽之資料型態與目標較為適合使用 Soft-Masked BERT，效果會較為穩定，也有望取得更佳的成效。

此外，NLP 大多屬於「重量級」的模型，故單筆處理時間稍長。而當時並未特別重視後端要求相當高，而後發現時已經是「非救不可」的程度，勢必得在後端上改善數倍以上的效率才可以通過競賽的檢核。雖自身對於後端有一定掌握度，但並沒把握大幅改善 AI model 形式的 API 服務。

而競賽於伺服器架設上並無特別限制或資源，倘若選擇雲端服務，則需綁定信用卡，多數學生並無信用卡，倘若有也很擔心會不慎操作滋生大筆費用。而當然，若使用雲端服務架設，後端伺服器設定就非常簡單，只需要簡單設定就可以達到 balance，也能有很優異的效能處理速度。

選擇架設在自身電腦、實驗室主機等等，與雲端架設方式相比，後端技術力所需相差相當大，這也使得我們即便有不差的後端技術力，效率仍不及架設於雲端平台的伺服器，因而無法讓模型做過多複雜的運算。

藉由本次競賽，充分了解到後端人員之於一個正式服務有多麽重要，其技術能力也遠遠左右了服務的效能，以我們的模型為例，起始每筆只有 2300 ms 的效能，若有技術純熟的後端工程師，則可以使其承受同時十筆 request，每筆更能在一秒之內回傳。

在本次競賽中透過實戰學習到許多後端技術，以及伺服器如何最大彈性的與 AI model 互動，也是參與競賽最令人著迷之處！
