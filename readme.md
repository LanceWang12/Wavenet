# Wavenet for time series forecasting

## -Data flow-

1. PyTorch version![Screen Shot 2020-09-22 at 8.29.35 PM](/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.29.35 PM.png)

2. Keras version![Screen Shot 2020-09-22 at 8.30.34 PM](/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.30.34 PM.png)



## -File explaination-

### -- ./Keras/train_and_test.py --

1. def scheduler(epoch, lr):
    a. 用途:使 lr 隨著 epoch 慢慢變小

2. def train_val_plot(history, metric): a. 繪製 lr, loss, accuracy 下降曲線

3. def rolling_testing(model, model_tag, data, data_info, baseline, rolling_info):
   a. 用途:滾動測試

   b. 參數解釋
      model: 預測是之模型
      model_tag: 要測試的是 deep learning or machine learning('dl' or 'ml')
      data: 經過特徵工程的資料集

      -----------------------------------------------
      data_info: 為一字典，存放資料如下
      data_info = {
          'label': dataframe 中 label 的欄位名稱, 
          'date': dataframe 中 target_date 的欄位名稱,
          'start': 測試開始日期,
          'end': 測試結束日期
      }
      -----------------------------------------------

      -----------------------------------------------
      rolling_info: 為一字典，存放資料如下
      rolling_info = {
          'lag': 向前看幾天,
          'gap': 預測未來第幾天,
          'check_period': 檢查週期,
          'check_size': 計算準確率的資料數量,
          'sample': 是否使用 sample 的方式,
          'optimizer': Keras 的優化器(Ex. Adam),
          'metric': Keras 的評價方式(Ex. binary_accuracy),
          'scheduler': Scheduler,
          'early_stop': Early_stopping
      }
      -----------------------------------------------

   

   c. 回傳: Accuracy 之歷史紀錄與預測結果

4. gen_chi2_prob(n, field = (0, 10), k = 2):

   用途:產生 Chi-square 機率分佈

### --./Keras/dataset.py--

1. def generate_data(df, label_idx, date_idx, model_tag = 'dl', lag = 28, gap = 22, normalize_method = 'min-max', start = None, end = None):

   a. 參數解釋:
      df: 特徵工程完之資料
      label_idx: label 對應的欄位
      date_idx: 日期對應之欄位
      model_tag: ml or dl
      lag: 向前看幾天
      gap: 預測未來第幾天
      normalize_method: normalize 之方法
      start: 開始日期
      end: 結束日期

   

   b. 用途:生成 wavenet 所需的格式資料
    	Keras 所需格式:(batch_size, lag, feature_num) Pytorch 所需格式:(batch_size, feature_num, lag)

   c. 回傳:X, Y, 目前日期, 預測日期

### --./Keras/Wavenet_model.py--

1. class Wavenet
    a. 參數解釋:
       time_step: 向前看幾天
       feature_num: 一天的特徵數量(技術指標, 經濟指標等......)
       kernel_size: Conv1D 的 kernel 大小
       num_blocks: 需要有幾個 Wavenet block
       num_layers: 一個 Wavenet block 要有幾層 layer
       output_channel: 每個 Conv1D 的輸出 channel(務必大於 feature_num)
       reg: L2 regularization 之係數

2. 用途:
   構建 Wavenet 架構之類別

### --./Keras/Wavenet_unit.py--

1. class GatedConv1D

   a. 參數解釋:
      filters: 幾個 kernel
      kernel_size: kernel 寬度
      strides: 步伐大小(原始 Wavenet 架構皆為 1)

      padding: 必定為 'causal'(Valid: [1, 2, 3], Same: [0, 1, 2, 3, 0], Causal: [0, 0, 1, 2, 3])
      causal 補值只會在前面補， 因為不能擅自更動未來！

      dilation rate: 空洞大小
      use_bias: 是否需要 bias

   b. 用途:組合成 GatedConv1D 單元


   

2. class GatedResidualBlock 

   a. 參數解釋: 同上
   b. 組成含有 residual, skip 的單元架構

   <img src="/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.40.24 PM.png" alt="Screen Shot 2020-09-22 at 8.40.24 PM" style="zoom:67%;" />
