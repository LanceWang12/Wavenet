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

   <img src="/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.33.07 PM.png" alt="Screen Shot 2020-09-22 at 8.33.07 PM" style="zoom:70%;" />

   

   c. 回傳: Accuracy 之歷史紀錄與預測結果

4. gen_chi2_prob(n, field = (0, 10), k = 2):

   用途:產生 Chi-square 機率分佈

### --./Keras/dataset.py--

1. def generate_data(df, label_idx, date_idx, model_tag = 'dl', lag = 28, gap = 22, normalize_method = 'min-max', start = None, end = None):

   a. 參數解釋:

   <img src="/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.36.11 PM.png" alt="Screen Shot 2020-09-22 at 8.36.11 PM" style="zoom:80%;" />

   

   b. 用途:生成 wavenet 所需的格式資料
    	Keras 所需格式:(batch_size, lag, feature_num) Pytorch 所需格式:(batch_size, feature_num, lag)

   c. 回傳:X, Y, 目前日期, 預測日期

### --./Keras/Wavenet_model.py--

1. class Wavenet
    a. 參數解釋:

   ![Screen Shot 2020-09-22 at 8.37.52 PM](/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.37.52 PM.png)

   

2. 用途:
   構建 Wavenet 架構之類別

### --./Keras/Wavenet_unit.py--

1. class GatedConv1D

   a. 參數解釋:![Screen Shot 2020-09-22 at 8.38.52 PM](/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.38.52 PM.png)

   

   b. 用途:組合成 GatedConv1D 單元

   <img src="/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.39.38 PM.png" alt="Screen Shot 2020-09-22 at 8.39.38 PM" style="zoom:67%;" />

   

2. class GatedResidualBlock 

   a. 參數解釋: 同上
   b. 組成含有 residual, skip 的單元架構

   <img src="/Users/wangruicheng/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 8.40.24 PM.png" alt="Screen Shot 2020-09-22 at 8.40.24 PM" style="zoom:67%;" />