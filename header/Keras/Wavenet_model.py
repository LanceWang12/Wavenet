from header.Keras.Wavenet_unit import GatedResidualBlock
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import BatchNormalization, Activation, \
                                    Dense, Dropout, Flatten, Conv1D, Add


# --------------Warning---------------
# input_shape = (batch_size, time_step, feature_num)

class Wavenet(Model):
    def __init__(self, time_step, feature_num, kernel_size,
                 num_blocks, num_layers, output_channel, reg = 1e-2):
        # Parameter
        # time_step: 向前看幾天
        # feature_num: 一天的特徵數量(技術指標, 經濟指標等......)
        # kernel_size: Conv1D 的 kernel 大小
        # num_blocks: 需要有幾個 Wavenet block
        # num_layers: 一個 Wavenet block 要有幾層 layer
        # output_channel: 每個 Conv1D 的輸出 channel(務必大於 feature_num)
        # reg: L2 regularization 之係數

        super(Wavenet, self).__init__()
        self.reg = reg
        self.receptive_field = 1 + (kernel_size - 1) * \
                               num_blocks * sum([2 ** k for k in range(num_layers)])
        self.output_width = time_step - self.receptive_field + 1


        hs = []
        batch_norms = []

        # add gated convs
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                h = GatedResidualBlock(filters = output_channel, kernel_size = kernel_size,
                                       padding = 'causal', dilation_rate =rate)
                hs.append(h)
                batch_norms.append(BatchNormalization(trainable = False))

        self.hs = hs
        self.batch_norms = batch_norms

        self.relu = Activation('relu')
        self.conv_1_1 = Conv1D(filters = output_channel, kernel_size = 1, 
                               kernel_regularizer = regularizers.l2(self.reg),
                               bias_regularizer = regularizers.l2(self.reg))
        self.conv_1_2 = Conv1D(filters = 1, kernel_size = 2,
                               kernel_regularizer = regularizers.l2(self.reg),
                               bias_regularizer = regularizers.l2(self.reg))
        self.dropout = Dropout(0.4)
        self.flatten = Flatten()
        self.dense_1 = Dense(128, kernel_regularizer = regularizers.l2(self.reg),
                                  bias_regularizer = regularizers.l2(self.reg))
        self.dense_2 = Dense(1, kernel_regularizer = regularizers.l2(self.reg),
                                bias_regularizer = regularizers.l2(self.reg))
        self.sig = Activation('sigmoid')

    def call(self, x, training = False):
        skips = []
        for layer, batch_norm in zip(self.hs, self.batch_norms):
        # for layer in self.hs:
            x, skip = layer(x)
            x = batch_norm(x)
            skips.append(skip)

        x = Add()(skips)
        x = self.relu(self.conv_1_1(x))
        x = self.dropout(x)
        x = self.relu(self.conv_1_2(x))
        x = self.flatten(x)

        if training:
            x = self.dropout(x)
        
        x = self.dense_1(x)
        x = self.relu(x)
        
        if training:
            x = self.dropout(x)
        
        x = self.dense_2(x)
        x = self.relu(x)
        
        if training:
            x = self.dropout(x)
        
        x = self.sig(x)
        
        return x