import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import Layer, Conv1D, Add, Multiply

class GatedConv1D(Layer):
    def __init__(self, filters, kernel_size, strides = 1, padding = 'causal', 
                 dilation_rate = 1, use_bias = True):
        # Parameter
        # filters: 幾個 kernel
        # kernel_size: kernel 寬度
        # strides: 步伐大小(原始 Wavenet 架構皆為 1)

        # padding: 必定為 'causal'(Valid: [1, 2, 3], Same: [0, 1, 2, 3, 0], Causal: [0, 0, 1, 2, 3])
        # causal 補值只會在前面補， 因為不能擅自更動未來！

        # dilation rate: 空洞大小
        # use_bias: 是否需要 bias
        
        super(GatedConv1D, self).__init__()
        self.dilation = dilation_rate
        self.tanh_out = Conv1D(filters = filters, kernel_size = kernel_size, 
                             strides = strides, padding = padding, dilation_rate = dilation_rate,
                             use_bias = use_bias, activation = 'tanh')
        self.sig_out = Conv1D(filters = filters, kernel_size = kernel_size, 
                             strides = strides, padding = padding, dilation_rate = dilation_rate,
                             use_bias = use_bias, activation = 'sigmoid')

    def call(self, x):
        return Multiply()([self.tanh_out(x), self.sig_out(x)])

class GatedResidualBlock(Layer):
    def __init__(self, filters, kernel_size, strides = 1, padding = 'causal', 
                 dilation_rate = 1, use_bias = True):
        # filters: kernel 數量
        # kernel_size: kernel 寬度
        # strides: 步伐大小(原始 Wavenet 架構皆為 1)
        # padding: 必定為 'causal'(Valid: [1, 2, 3], Same: [0, 1, 2, 3, 0], Causal: [0, 0, 1, 2, 3])
        # dilation rate: 空洞大小
        # use_bias: 是否需要 bias

        super(GatedResidualBlock, self).__init__()
        self.output_width = filters
        self.gatedconv = GatedConv1D(filters = filters, kernel_size = kernel_size,
                                     strides = 1, padding = 'causal', 
                                     dilation_rate = dilation_rate, use_bias = True)
        self.conv_1 = Conv1D(filters = filters, kernel_size = 1, strides=1, padding = 'causal',
                             dilation_rate = 1, use_bias = True)

    def call(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = Add()([x, skip])
        return residual, skip

