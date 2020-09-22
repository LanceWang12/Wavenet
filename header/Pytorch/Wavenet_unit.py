import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        padding = self.dilation - (x.shape[-1] + self.dilation - 1) % self.dilation
        x = nn.functional.pad(x, (self.dilation, 0))
        return torch.mul(self.tanh(self.conv_f(x)), self.sig(self.conv_g(x)))

class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_width, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        self.output_width = output_width
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)

        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)
        return residual, skip

class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, y):
        x = x.narrow(-1, y.shape[-1], x.shape[-1] - y.shape[-1])
        dims = [1] * len(x.shape)
        dims[-1] = y.shape[-1]
        y = y.reshape(dims)
        return torch.cat([x, self.dataset._to_tensor(y)], -1)

    def tensor2numpy(self, x):
        return x.data.numpy()

    def predict(self, x):
        x = x.to(self.model.device)
        self.model.to(self.model.device)
        return self.model(x)

    def run(self, x, num_samples, disp_interval=None):
        x = self.dataset._to_tensor(self.dataset.preprocess(x))
        x = torch.unsqueeze(x, 0)

        y_len = self.dataset.y_len
        out = np.zeros((num_samples // y_len + 1) * y_len)
        n_predicted = 0
        for i in range(num_samples // y_len + 1):
            if disp_interval is not None and i % disp_interval == 0:
                print('Sample {} / {}'.format(i * y_len, num_samples))

            y_i = self.tensor2numpy(self.predict(x).cpu())
            y_i = self.dataset.label2value(y_i.argmax(axis=1))[0]
            y_decoded = self.dataset.encoder.decode(y_i)

            out[n_predicted:n_predicted + len(y_decoded)] = y_decoded
            n_predicted += len(y_decoded)

            # shift sequence and insert generated value
            x = self._shift_insert(x, y_i)

        return out[0:num_samples]