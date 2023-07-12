# PyTorch_CNN
```
!pip install jupyterthemes
```
+ change the jupyter themes
```
!jt -l
```
+ initialize
```
!jt -h
```
+ help method
```
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        return self.conv(x)
```
+ Convolution
```
class MaxPool2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__()
        self.pool = nn.MaxPool2d(pool_size)
        
    def forward(self, x):
        return self.pool(x)
```
+ Maxpooling
```
class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.fc(x)
```
+ Fully Connect
```
class ReLU(nn.Module):
    def forward(self, x):
        return torch.max(x, torch.tensor(0.0))
```
+ ReLU
```
class Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)
```
+ SoftMax
