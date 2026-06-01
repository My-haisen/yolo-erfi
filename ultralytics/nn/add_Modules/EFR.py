import torch
import torch.nn as nn

# -----------------------------
# Conv + BN + Act
# -----------------------------
def autopad(k, p=None, d=1):
    # 自动计算padding
    if d > 1:
        k = d*(k-1)+1 if isinstance(k,int) else [d*(x-1)+1 for x in k]
    if p is None:
        p = k//2 if isinstance(k,int) else [x//2 for x in k]
    return p

class ConvBNAct(nn.Module):
    """卷积 + BN + 激活"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k,p,d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -----------------------------
# EFR-Block
# -----------------------------
class EFRBlock(nn.Module):
    """
    Efficient Feature Reallocation Block
    公式：
        F1 = Conv1x1(X)
        F2 = Conv1x1(F1)
        F3 = Conv3x3(F2)
        F4 = Conv1x1(F3)
        Fr = F4 + F1
        Y  = Conv1x1(Fr)
    """
    def __init__(self, c, hidden_ratio=0.5):
        """
        c: 输入输出通道数
        hidden_ratio: 中间卷积通道压缩比例
        """
        super().__init__()
        hidden_ch = max(1, int(c * hidden_ratio))
        
        self.conv1 = ConvBNAct(c, hidden_ch, k=1)
        self.conv2 = ConvBNAct(hidden_ch, hidden_ch, k=1)
        self.conv3 = ConvBNAct(hidden_ch, hidden_ch, k=3)
        self.conv4 = ConvBNAct(hidden_ch, c, k=1)
        self.out_conv = ConvBNAct(c, c, k=1)
    
    def forward(self, x):
        F1 = self.conv1(x)
        F2 = self.conv2(F1)
        F3 = self.conv3(F2)
        F4 = self.conv4(F3)
        Fr = F4 + F1 
        Y = self.out_conv(Fr)
        return Y


if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)
    block = EFRBlock(64)
    y = block(x)
    print(y.shape)  # torch.Size([1, 64, 128, 128])