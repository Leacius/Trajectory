import torch.nn as nn

class permute(nn.Module):
    def __init__(self, shape=None):
        super(permute, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(self.shape)
    
class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(basic_block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, classes, sample_length=38):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 32),
            permute(shape=(0, 2, 1)),
            basic_block(32, 64, 3, stride=1, padding=1),
            permute(shape=(0, 2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128),
            permute(shape=(0, 2, 1)),
            basic_block(128, 256, 3, stride=1, padding=1),
            permute(shape=(0, 2, 1)),
        )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     permute(shape=(0, 2, 1)),
        #     basic_block(512, 1024, 3, stride=1, padding=1),
        #     permute(shape=(0, 2, 1)),
        # )
        # self.layer4 = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     permute(shape=(0, 2, 1)),
        #     basic_block(2048, 2048, 1, stride=1, padding=0),
        #     permute(shape=(0, 2, 1)),
        # )
        
        self.fc = nn.Linear(256, classes)

    def forward(self, x):
        
        # temporal feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = x.view(x.size(0), -1)
        x = x.mean(dim=1)
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    from torchviz import make_dot
    import torch

    model = MLP(10)
    a = torch.randn(16, 38, 3)
    y = model(a)

    make_dot(y, params=dict(list(model.named_parameters()))).render("test", format="svg")
    # print(list(model.parameters()))