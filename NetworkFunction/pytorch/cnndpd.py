from torch import nn


class CNNDPD(nn.Module):
    def __init__(self, in_c=3, num_classes=128, init_weights=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=(3, 3), stride=(1, 1),
                              padding=(3 // 2, 3 // 2))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                               padding=(3 // 2, 3 // 2))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),
                               padding=(3 // 2, 3 // 2))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.fc = nn.Linear(int(36864), num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.fc(x.flatten(1))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


