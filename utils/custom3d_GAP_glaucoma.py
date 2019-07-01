import torch.nn as nn

class Glaucoma3d(nn.Module):

    def __init__(self, num_classes = 2):

        super(Glaucoma3d, self).__init__()
        self.conv_layer1 = self.conv_layer_operations(1,  32, 3, 7, 7, 2)
        self.conv_layer2 = self.conv_layer_operations(32, 32, 2, 5, 5, 1)
        self.conv_layer3 = self.conv_layer_operations(32, 32, 2, 3, 3, 1)
        self.conv_layer4 = self.conv_layer_operations(32, 32, 2, 3, 3, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))# global average pooling layer useful for extracting class activation maps (CAMs) 
        self.fc = nn.Linear(32, num_classes)#single fully connected layer necessary to extract CAMs

    def conv_layer_operations(self, in_c, out_c, k1, k2, k3,s):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(k1, k2, k3), stride=s), #3d convolution with a kernel (k)size values for each dimension
        nn.ReLU(),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x