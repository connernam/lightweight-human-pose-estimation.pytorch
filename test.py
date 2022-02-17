import torch
from torchsummary import summary
from models.with_mobilenetV3_sh import mobilenet_v3_small, mobilenet_v3_large
from models.with_mobilenet import PoseEstimationWithMobileNet

if __name__ == '__main__':
    net = mobilenet_v3_small()
    # net = PoseEstimationWithMobileNet(1)
    summary(net, input_size=(3, 368, 368), device='cpu')  # 12 12 -> 46 46
    # print(net)