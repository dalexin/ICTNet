import torchvision.models as models
from models.pidnet import PIDNet


def get_resnet18():

    model = models.resnet18(pretrained=False)

    return model

def get_pidnet_s():

    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)

    return model