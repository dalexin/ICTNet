# ------------------------------------------------------------------------------
# Written by XinZhang
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .backbone_base import get_resnet18, get_pidnet_s
from .module_base import ICFusion, DAPPM, PAPPM, MSContext, MLContext, Decoder, ICPG_pool, ICPG_conv, ICPG_pool_compare, ICPG_pool_more
import logging
from thop import profile

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class Backbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Backbone, self).__init__()

        if backbone == 'pidnet_s':
            pidnet_s = get_pidnet_s()
            self.enc_layer = nn.Sequential(pidnet_s.conv1, pidnet_s.layer1, pidnet_s.layer2)
            self.context_1 = pidnet_s.layer3
            self.context_2 = pidnet_s.layer4
            self.context_3 = pidnet_s.layer5
        
        elif backbone == 'resnet18':
            resnet18 = get_resnet18()
            self.enc_layer = nn.Sequential(resnet18.conv1, resnet18.bn1, nn.ReLU(inplace=True), resnet18.maxpool, resnet18.layer1)  # 64
            self.context_1 = resnet18.layer2
            self.context_2 = resnet18.layer3
            self.context_3 = resnet18.layer4

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))


    def forward(self, x):
        x = self.enc_layer(x)
        c1 = self.context_1(x)
        c2 = self.context_2(c1)
        c3 = self.context_3(c2)
        return x, c1, c2, c3



class ICTEDNet(nn.Module):

    def __init__(self, backbone_name='resnet18', class_num=19, augment=True):
        super(ICTEDNet, self).__init__()
        self.augment = augment

        self.backbone = Backbone(backbone_name)

        # considering
        self.ic_enc_1 = nn.Sequential(nn.Conv2d(64,64,3,1,1, groups=64),nn.BatchNorm2d(64), nn.ReLU()
                                )
        self.ic_enc_2 = nn.Sequential(nn.Conv2d(64,64,3,1,1, groups=64),nn.BatchNorm2d(64), nn.ReLU()
                                )
        self.ic_enc_3 = nn.Sequential(nn.Conv2d(64,1,3,1,1),nn.BatchNorm2d(1)
                                )
        self.ic_enc_4 = nn.Sequential(nn.Conv2d(1,1,3,1,1),nn.BatchNorm2d(1), nn.ReLU()
                                   )


        self.ic_fuse1 = ICFusion(64, 128)
        self.ic_fuse2 = ICFusion(64, 256)

        # self.icpg = ICPG_conv(256, 1)
        self.icpg = ICPG_pool_more(ic_num=2)

        self.ms_context = MSContext(in_channels = 512, out_channels= 128 , grids=(6, 3, 2, 1))
        self.ml_context = MLContext(low_channels = 128, high_channels = 256, out_channels = 256)
        
        # Prediction Head
        if self.augment:
            self.aux_decoder =  Decoder(in_chan=256, mid_chan=128, n_classes=class_num, up_scale=16)

        self.decoder = Decoder(in_chan=256, mid_chan=128, n_classes=class_num, up_scale=8)

        self.keras_init_weight()

        
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        
    # def keras_init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


    def forward(self, x):

        feat1, feat2, feat3, feat4 = self.backbone(x)

        feat_spic_1 = self.ic_enc_1(feat1)
        feat_spic_1 = self.ic_fuse1(feat_spic_1, feat2)

        feat_spic_2 = self.ic_enc_2(feat_spic_1)
        feat_spic_2 = self.ic_fuse2(feat_spic_2, feat3)

        ic_map = self.ic_enc_3(feat_spic_2)
        ic_feature = self.ic_enc_4(ic_map)

        feat_ms_context = self.ms_context(feat4)

        feat_ml_context = self.ml_context(feat_ms_context, feat3)

        feature_ic_refine = self.icpg(feat_ml_context, ic_feature)

        feature_out = self.decoder(feature_ic_refine)
        

        if self.augment: 
            aux_feature_out = self.aux_decoder(feat3)
            ic_map = F.sigmoid(F.interpolate(ic_map, scale_factor=8, mode = 'bilinear'))
            return [aux_feature_out, feature_out], ic_map
        else:
            return feature_out, ic_map

def get_seg_model(cfg, imgnet_pretrained):
    
    
    model = ICTEDNet(backbone_name=cfg.MODEL.BACKBONE, class_num=cfg.DATASET.NUM_CLASSES, augment=True)
    
    
    # if imgnet_pretrained:
    #     pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
    #     model_dict = model.state_dict()
    #     pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
    #     model_dict.update(pretrained_state)
    #     msg = 'Loaded {} parameters!'.format(len(pretrained_state))
    #     logging.info('Attention!!!')
    #     logging.info(msg)
    #     logging.info('Over!!!')
    #     model.load_state_dict(model_dict, strict = False)
    # else:
    #     pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
    #     if 'state_dict' in pretrained_dict:
    #         pretrained_dict = pretrained_dict['state_dict']
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    #     msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    #     logging.info('Attention!!!')
    #     logging.info(msg)
    #     logging.info('Over!!!')
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict = False)
    
    return model

def get_pred_model(backbone_name, num_classes):
    

    model = ICTEDNet(backbone_name=backbone_name, class_num=num_classes, augment=False)
   
    return model

if __name__ == '__main__':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    input = torch.randn(1, 3, 720, 960).cuda()

    device = torch.device('cuda')
    model = get_pred_model(backbone_name='pidnet_s', num_classes=11)
    model.eval()
    model.to(device)
    iterations = None

    flops, params = profile(model, inputs=(input,))
    print(f"GFLOPs: {flops / 1e9} G")
    print(f"Parameters: {params / 1e6} M")


    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    
    
    


