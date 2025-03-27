import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sanet import SCE, SFF, BiSeNetOutput
from models.bricks import BuildNormalization, BuildActivation, DepthwiseSeparableConv2d

algc = False

class ICFusion(nn.Module):
    def __init__(self, inplane_feat, inplane_icmp, kernel_size=64):
        super(ICFusion,self).__init__()

        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)

        # self.bottleneck1 = nn.Conv2d(inplane,1,3,1,1)
        # self.bottleneck2 = nn.Conv2d(inplane,1,3,1,1)
        # self.bottleneck3 = nn.Conv2d(inplane,1,3,1,1)

        self.bottleneck1 = nn.Sequential(nn.Conv2d(inplane_feat,1,3,1,1),nn.BatchNorm2d(1), nn.PReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(inplane_icmp,1,3,1,1),nn.BatchNorm2d(1), nn.PReLU())
        self.bottleneck3 = nn.Sequential(nn.Conv2d(2,inplane_feat,3,1,1),nn.BatchNorm2d(inplane_feat), nn.PReLU())

        self.feat_conv_row = nn.Conv2d(1,1,(self.kernel_size,1),1,0, groups=1)
        self.feat_conv_col = nn.Conv2d(1,1,(1,self.kernel_size),1,0, groups=1)
        self.icmp_conv_row = nn.Conv2d(1,1,(self.kernel_size,1),1,0, groups=1)
        self.icmp_conv_col = nn.Conv2d(1,1,(1,self.kernel_size),1,0, groups=1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat_in, icmp_in):

        ori_h, ori_w = feat_in.size()[-2], feat_in.size()[-1]
        feat = F.interpolate(feat_in, size=[self.kernel_size, self.kernel_size], mode='bilinear')
        icmp = F.interpolate(icmp_in, size=[self.kernel_size, self.kernel_size], mode='bilinear')

        feat = self.bottleneck1(feat)
        if icmp_in.size(-3) != 1:
            icmp = self.bottleneck2(icmp)
        

        f_rv = self.feat_conv_row(feat)
        f_cv = self.feat_conv_col(feat)
        i_rv = self.icmp_conv_row(icmp)
        i_cv = self.icmp_conv_col(icmp)

        cross_map1 = torch.matmul(f_rv, i_cv)
        cross_map2 = torch.matmul(i_rv, f_cv)

        # cross_map1 = torch.matmul(f_cv, i_rv)
        # cross_map2 = torch.matmul(i_cv, f_rv)

        cross_map = cross_map1 + cross_map2
        cross_weight = self.sigmoid(cross_map)
        # print(cross_weight.size())

        weighted_feat = torch.cat((feat, icmp),dim=1) * cross_weight

        feature_out = self.bottleneck3(weighted_feat) 

        feature_out = F.interpolate(feature_out, size=[ori_h, ori_w], mode='bilinear') + feat_in

        return feature_out
    
class ICFusion_map(nn.Module):
    def __init__(self, inplane_feat, inplane_icmp):
        super(ICFusion_map,self).__init__()

        self.kernel_size = 64

        self.relu = nn.ReLU(inplace=True)

        # self.bottleneck1 = nn.Conv2d(inplane,1,3,1,1)
        # self.bottleneck2 = nn.Conv2d(inplane,1,3,1,1)
        # self.bottleneck3 = nn.Conv2d(inplane,1,3,1,1)

        self.bottleneck1 = nn.Sequential(nn.Conv2d(inplane_feat,1,3,1,1),nn.BatchNorm2d(1), nn.PReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(inplane_icmp,1,3,1,1),nn.BatchNorm2d(1), nn.PReLU())
        self.bottleneck3 = nn.Sequential(nn.Conv2d(2,inplane_feat,3,1,1),nn.BatchNorm2d(inplane_feat), nn.PReLU())

        self.feat_conv_row = nn.Conv2d(1,1,(self.kernel_size,1),1,0, groups=1)
        self.feat_conv_col = nn.Conv2d(1,1,(1,self.kernel_size),1,0, groups=1)
        self.icmp_conv_row = nn.Conv2d(1,1,(self.kernel_size,1),1,0, groups=1)
        self.icmp_conv_col = nn.Conv2d(1,1,(1,self.kernel_size),1,0, groups=1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat_in, icmp_in):

        ori_h, ori_w = feat_in.size()[-2], feat_in.size()[-1]
        feat = F.interpolate(feat_in, size=[self.kernel_size, self.kernel_size], mode='bilinear')
        icmp = F.interpolate(icmp_in, size=[self.kernel_size, self.kernel_size], mode='bilinear')

        feat = self.bottleneck1(feat)
        if icmp_in.size(-3) != 1:
            icmp = self.bottleneck2(icmp)
        

        f_rv = self.feat_conv_row(feat)
        f_cv = self.feat_conv_col(feat)
        i_rv = self.icmp_conv_row(icmp)
        i_cv = self.icmp_conv_col(icmp)

        # cross_map1 = torch.matmul(f_rv, i_cv)
        # cross_map2 = torch.matmul(i_rv, f_cv)

        cross_map1 = torch.matmul(f_cv, i_rv)
        cross_map2 = torch.matmul(i_cv, f_rv)

        cross_map = cross_map1 + cross_map2
        cross_weight = self.sigmoid(cross_map)
        # print(cross_weight.size())

        weighted_feat = torch.cat((feat, icmp),dim=1) * cross_weight

        feature_out = self.bottleneck3(weighted_feat) 

        feature_out = F.interpolate(feature_out, size=[ori_h, ori_w], mode='bilinear') + feat_in

        return feature_out

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        input_size = x.size()
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        
        return x

# class ICPG(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(ICPG, self).__init__()

#         # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = (kernel_size - 1) // 2

#         self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x,ic):
#         x = F.interpolate(x, size=ic.size()[-2:], mode='bilinear')
#         feat_avg_out = torch.mean(x, dim=1, keepdim=True)
#         feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
#         ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
#         ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
#         x_fuse = torch.cat([feat_avg_out, feat_max_out,ic_avg_out, ic_max_out], dim=1)
#         attn = self.sigmoid(self.conv1(x_fuse))
#         out = attn*x + (1-attn)*ic
#         return out


# class ICPG(nn.Module):
#     def __init__(self, feat_channel=128, ic_channel=64, kernel_size=3, BatchNorm=nn.BatchNorm2d):
#         super(ICPG, self).__init__()

#         # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = (kernel_size - 1) // 2

#         self.conv1 = nn.Sequential(nn.Conv2d(feat_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))
#         self.conv2 = nn.Sequential(nn.Conv2d(ic_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

#         self.conv3 = nn.Sequential(nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

#         self.conv4 = nn.Sequential(nn.Conv2d(feat_channel, ic_channel, kernel_size, padding=padding, bias=False), BatchNorm(ic_channel))

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, ic):

#         ic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')

#         context = self.conv1(x)
#         spic = self.conv2(ic)

        
#         attn = self.sigmoid(self.conv3(torch.cat([context, spic, context*spic], dim=1)))

#         # x = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
#         x = self.conv4(x)


#         out = attn*x + (1-attn)*ic

#         # x = F.interpolate(x, size=ic.size()[-2:], mode='bilinear')
#         # feat_avg_out = torch.mean(x, dim=1, keepdim=True)
#         # feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
#         # ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
#         # x_fuse = torch.cat([feat_avg_out, feat_max_out,ic_avg_out, ic_max_out], dim=1)
#         # attn = self.sigmoid(self.conv1(x_fuse))
#         # out = attn*x + (1-attn)*ic
#         return out

class ICPG(nn.Module):
    def __init__(self, feat_channel=128, ic_channel=64, kernel_size=3, BatchNorm=nn.BatchNorm2d):
        super(ICPG, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(feat_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))
        self.conv2 = nn.Sequential(nn.Conv2d(ic_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

        self.conv3 = nn.Sequential(nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

        self.conv4 = nn.Sequential(nn.Conv2d(ic_channel, feat_channel, kernel_size, padding=padding, bias=False), BatchNorm(feat_channel))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ic):

        context = self.conv1(x)
        spic = self.conv2(ic)

        context = F.interpolate(context, size=ic.size()[-2:], mode='bilinear')
        attn = self.sigmoid(self.conv3(torch.cat([context, spic, context*spic], dim=1)))

        x = F.interpolate(x, size=ic.size()[-2:], mode='bilinear')
        ic = self.conv4(ic)


        out = attn*x + (1-attn)*ic
        return out
    
class ICPG_conv(nn.Module):
    def __init__(self, feat_channel=128, ic_channel=64, kernel_size=3, BatchNorm=nn.BatchNorm2d):
        super(ICPG_conv, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(feat_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ic):
        spic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
        context = self.conv1(x)
        attn = self.sigmoid(self.conv2(torch.cat([context, spic, context*spic], dim=1)))
        out = attn*x + (1-attn)*spic
        return out


class ICPG_pool(nn.Module):
    def __init__(self, kernel_size=3):
        super(ICPG_pool, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,ic):
        ic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
        feat_avg_out = torch.mean(x, dim=1, keepdim=True)
        feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
        ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
        ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
        x_fuse = torch.cat([feat_avg_out, feat_max_out,ic_avg_out, ic_max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_fuse))
        out = attn*x + (1-attn)*ic
        return out
    
class ICPG_pool_correct(nn.Module):
    def __init__(self, kernel_size=3):
        super(ICPG_pool_correct, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,ic):
        ic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
        feat_avg_out = torch.mean(x, dim=1, keepdim=True)
        feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
        # ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
        # ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
        x_fuse = torch.cat([feat_avg_out, feat_max_out,ic], dim=1)
        attn = self.sigmoid(self.conv1(x_fuse))
        out = attn*x + (1-attn)*ic
        return out
    
class ICPG_pool_compare(nn.Module):
    def __init__(self, kernel_size=3):
        super(ICPG_pool_compare, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,ic):
        ic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
        feat_avg_out = torch.mean(x, dim=1, keepdim=True)
        feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
        # ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
        # ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
        x_fuse = torch.cat([feat_avg_out, feat_max_out,ic, ic], dim=1)
        attn = self.sigmoid(self.conv1(x_fuse))
        out = attn*x + (1-attn)*ic
        return out
    
class ICPG_pool_more(nn.Module):
    def __init__(self, kernel_size=3, ic_num=4):
        super(ICPG_pool_more, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(2+ic_num, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.ic_num = ic_num

    def forward(self, x,ic):
        ic = F.interpolate(ic, size=x.size()[-2:], mode='bilinear')
        feat_avg_out = torch.mean(x, dim=1, keepdim=True)
        feat_max_out, _ = torch.max(x, dim=1, keepdim=True)
        # ic_avg_out = torch.mean(ic, dim=1, keepdim=True)
        # ic_max_out, _ = torch.max(ic, dim=1, keepdim=True)
        cat_feat = [feat_avg_out, feat_max_out]
        for i in range(self.ic_num):
            cat_feat.append(ic)
        x_fuse = torch.cat(cat_feat, dim=1)
        attn = self.sigmoid(self.conv1(x_fuse))
        out = attn*x + (1-attn)*ic
        return out


class ICPG_conv_up(nn.Module):
    def __init__(self, feat_channel=128, ic_channel=64, kernel_size=3, BatchNorm=nn.BatchNorm2d):
        super(ICPG_conv_up, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(feat_channel, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False), BatchNorm(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ic):
        h,w = ic.size()[-2:]
        spic = F.interpolate(ic, size=(h//2, w//2), mode='bilinear')
        x = F.interpolate(x, size=spic.size()[-2:], mode='bilinear')
        context = self.conv1(x)
        attn = self.sigmoid(self.conv2(torch.cat([context, spic, context*spic], dim=1)))
        out = attn*x + (1-attn)*spic
        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out

class MSContext(nn.Module):
    def __init__(self, in_channels, out_channels, grids):
        super(MSContext, self).__init__()

        self.ms_context_aggregation = SCE(in_channels=in_channels, out_channels=out_channels , grids=grids)

    def forward(self, x):
        out = self.ms_context_aggregation(x)

        return out


class MLContext(nn.Module):
    def __init__(self, low_channels = 128, high_channels = 128, out_channels = 256):
        super(MLContext,self).__init__()

        self.ml_context_aggregation = SFF(low_channels, high_channels, out_channels)

    def forward(self, x_low, x_high):
        out = self.ml_context_aggregation(x_low, x_high)

        return out

class Decoder(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_scale):
        super(Decoder, self).__init__()

        self.decoder = BiSeNetOutput(in_chan, mid_chan, n_classes, up_scale)

    def forward(self, x):
        x = self.decoder(x)
        return x

'''Bilateral Guided Aggregation Layer to fuse the complementary information from both Detail Branch and Semantic Branch, from BiSeNetv2'''
class BGALayer(nn.Module):
    def __init__(self, out_channels=128, align_corners=False, norm_cfg=None, act_cfg=None):
        super(BGALayer, self).__init__()
        # set attrs
        self.out_channels = out_channels
        self.align_corners = align_corners
        # define modules
        self.detail_dwconv = nn.Sequential(DepthwiseSeparableConv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dw_norm_cfg=norm_cfg,
            dw_act_cfg=None,
            pw_norm_cfg=None,
            pw_act_cfg=None,
        ))
        self.detail_down = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
        )
        self.semantic_dwconv = nn.Sequential(DepthwiseSeparableConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dw_norm_cfg=norm_cfg,
            dw_act_cfg=None,
            pw_norm_cfg=None,
            pw_act_cfg=None,
        ))
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x_d, x_s):
        detail_dwconv = self.detail_dwconv(x_d)
        detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s)
        semantic_conv = F.interpolate(
            semantic_conv, size=detail_dwconv.shape[2:], mode='bilinear', align_corners=self.align_corners,
        )
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv)
        fuse_2 = F.interpolate(
            fuse_2, size=fuse_1.shape[2:], mode='bilinear', align_corners=self.align_corners
        )
        output = self.conv(fuse_1 + fuse_2)
        return output
    
'''Feature Fusion Module to fuse low level output feature of Spatial Path and high level output feature of Context Pat, from BiSeNetv1'''
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Sigmoid(),
        )
    '''forward'''
    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out
    


'''ASPP'''
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 12, 24, 36], align_corners=False, norm_cfg=None, act_cfg=None):
        super(ASPP, self).__init__()
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg)
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg)
                )
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features
    
'''PyramidPoolingModule'''
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=[1, 2, 3, 6], align_corners=False, norm_cfg=None, act_cfg=None):
        super(PyramidPoolingModule, self).__init__()
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for pool_scale in pool_scales:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg)
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_scales), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_lvls = [x]
        for branch in self.branches:
            out = branch(x)
            pyramid_lvls.append(F.interpolate(out, size=(h, w), mode='bilinear', align_corners=self.align_corners))
        output = torch.cat(pyramid_lvls, dim=1)
        output = self.bottleneck(output)
        return output