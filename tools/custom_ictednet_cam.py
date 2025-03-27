# ------------------------------------------------------------------------------
# Modified from PIDNet
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# color_map = [(128, 64,128),
#              (244, 35,232),
#              ( 70, 70, 70),
#              (102,102,156),
#              (190,153,153),
#              (153,153,153),
#              (250,170, 30),
#              (220,220,  0),
#              (107,142, 35),
#              (152,251,152),
#              ( 70,130,180),
#              (220, 20, 60),
#              (255,  0,  0),
#              (  0,  0,142),
#              (  0,  0, 70),
#              (  0, 60,100),
#              (  0, 80,100),
#              (  0,  0,230),
#              (119, 11, 32)]
color_map = [(0, 128, 192), (128, 0, 0), (64, 0, 128),
                             (192, 192, 128), (64, 64, 128), (64, 64, 0),
                             (128, 64, 128), (0, 0, 192), (192, 128, 128),
                             (128, 128, 128), (128, 128, 0)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='ictednet_small', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=False)
    parser.add_argument('--p', help='dir for pretrained model', default='./trained_weights/camvid/ictednet_small_camvid.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='./samples/imgs/', type=str)
    parser.add_argument('--o', help='the output dir', default='./samples/outputs/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = True)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    # images_list = glob.glob(args.r+'*'+args.t)
    images_list = os.listdir(args.r)
    sv_path = args.o
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    

    model_name = getattr(models, args.a)
    
    if args.a == 'ictednet_large':
        model = model_name.get_pred_model('resnet18', 19 if args.c else 11)
    elif args.a == 'ictednet_small':
        model = model_name.get_pred_model('pidnet_s', 19 if args.c else 11)

    model = load_pretrained(model, args.p).cuda()
    model.eval()

    img_list_bar = tqdm(images_list)
    
    with torch.no_grad():
        # print(images_list)
        # print('start')
        for img_path in img_list_bar:
            
            img_name = img_path.split("\\")[-1]
            
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            # cv2.imwrite(sv_path+'2.png', img)
            
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()

            # cv2.imwrite(sv_path+'2.png', img)
            
            # print(img.size()[-2:])
            # plt.matshow(img[0, 0].cpu())
            # plt.colorbar()
            # plt.savefig(sv_path+'2.png')
            pred, icmap = model(img)
            # plt.matshow(pred[0, 0].cpu())
            # plt.colorbar()
            # plt.savefig(sv_path+'3.png')
            # se = F.interpolate(se, size=img.size()[-2:], 
            #                      mode='bilinear')
            # sp = F.interpolate(sp, size=img.size()[-2:], 
            #                      mode='bilinear')
            # bd = F.interpolate(bd, size=img.size()[-2:], 
            #                      mode='bilinear')
            # ic = F.interpolate(ic, size=img.size()[-2:], 
            #                      mode='bilinear')
            # ic = (ic * 255)
            # .round().squeeze().detach().cpu().numpy().astype('uint8')
            # print('1')
            # sp,se,sd = mid[0],mid[1], mid[2]
            # pred = F.interpolate(pred, size=img.size()[-2:], 
            #                      mode='bilinear', align_corners=True)
            if pred.size()[-2] != img.size()[-2] or pred.size()[-1] != img.size()[-1]:
                pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            # torch.save(pred, sv_path+'sample.pth')
            

            
            # feature_map = [sp,se,sd,icmap]
            # name = ['sp','se','bd','ic']

            #     # 可视化特征图
            # for i in range(len(feature_map)):
            #     print(i)
            #     map = feature_map[i][:,-1,:,:].transpose(2,0)
            #     # map = torch.mean(feature_map[i],dim=1).transpose(2,0)
            #     # f1 = feature_map[0][:,-1,:,:]
            #     # print(map.size())
            #     plt.figure()
            #     plt.imshow(map.cpu(), cmap='viridis')
            #     plt.axis('off')
            #     plt.savefig('./samples_out/sec_icfrnet_feature_map_'+ name[i] +'.png', dpi=600)
            #     plt.show()

            # cv2.imwrite('./samples/feature_map_ic.png', ic)
            
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            
            
            sv_img.save(sv_path+img_name)
            # cv2.imwrite(sv_path+'1.png', pred)

            # break
            
            
            
        
        