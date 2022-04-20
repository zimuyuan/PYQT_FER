import torch as t
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont, ImageDraw, Image

import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from torch.autograd import Variable
import os
import time


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, pool="max", k_size=3):
        super(eca_layer, self).__init__()

        if pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MainModel(nn.Module):
    def __init__(self, backbone_arch, swap_num=[7, 7]):
        super(MainModel, self).__init__()
        # self.use_dcl = False
        self.use_dcl = True
        self.num_classes = 7
        self.backbone_arch = backbone_arch
        self.swap_num = swap_num

        self.use_catt_pool = 'max'

        self.channel_num = 2048

        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
        else:
            print("err")
            exit()

        if self.backbone_arch == 'resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        elif self.backbone_arch == 'resnet18' or self.backbone_arch == 'resnet34':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.channel_num = 512
        elif self.backbone_arch == 'vgg16' or self.backbone_arch == 'vgg16_bn' \
             or self.backbone_arch == 'vgg19' or self.backbone_arch == 'vgg19_bn':
            self.model = nn.Sequential(*list(self.model.children())[0][0:-1])
            self.channel_num = 512
        else:
            print("eee")
            exit()

        # attension
        if self.use_catt_pool:
            self.cattension = eca_layer(self.channel_num,
                                        pool=self.use_catt_pool)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(self.channel_num, self.num_classes, bias=False)

        if self.use_dcl:
            self.classifier_swap = nn.Linear(self.channel_num, 2, bias=False)
            self.Convmask = nn.Conv2d(self.channel_num,
                                      self.swap_num[0] * self.swap_num[0], 1,
                                      stride=1, padding=0, bias=True)

            self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x, last_cont=None):

        x = self.model(x)
        if self.use_catt_pool:
            x = self.cattension(x)

        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        return out


def preprocess_image(img, resize=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        normalize
    ])

    img_pil = Image.open(img)

    img_tensor = preprocess(img_pil)

    return img_tensor


class ADC_FER():
    def __init__(self, backbone='resnet18',swap_num=[7,7], resize=256, crop=224, model= '../models/fer_model/res18_112_0.8543_32.pth',use_cuda=True):

        self.backbone = backbone
        self.swap_num = swap_num
        self.resize = resize
        self.crop = crop

        # self.net = MainModel(backbone, swap_num=swap_num).eval()
        # self.model = model
        # self.loadmodel()

        # self.net = MainModel('resnet50')
        # self.model = '../models/fer_model/resnet50_raf-db_gmp_0.8872.pth'
        # self.loadmodel()

        self.use_cuda = use_cuda
        self.emotion = ['Sur', 'Fear', 'Dis', 'Hap', 'Sad', 'Ang', 'Neu']
        # self.net = MainModel('resnet18', swap_num=[4, 4])
        # self.net = MainModel('vgg16', swap_num=[7, 7])
        # self.model = '../models/fer_model/weights_0.8595_27.pth'

        # self.net = MainModel('vgg19_bn', swap_num=[7, 7])
        # self.model = '../models/fer_model/weights_0.8699_43.pth'

        # net_weights = '../models/fer_model/vgg19_bn_256_224_7_net_weights_0.8699.pth'
        # net_weights = '../models/fer_model/vgg16_256_224_7_net_weights_0.8595.pth'
        # net_weights = '../models/fer_model/resnet18_256_224_4_net_weights_0.8709.pth'
        # net_weights = '../models/fer_model/resnet34_256_224_7_net_weights_0.8774.pth'
        net_weights = model

        self.loadNetModel(net_weights)
        # self.resize = int(net_weights.split('/')[-1].split('_')[1])
        # self.crop = int(net_weights.split('/')[-1].split('_')[2])
        # self.loadmodel()
        # torch.save(self.net, net_weights)
        # exit()

        # self.net = torch.load(net_weights).eval()
        # # exit()
        # if self.use_cuda:
        #     self.net = self.net.cuda()

    def loadNetModel(self, net_weights ):
        # self.resize = int(net_weights.split('/')[-1].split('_')[1])
        # self.crop = int(net_weights.split('/')[-1].split('_')[2])

        self.net = torch.load(net_weights).eval()
        # exit()
        if self.use_cuda:
            self.net = self.net.cuda()

    def setCudaModel(self):
        self.use_cuda = True
        self.net = self.net.cuda()

    def setCpuModel(self):
        self.use_cuda = False
        self.net = self.net.cpu()

    def fer_numpy(self, image):
        if self.use_cuda:
            return self.fer_numpy_cuda(image)
        else:
            return self.fer_numpy_cpu(image)

    def fer_numpy_cuda(self, image):

        input = self.preprocess_numpy_image(image=image)
        input = Variable(input.unsqueeze(0)).cuda()
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)

        return net_pred, net_pro.item(), h_x.cpu().numpy().tolist()

    def fer_numpy_cpu(self, image):

        input = self.preprocess_numpy_image(image=image)
        input = Variable(input.unsqueeze(0))
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)
        return net_pred, net_pro.item(), h_x.numpy().tolist()

    def fer_image(self, image):
        if self.use_cuda:
            return self.fer_image_cuda(image)
        else:
            return self.fer_image_cpu(image)

    def fer_image_cpu(self, image):
        input = self.preprocess_image(image=image)
        input = Variable(input.unsqueeze(0))
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)
        return self.emotion[net_pred], net_pro.item(), h_x.numpy().tolist()

    def fer_image_cuda(self, image):
        input = self.preprocess_image(image=image)
        input = Variable(input.unsqueeze(0)).cuda()
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)
        return self.emotion[net_pred], net_pro.item(), h_x.cpu().numpy().tolist()

    def loadmodel(self):
        pretrained_state_dict = torch.load(self.model)
        net_state_dict = self.net.state_dict()
        for key in net_state_dict:
            if 'module.' + key not in pretrained_state_dict:
                print(key)
                print('err')
                exit()
            net_state_dict[key] = pretrained_state_dict['module.' + key]
        self.net.load_state_dict(net_state_dict)

    def preprocess_numpy_image(self, image):

        image = Image.fromarray(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.CenterCrop((self.crop, self.crop)),
            transforms.ToTensor(),
            normalize
        ])
        img_tensor = preprocess(image)

        return img_tensor

    def preprocess_image(self, image):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.CenterCrop((self.crop, self.crop)),
            transforms.ToTensor(),
            normalize
        ])
        img_pil = Image.open(image)
        img_tensor = preprocess(img_pil)

        return img_tensor


class ADC_Net():
    def __init__(self, backbone='resnet18', swap_num=[7, 7], resize=256, crop=224,
                 model='../models/fer_model/resnet50_256_224_7_net_weights_0.8872.pth', use_cuda=True):

        self.backbone = backbone
        self.swap_num = swap_num
        self.resize = resize
        self.crop = crop
        self.model = model
        self.model_name = self.model.split('/')[-1]

        backbone, resize, crop, swap_num, _, _, _ = self.model_name.split('_'),

        self.backbone = backbone
        self.swap_num = [int(swap_num), int(swap_num)]
        self.resize = int(resize)
        self.crop = int(crop)

        self.use_cuda = use_cuda

        self.emotion = ['Sur', 'Fear', 'Dis', 'Hap', 'Sad', 'Ang', 'Neu']

        self.net = torch.load(self.model).eval()

        # exit()
        if self.use_cuda:
            self.net = self.net.cuda()

    def SetCudaModel(self):
        self.net = self.net.cuda()

    def SetCpuModel(self):
        self.net = self.net.cpu()

    def FERNumpyCuda(self, image):

        input = self.PreprocessNumpyImage(image=image)
        input = Variable(input.unsqueeze(0)).cuda()
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)

        return net_pred, net_pro.item(), h_x.cpu().numpy().tolist()

    def FERImageCpu(self, image, resize=112):

        input = self.PreprocessImage(image=image, resize=resize)
        input = Variable(input.unsqueeze(0))
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)
        return self.emotion[net_pred], net_pro.item(), h_x

    def fer_image_cuda(self, image, resize=112):
        self.loadmodel()
        input = self.PreprocessImage(image=image, resize=resize)
        input = Variable(input.unsqueeze(0)).cuda()
        output = self.net(input)
        h_x = F.softmax(output[0], dim=1).data
        net_pro, net_pred = torch.max(h_x, 1)
        return self.emotion[net_pred], net_pro.item(), h_x

    def LoadModel(self):
        pretrained_state_dict = torch.load(self.model)
        net_state_dict = self.net.state_dict()
        for key in net_state_dict:
            if 'module.' + key not in pretrained_state_dict:
                print(key)
                print('err')
                exit()
            net_state_dict[key] = pretrained_state_dict['module.' + key]
        self.net.load_state_dict(net_state_dict)

    def PreprocessNumpyImage(self, image):

        image = Image.fromarray(image)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.CenterCrop((self.crop, self.crop)),
            transforms.ToTensor(),
            normalize
        ])
        img_tensor = preprocess(image)

        return img_tensor

    def PreprocessImage(self, image):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.CenterCrop((self.crop, self.crop)),
            transforms.ToTensor(),
            normalize
        ])

        img_pil = Image.open(image)
        img_tensor = preprocess(img_pil)

        return img_tensor

#
# if __name__ == '__main__':
#
#     adc_fer = ADC_FER()
#     image = 'test_0013.jpg'
#     image = cv2.imread(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = torch.from_numpy(image)
#
#     e, p = adc_fer.fer_numpy_cuda(image, resize=112)
#     # e, p = adc_fer.fer_image_cuda('test_0013.jpg', resize=112)
#     print(e, p)

#
if __name__ == '__main__':

    # path_root = 'F:/DeepLearning/pytorch-pretrain-models'
    #
    # pretrained_model = {
    #     'resnet18': path_root + '/resnet18-5c106cde.pth',
    #     'resnet34': path_root + '/resnet34-333f7ec4.pth',
    #     'resnet50': path_root + '/resnet50-19c8e357.pth',
    #     'se_resnet50': path_root + '/se_resnet50-ce0d4300.pth',
    #     'vgg19': path_root + '/vgg19_bn-c79401a0.pth',
    #     'vgg16': path_root + '/vgg16-397923af.pth',
    #     'resnest50': path_root + '/resnest50-528c19ca.pth',
    #     'raf_db_resnest50': path_root + '/resnet50_raf-db_gmp_0.8872.pth',
    #     'raf_db_resnest50_baseline': path_root + '/resnet50_baseline_0.8615.pth'}
    #
    # model_dir = r'E:\pyqt\FER_pyqt\models\fer_model\res18_112_0.8435_24.pth'
    #
    emotion = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    # net = MainModel("resnet50")
    # pretrained_state_dict = torch.load(pretrained_model['raf_db_resnest50'])

    # net = MainModel("resnet18")
    # pretrained_state_dict = torch.load(model_dir)

    # pretrained_state_dict = torch.load(pretrained_model['raf_db_resnest50_baseline'])

    # net_state_dict = net.state_dict()
    # net.eval()
    # # net.cuda()
    #
    # for key in net_state_dict:
    #     if 'module.' + key not in pretrained_state_dict:
    #         print(key)
    #         print('err')
    #         exit()
    #     net_state_dict[key] = pretrained_state_dict['module.' + key]

    # net.load_state_dict(net_state_dict)

    # net_weights = '../models/fer_model/vgg16_256_224_7_net_weights_0.8595.pth'
    net_weights = '../models/fer_model/vgg19_bn_256_224_7_net_weights_0.8699.pth'

    # net_weights = '../models/fer_model/resnet18_256_224_4_net_weights_0.8709.pth'
    # net_weights = '../models/fer_model/resnet34_256_224_7_net_weights_0.8774.pth'
    # net_weights = '../models/fer_model/resnet50_256_224_7_net_weights_0.8872.pth'

    # self.loadmodel()
    # torch.save(self.net, net_weights)
    # exit()

    print('net')
    # net = torch.load(net_weights).eval()
    net = torch.load(net_weights).eval().cuda()
    # input = preprocess_image('test_0013.jpg', resize=224)
    input = preprocess_image('test_0013.jpg', resize=224).cuda()
    input = Variable(input.unsqueeze(0))

    for i in range(50):
        # t = time.time()
        output = net(input)

    print('测试')
    time_time = time.time()
    for i in range(1000):
        # t = time.time()
        output = net(input)
        # print("cost time:{}".format(time.time() - t))
    print("total cost time:{}".format(time.time() - time_time))

    h_x = F.softmax(output[0], dim=1).data
    net_pro, net_pred = torch.max(h_x, 1)
    net_pro = '%0.4f' % net_pro

    print('%s:%s'% ( emotion[net_pred], net_pro))

#
