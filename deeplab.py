import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import  torch.nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

affine_par=True
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv =nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
               out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query=proj_query.view(m_batchsize, -1, width*height)
        proj_query=proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        proj_value=proj_value.view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(2048, depth, 1, 1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1,
                                  padding=padding_series[0], dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1,
                                  padding=padding_series[1], dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1,
                                  padding=padding_series[2], dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(
            depth*5, 256, kernel_size=3, padding=1)  # 512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        self.sa = PAM_Module(256)
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256, affine=affine_par),
                                    nn.ReLU())
        self.dropout = nn.Dropout2d(p=0.1)
        # for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1,
                         padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        #out = self.conv2d_list[0](x)
        #mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(
        image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.sa(out)
        out = self.conv51(out)
        out = self.dropout(out)
        # for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.device = torch.device('cpu')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(
            ASPP, [6, 12, 18], [6, 12, 18], 512)
        self.main_classifier = nn.Conv2d(
            256, num_classes, kernel_size=1, bias=True)
        self.softmax = nn.Sigmoid()  # nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                      dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fea = self.layer5(x)
        x = self.main_classifier(fea)
        #print("before upsample, tensor size:", x.size())
        # upsample to the size of input image, scale=8
        x = F.upsample(x, input_size, mode='bilinear')
        #print("after upsample, tensor size:", x.size())
        x = self.softmax(x)
        return  x


def GNNNet(num_classes=2,modality='normal'):
    if(modality=='normal'):
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes) #resnet-101
    # model = CoattentionModel(Bottleneck, [3, 4, 6, 3], num_classes - 1, modality)
    return model


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
               out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query=proj_query.view(m_batchsize, -1, width*height)
        proj_query=proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        proj_value=proj_value.view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x

        return out

class AttentionDeeplabV3(nn.Module):
    def __init__(self,num_classes,model=None):
        super(AttentionDeeplabV3, self).__init__()
        if(model!= None):
            self.body=model
        else:
            model = models.segmentation.deeplabv3()
            model.classifier = DeepLabHead(2048, 21)
            aspp_layer = create_feature_extractor(model, return_nodes={'classifier.0.project.3': 'out'})
            self.body=aspp_layer
        self.sa = PAM_Module(256)
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256, affine=affine_par),
                                    nn.ReLU())
        self.dropout = nn.Dropout2d(p=0.1)
        self.main_classifier = nn.Conv2d(
            256, num_classes, kernel_size=1, bias=True)
        self.softmax = nn.Sigmoid()  # nn.Softmax()

    def forward(self, x):
        input_size = x.size()[2:]
        x=self.body(x)['out']
        x = self.sa(x)
        x = self.conv51(x)
        x = self.dropout(x)
        x = self.main_classifier(x)
        x = F.upsample(x, input_size, mode='bilinear')
        x = self.softmax(x)
        return  x