from collections import OrderedDict
import  numpy as np
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101()
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
model=createDeepLabv3(21)
for param in model.parameters():
    param.requires_grad = False

pretrained=torch.load('./pretrained/best_deeplabv3plus_resnet101_voc_os16.pth')



new_state_dict=model.state_dict()
new_pretrained= OrderedDict()

for k,v in pretrained['model_state'].items():
    name_split=k[:8]
    if(name_split=='backbone'):
        new_pretrained[k]=v
    else:
        #aspp extractor##############################################
        name_split = k[11:]
        name_split2=name_split[:4]
        if(name_split2=='aspp'):
            new_name='classifier.0.'+name_split[5:]
            new_pretrained[new_name]=v
        #################################################################
        else:
            name_split2=name_split[:10]
            if(name_split2=='classifier'):
                name_split3=name_split.split('.')
                num=int(name_split3[1])
                num=num+1
                new_name = name_split3[0]+'.'+str(num)+'.'+name_split3[2]
                new_pretrained[new_name]=v
model.load_state_dict(new_state_dict)

model.eval()
model.cuda()

val_transform = transforms.Compose([
    transforms.Resize(513),
    transforms.CenterCrop(513),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_path='samples/2008_000004.jpg'
result_path = r"samples/image_results.png"
img = Image.open(img_path).convert('RGB')
img = val_transform(img).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'voc'
with torch.no_grad():
    print(img.shape)
    img = img.to(device)
    # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    pred = model(img)['out']
    pred=pred.max(dim=1)[1].cpu().numpy()[0, :, :]

    if dataset == 'voc':
        pred = voc_cmap()[pred].astype(np.uint8)


    Image.fromarray(pred).save(result_path)
    print("Prediction is saved in %s" % result_path)