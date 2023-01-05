from collections import OrderedDict
import  numpy as np
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from deeplab import AttentionDeeplabV3
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
    model = models.segmentation.deeplabv3_resnet101()
    model.classifier = DeepLabHead(2048, outputchannels)
    return model
weighted_model=createDeepLabv3(21)
# train_nodes, _ = get_graph_node_names(model) ;print(train_nodes)
return_nodes = {
        'classifier.0.project.3': 'out'}


for param in weighted_model.parameters():
    param.requires_grad = False

pretrained=torch.load('./pretrained/resnet101-5d3b4d8f.pth')



new_state_dict=weighted_model.state_dict()
new_pretrained= OrderedDict()

for k,v in pretrained['model_state'].items():
    name_split=k[:8]
    if(name_split=='backbone'):
        new_pretrained[k]=v
    else:
        name_split = k[:10]
        if (name_split == 'classifier'):
            new_name=k[11:]
            new_pretrained[new_name] = v

weighted_model.load_state_dict(new_pretrained)

weighted_model.eval()
weighted_model.cuda()
aspp_layer = create_feature_extractor(weighted_model, return_nodes=return_nodes)
model=AttentionDeeplabV3(aspp_layer,21)
model.eval()
model.cuda()
val_transform = transforms.Compose([
    transforms.Resize(513),
    transforms.CenterCrop(513),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_path='samples/2007_000346.jpg'
result_path = r"samples/image_results.png"
img = Image.open(img_path).convert('RGB')
img = val_transform(img).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'voc'
with torch.no_grad():
    print(img.shape)
    img = img.to(device)
    # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    pred = model(img)

    # intermediate_outputs = result(img)

    # model2 = create_feature_extractor(model, return_nodes=model.classifier.)
    pred=pred.max(dim=1)[1].cpu().numpy()[0, :, :]

    if dataset == 'voc':
        pred = voc_cmap()[pred].astype(np.uint8)


    Image.fromarray(pred).save(result_path)
    print("Prediction is saved in %s" % result_path)