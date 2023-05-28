import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
import copy

class AgeModel(nn.Module):
    layer_name_mapping = {
        "conv1_1.weight": "features.0.weight",
        "conv1_1.bias": "features.0.bias",
        "conv1_2.weight": "features.2.weight",
        "conv1_2.bias": "features.2.bias",
        "conv2_1.weight": "features.5.weight",
        "conv2_1.bias": "features.5.bias",
        "conv2_2.weight": "features.7.weight",
        "conv2_2.bias": "features.7.bias",
        "conv3_1.weight": "features.10.weight",
        "conv3_1.bias": "features.10.bias",
        "conv3_2.weight": "features.12.weight",
        "conv3_2.bias": "features.12.bias",
        "conv3_3.weight": "features.14.weight",
        "conv3_3.bias": "features.14.bias",
        "conv4_1.weight": "features.17.weight",
        "conv4_1.bias": "features.17.bias",
        "conv4_2.weight": "features.19.weight",
        "conv4_2.bias": "features.19.bias",
        "conv4_3.weight": "features.21.weight",
        "conv4_3.bias": "features.21.bias",
        "conv5_1.weight": "features.24.weight",
        "conv5_1.bias": "features.24.bias",
        "conv5_2.weight": "features.26.weight",
        "conv5_2.bias": "features.26.bias",
        "conv5_3.weight": "features.28.weight",
        "conv5_3.bias": "features.28.bias",
        "fc6.weight": "classifier.0.weight",
        "fc6.bias": "classifier.0.bias",
        "fc7.weight": "classifier.3.weight",
        "fc7.bias": "classifier.3.bias",
        "fc8-101.weight": "classifier.6.weight",
        "fc8-101.bias": "classifier.6.bias",
    }

    def __init__(self, num_classes=101, checkpoint_path="weights/pretrained/age_model.pt"):
        super().__init__()
        self.num_classes = num_classes
        self.vgg = vgg16(num_classes=num_classes)

        state_dict = torch.load(checkpoint_path)
        state_dict_copy = copy.deepcopy(state_dict)
        for key in state_dict_copy:
            new_name = self.layer_name_mapping[key]
            state_dict[new_name] = state_dict.pop(key)

        self.vgg.load_state_dict(state_dict)
    
    def vgg_transform(self, x):
        x = x.mul(0.5).add(0.5)
        x[:,0,:,:] = x[:,0,:,:] - 0.48501961
        x[:,1,:,:] = x[:,1,:,:] - 0.45795686
        x[:,2,:,:] = x[:,2,:,:] - 0.40760392
        r, g, b = torch.split(x, 1, 1)
        out = torch.cat((b, g, r), dim=1)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        out = out * 255.
        return out

    def forward(self, x):
        x = self.vgg_transform(x)
        return self.vgg(x)
    
    def get_age(self, logits):
        predicted_age_pb = F.softmax(logits)
        predicted_age = torch.zeros(logits.size(0)).type_as(predicted_age_pb)
        for i in range(logits.size(0)):
            for j in range(logits.size(1)):
                predicted_age[i] += j * predicted_age_pb[i][j]
        return predicted_age