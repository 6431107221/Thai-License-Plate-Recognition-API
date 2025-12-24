
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCRNN(nn.Module):
    """
    CRNN Architecture = ResNet18 Backbone + BiLSTM + CTC
    Customized strides in ResNet to maintain feature width for sequence modeling.
    """
    def __init__(self, img_channel=1, num_classes=46, hidden_size=256, num_rnn_layers=2, dropout_rnn=0.3):
        super().__init__()
        
        # 1. Backbone (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify Input for Grayscale
        resnet.conv1 = nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify Strides to preserve Width (Time dimension)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # 2. RNN Head
        self.out_channels = 512
        self.rnn = nn.LSTM(self.out_channels, hidden_size, num_layers=num_rnn_layers,
                           bidirectional=True, batch_first=True, dropout=dropout_rnn)
        
        # 3. Classifier
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = torch.mean(features, dim=2)
        features = features.permute(0, 2, 1)
        
        rnn_out, _ = self.rnn(features)
        output = self.linear(rnn_out)
        return output

class ProvinceClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.last_channel, n_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Utilities ---

def best_path_decode(log_probs, int_to_char, blank=0):
    preds = log_probs.argmax(-1)
    results = []
    
    for seq in preds:
        text = []
        prev = None
        for p in seq.tolist():
            if p != blank and p != prev:
                char = int_to_char.get(str(p)) or int_to_char.get(p)
                if char: text.append(char)
            prev = p
        results.append("".join(text))
    return results