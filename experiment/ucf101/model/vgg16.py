import torch.nn as nn
import torchvision.models as models

class MyVGG16(nn.Module):
    def __init__(self, train_type="RGB"):
        super(MyVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=(train_type == "RGB"))
        self.vgg16.classifier[2] = nn.Dropout(p=0.9, inplace=False)
        self.vgg16.classifier[5] = nn.Dropout(p=0.9, inplace=False)
        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=101, bias=True)
        if (train_type == "FLOW"):
            self.vgg16.features[0] = nn.Conv2d(20, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.vgg16(x)
        return x

class PtVGG16(nn.Module):
    def __init__(self, train_type="RGB"):
        super(PtVGG16, self).__init__()
        model = models.vgg16(pretrained=False)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
        )
        if(train_type == "FLOW"):
            self.features[0] = nn.Conv2d(20, 64, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.fc_action = nn.Linear(in_features=4096, out_features=101)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x
