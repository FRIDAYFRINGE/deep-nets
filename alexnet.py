import torch
from torch  import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self,num_classes = 1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            # batch_size, channel , height, width
            nn.Conv2d(3,96,stride=4, kernel_size=11,padding= 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96,256,stride = 2, kernel_size= 5, padding= 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256,384, kernel_size= 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384, kernel_size= 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256, kernel_size= 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)  

        )        
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),           
        )
    
    def forward(self,input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = AlexNet(num_classes=1000)
print(model)

num_para=  sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_para)


"""
output = model(input_data)
probab = F.softmax(output, dim =1)
"""