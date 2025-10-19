import torch
import torch.nn as nn
from modules.pyramidpooling import TemporalPyramidPooling
from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling'  #model name for registration
]

class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()

        #VGG architecture
        #Input 3x50x250 RGB height width
        self.conv = nn.Sequential(
            #1: 2 64-channel conv layers + max pooling = 64x25x125
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #2: 2 128-channel conv layers + max pooling = 128x12x62
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #3: 3 256-channel conv layers + max pooling = 256x6x31
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #4: 3 512-channel conv layers + max pooling = 512x3x15
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #5: 3 512-channel conv layers, no pooling
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #512x3x15
        )

        #multi-scale aggregation for pyramid levels
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        #PHOS
        self.phos = nn.Sequential(
            nn.Linear(self.temporal_pool.get_output_size(512), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),    #prevent overfitting
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 165)    #out 165-dimensional PHOS vector
        )

        #PHOC
        self.phoc = nn.Sequential(
            nn.Linear(self.temporal_pool.get_output_size(512), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 604)  #out604-dimensional PHOC vector
        )

        #Init weights using Kaiming and Normal init
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #Kaiming init for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #Normal init for linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        #extraction
        x = self.conv(x)

        #Temporal pyramid pooling
        x = self.temporal_pool(x)  #B, flattened_features

        #separate prediction
        phos_output = self.phos(x)  #Spatial character distribution
        phoc_output = self.phoc(x)  #Character n-gram presence

        return {'phos': phos_output, 'phoc': phoc_output}


#Model registration for timm library
@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    #Model testing and verification
    model = PHOSCnet()

    #create dummy input batch: 5 samples, 3 channels, 50x250 pixels
    x = torch.randn(5, 3, 50, 250)

    #forward
    y = model(x)

    print("PHOS output shape:", y['phos'].shape)  #Expected 5, 165
    print("PHOC output shape:", y['phoc'].shape)  #Expected 5, 604