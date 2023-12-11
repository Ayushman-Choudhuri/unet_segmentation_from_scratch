import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):  # modularized as this pattern keeps repeating throughout the network
    def __init__(self , in_channels , out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=3,stride=1,padding=1, bias = False),# Padding of 1 used to get same convolution and bias removed due to batch norm 
            nn.BatchNorm2d(out_channels), #batch normalization alongthe batch channel independently for each channel
            nn.ReLU(inplace = True), #modify input normalized tensor directly
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3 , out_channels=1 , features =[64,128,256,512]):  #in channels set to 3 due to RGB and out_chennel is 1 due to binary image segmentation
        super(UNet,self).__init__()
        self.decoder = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2 , stride=2)

        #Encoding part of UNET
        for feature in features: 
            self.encoder.append(DoubleConv(in_channels,feature))
            in_channels=feature

        #Decoder part of UNET
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(in_channels=feature*2 , out_channels=feature , kernel_size=2, stride=2)
                )
            
            self.decoder.append(DoubleConv(in_channels = feature*2,out_channels= feature))
            
        #bottleneck layer of Unet
        self.bottleneck = DoubleConv(features[-1] , features[-1]*2)

        #final convolution before output
        self.final_conv =nn.Conv2d(features[0], out_channels=1, kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the order of the previously stored skip connections

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)

            x = self.decoder[idx+1](concat_skip)

        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 160, 240))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()


