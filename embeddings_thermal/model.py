import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(Encoder, self).__init__()

        # encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(8) if use_batch_norm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # encode
        x = F.relu(self.bn1(self.enc1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.enc2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.enc3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.enc4(x)))
        # x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(Decoder, self).__init__()

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=3, padding=1, output_padding=0)
        self.dbn1 = nn.BatchNorm2d(8) if use_batch_norm else nn.Identity()
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dbn2 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dbn3 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dbn4 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # decode
        x = F.relu(self.dbn1(self.dec1(x)))
        x = F.relu(self.dbn2(self.dec2(x)))
        x = F.relu(self.dbn3(self.dec3(x)))
        # x = F.relu(self.dbn4(self.dec4(x)))
        x = torch.sigmoid(self.out(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(Autoencoder, self).__init__()

        # encoder layers
        self.encoder = Encoder(use_batch_norm=use_batch_norm)

        # decoder layers
        self.decoder = Decoder(use_batch_norm=use_batch_norm)

    def forward(self, x):
        # encode
        x = self.encoder(x)
        x_embedding = torch.flatten(x, start_dim=1)

        # decode
        x = self.decoder(x)
        return x, x_embedding


# # Assuming the model and input size
# device = torch.device("cuda:1")
# model = Autoencoder(use_batch_norm=True).to(device)  # or False, depending on what you want to check
# batch_size = 256
# summary(model, input_size=(batch_size, 1, 85, 85), col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=4)
#
# # # Test the model
# tensor = torch.randn(batch_size, 1, 85, 85).to(device)
# output = model(tensor)
# print(output.shape)