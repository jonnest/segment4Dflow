import torch
import torch.nn as nn



class DualPathUNet(nn.Module):
    def __init__(self, local_model, global_model, num_classes=1):
        super(DualPathUNet, self).__init__()
        self.local_model = local_model  # The model handling local patches
        self.global_model = global_model  # The model handling global context

        # Fusion layers
        self.fusion_conv1 = nn.Conv3d(in_channels=local_model.num_features + global_model.num_features,
                                      out_channels=64, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm3d(64)
        self.fusion_relu1 = nn.ReLU(inplace=True)

        self.fusion_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, local_input, global_input):
        local_features = self.local_model(local_input)
        global_features = self.global_model(global_input)
        
        # Upsample global features to match local features' resolution
        global_features_upsampled = nn.functional.interpolate(global_features, size=local_features.shape[2:], mode='trilinear', align_corners=False)

        # Concatenate the features
        fused_features = torch.cat((local_features, global_features_upsampled), dim=1)

        # Fusion
        x = self.fusion_conv1(fused_features)
        x = self.fusion_bn1(x)
        x = self.fusion_relu1(x)
        output = self.fusion_conv2(x)
        return output
