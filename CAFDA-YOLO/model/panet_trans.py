import torch.nn as nn

from model.layers import CBL, DownSampleConv, UpSampleConv
from model.trans_fusion import TransFusion


class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()
        self.feature_transform_3 = CBL(feature_channels[0], feature_channels[0] // 2, 1)  
        self.feature_transform_4 = CBL(feature_channels[1], feature_channels[1] // 2, 1)  
        self.resample_5_4 = UpSampleConv(feature_channels[2] // 2, feature_channels[1] // 2) 
        self.resample_4_3 = UpSampleConv(feature_channels[1] // 2, feature_channels[0] // 2)
        self.resample_3_4 = DownSampleConv(feature_channels[0] // 2, feature_channels[1] // 2)
        self.resample_4_5 = DownSampleConv(feature_channels[1] // 2, feature_channels[2] // 2)
        self.down_stream_conv5 = nn.Sequential(
            CBL(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            CBL(feature_channels[2] // 2, feature_channels[2], 3),
            CBL(feature_channels[2], feature_channels[2] // 2, 1)
        )
        self.down_stream_conv4 = nn.Sequential(
            CBL(feature_channels[1], feature_channels[1] // 2, 1),
            CBL(feature_channels[1] // 2, feature_channels[1], 3),
            CBL(feature_channels[1], feature_channels[1] // 2, 1),
            CBL(feature_channels[1] // 2, feature_channels[1], 3),
            CBL(feature_channels[1], feature_channels[1] // 2, 1)
        )
        self.trans_fusion_down_stream_conv4 = TransFusion(feature_channels[1] // 2, feature_channels[1])
        self.down_stream_conv3 = nn.Sequential(
            CBL(feature_channels[0], feature_channels[0] // 2, 1),
            CBL(feature_channels[0] // 2, feature_channels[0], 3),
            CBL(feature_channels[0], feature_channels[0] // 2, 1),
            CBL(feature_channels[0] // 2, feature_channels[0], 3),
            CBL(feature_channels[0], feature_channels[0] // 2, 1),
        )
        self.trans_fusion_down_stream_conv3 = TransFusion(feature_channels[0] // 2, feature_channels[0])
        self.up_stream_conv4 = nn.Sequential(
            CBL(feature_channels[1], feature_channels[1] // 2, 1),
            CBL(feature_channels[1] // 2, feature_channels[1], 3),
            CBL(feature_channels[1], feature_channels[1] // 2, 1),
            CBL(feature_channels[1] // 2, feature_channels[1], 3),
            CBL(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.trans_fusion_up_stream_conv4 = TransFusion(feature_channels[1] // 2, feature_channels[1])
        self.up_stream_conv5 = nn.Sequential(
            CBL(feature_channels[2], feature_channels[2] // 2, 1),
            CBL(feature_channels[2] // 2, feature_channels[2], 3),
            CBL(feature_channels[2], feature_channels[2] // 2, 1),
            CBL(feature_channels[2] // 2, feature_channels[2], 3),
            CBL(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.trans_fusion_up_stream_conv5 = TransFusion(feature_channels[2] // 2, feature_channels[2])

    def forward(self, features):
        features = [
            self.feature_transform_3(features[0]), 
            self.feature_transform_4(features[1]),  
        ]
        downstream_feature5 = self.down_stream_conv5(features[2])  
      
        downstream_feature4 = self.down_stream_conv4(
            self.trans_fusion_down_stream_conv4(features[1], self.resample_5_4(downstream_feature5)))
     
        downstream_feature3 = self.down_stream_conv3(
            self.trans_fusion_down_stream_conv3(features[0], self.resample_4_3(downstream_feature4)))
      
        upstream_feature4 = self.up_stream_conv4(
            self.trans_fusion_up_stream_conv4(self.resample_3_4(downstream_feature3), downstream_feature4))
     
        upstream_feature5 = self.up_stream_conv5(
            self.trans_fusion_up_stream_conv5(self.resample_4_5(upstream_feature4), downstream_feature5))
        return [downstream_feature3, upstream_feature4, upstream_feature5]
