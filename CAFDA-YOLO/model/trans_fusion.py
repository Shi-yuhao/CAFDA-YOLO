import torch
import torch.nn as nn
import torch.nn.functional as F
from model.double_attention import DoubleAttention


class CustomMultiheadAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, factor=8):
        super().__init__()
        assert v_dim % factor == 0 and qk_dim % factor == 0
        assert v_dim == qk_dim * 2, "v_dim must be double of qk_dim"

        self.groups = factor
        self.v_group_channels = v_dim // factor
        self.qk_group_channels = qk_dim // factor

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_diag1 = lambda x: torch.diagonal(x, dim1=2, dim2=3).permute(0, 2, 1)
        self.pool_diag2 = lambda x: torch.diagonal(torch.flip(x, dims=[3]), dim1=2, dim2=3).permute(0, 2, 1)

        self.conv1x1 = nn.Conv2d(self.qk_group_channels, self.v_group_channels, 1)
        self.conv3x3 = nn.Conv2d(self.v_group_channels, self.v_group_channels, 3, padding=1)
        self.gn = nn.GroupNorm(1, self.v_group_channels)
        self.diag_proj = nn.Linear(self.qk_group_channels, self.v_group_channels)
        self.out_proj = nn.Conv2d(self.v_group_channels * self.groups, self.v_group_channels * self.groups, 1)

    def forward(self, q, k, v):
        b, _, h, w = v.size()

        q = q.reshape(b * self.groups, self.qk_group_channels, h, w)
        k = k.reshape(b * self.groups, self.qk_group_channels, h, w)
        v = v.reshape(b * self.groups, self.v_group_channels, h, w)

        h_feat = self.pool_h(q)
        w_feat = self.pool_w(q).permute(0, 1, 3, 2)
        hw_feat = torch.cat([h_feat, w_feat], dim=2)
        hw_weight = torch.sigmoid(self.conv1x1(hw_feat))
        h_weight, w_weight = torch.split(hw_weight, [h, w], dim=2)
        w_weight = w_weight.permute(0, 1, 3, 2)
        attn_hw = v * h_weight * w_weight

        diag1 = self.pool_diag1(k)
        diag2 = self.pool_diag2(k)
        diag_feat = diag1 + diag2                      
        diag_feat = diag_feat.mean(dim=1)             
        diag_feat = self.diag_proj(diag_feat)        
        diag_feat = diag_feat.view(b * self.groups, self.v_group_channels, 1, 1)

        
        diag_feat = F.interpolate(diag_feat, size=(h, w), mode='nearest')  
        diag_feat = diag_feat.expand(-1, self.v_group_channels, -1, -1)    

        attn_diag = v * torch.sigmoid(diag_feat)

        x1 = self.gn(attn_hw + attn_diag)
        x2 = self.conv3x3(v)

        a1 = F.softmax(F.adaptive_avg_pool2d(x1, (1, 1)).view(b * self.groups, 1, -1), dim=-1)
        v1 = x2.view(b * self.groups, self.v_group_channels, -1)
        out = torch.matmul(a1, v1).view(b * self.groups, 1, h, w)
        out = (v * out.sigmoid()).view(b, self.v_group_channels * self.groups, h, w)
        return self.out_proj(out), None


class TransMLP(nn.Module):
    def __init__(self, in_feature, out_feature, expand_dim_ratio=4):
        super(TransMLP, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(in_feature, int(in_feature * expand_dim_ratio)),
            nn.GELU(),
            nn.Linear(int(in_feature * expand_dim_ratio), out_feature)
        )

    def forward(self, x):
        return self.feedforward(x)


class TransFusion(nn.Module):
    def __init__(self, input_channel, hidden_feature, head_num=4, dropout=0.2, trans_block=4):
        super(TransFusion, self).__init__()
        self.linear_projection = nn.Conv2d(input_channel, hidden_feature, 1)

        self.cma = CustomMultiheadAttention(
            qk_dim=hidden_feature, v_dim=hidden_feature * 2, factor=8
        )

        self.double_attn = DoubleAttention(
            in_channels=hidden_feature,
            c_m=hidden_feature // 4,
            c_n=hidden_feature // 4,
            reconstruct=True
        )


        self.feature_transform = nn.Linear(hidden_feature * 2, hidden_feature)
        self.input_norm = nn.LayerNorm(hidden_feature)
        self.fusion_mhsa_norm = nn.LayerNorm(hidden_feature * 2)
        self.mhsa_norm = nn.BatchNorm2d(hidden_feature)
        self.ffn_norm = nn.LayerNorm(hidden_feature)
        self.ffn = TransMLP(hidden_feature, hidden_feature)
        self.trans_block = trans_block

    def forward(self, inputs_1, inputs_2):
        b, c, h, w = inputs_1.shape

        inputs_1 = self.linear_projection(inputs_1)
        inputs_2 = self.linear_projection(inputs_2)
        inputs_1_seq = inputs_1.flatten(2).permute(0, 2, 1)

        inputs_2_seq = inputs_2.flatten(2).permute(0, 2, 1)
        inputs_cat = torch.cat([inputs_1_seq, inputs_2_seq], dim=2)  
 
        res = self.feature_transform(inputs_cat)
        normed = self.fusion_mhsa_norm(inputs_cat)
        q = inputs_1_seq.permute(0, 2, 1).reshape(b, -1, h, w) 
        k = inputs_2_seq.permute(0, 2, 1).reshape(b, -1, h, w)
        v = normed.permute(0, 2, 1).reshape(b, -1, h, w)

        inputs, _ = self.cma(q, k, v)
        inputs = inputs.flatten(2).permute(0, 2, 1)
        
        inputs = self.ffn_norm(self.feature_transform(inputs) + res)
        res = inputs

        inputs = self.ffn(inputs) + res

        for _ in range(self.trans_block - 1):  
            
            res = inputs
            inputs = inputs.permute(0, 2, 1).reshape(b, c*2, h, w)
            inputs = self.mhsa_norm(inputs) 
            inputs = inputs.flatten(2).permute(0, 2, 1) 
            inputs = inputs.permute(0,2,1).reshape(b, c*2, h, w) 
            inputs = self.double_attn(inputs)
            inputs = inputs.flatten(2).permute(0,2,1) 
            inputs = inputs + res
            inputs = self.ffn_norm(inputs)
            res = inputs
            inputs = self.ffn(inputs) + res
            
        inputs = inputs.permute(0, 2, 1).reshape(b, c * 2, h, w)  
        return inputs
