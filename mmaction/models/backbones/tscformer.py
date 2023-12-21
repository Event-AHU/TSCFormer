import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, NonLocal3d
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from torch.nn.modules.utils import _ntuple

from mmaction.registry import MODELS
from .resnet_tsm import ResNetTSM
from .utils.former import Former, Res2Former
from mmengine.model import BaseModule

@MODELS.register_module()
class TSCFormer(BaseModule):
    def __init__(self,
                 depth,
                 pretrained='',
                 num_segments=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 pretrained2d=True,
                 dropout_ratio:float = 0.8,
                 **kwargs):
        super(TSCFormer, self).__init__()
        self.ResNetTSM = ResNetTSM(depth=depth,
                                   pretrained=pretrained,
                                   num_segments=num_segments,
                                   is_shift=is_shift,
                                   non_local=non_local,
                                   non_local_cfg=non_local_cfg,
                                   shift_div=shift_div,
                                   shift_place=shift_place,
                                   temporal_pool=temporal_pool,
                                   pretrained2d=pretrained2d
                                   )
        
        self.ResNetTSM_event = ResNetTSM(depth=depth,
                                   pretrained=pretrained,
                                   num_segments=num_segments,
                                   is_shift=is_shift,
                                   non_local=non_local,
                                   non_local_cfg=non_local_cfg,
                                   shift_div=shift_div,
                                   shift_place=shift_place,
                                   temporal_pool=temporal_pool,
                                   pretrained2d=pretrained2d
                                   )
        
        # tsm for rgb 
        self.ResNetTSM.init_weights()
        # tsm for event
        self.ResNetTSM_event.init_weights()
        former_dim = 64
        self.former3 = Former(dim=former_dim, depth=1)
        self.Res2former3 = Res2Former(dim=former_dim, heads=2, channel=1024)
        self.former4 = Former(dim=former_dim, depth=1)
        self.Res2former4 = Res2Former(dim=former_dim, heads=2, channel=2048)
        token_num = 3
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, token_num, former_dim)))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_res = nn.Dropout(p=dropout_ratio)

        self.fc_layer3 = nn.Linear(former_dim*token_num*2, 1*14*14)
        self.fc_layer4 = nn.Linear(former_dim*token_num*2, 1*7*7)
        self.conv_1x1_rgb_layer3 = nn.Conv2d(1025, 1024, kernel_size=1)
        self.conv_1x1_event_layer3 = nn.Conv2d(1025, 1024, kernel_size=1)
        self.conv_1x1_rgb_layer4 = nn.Conv2d(2049, 2048, kernel_size=1)
        self.conv_1x1_event_layer4 = nn.Conv2d(2049, 2048, kernel_size=1)
    
    
    
    def forward(self, x: torch.Tensor, num_seg=8):
        num_seg=8
        inputs_rgb = x[0]
        inputs_event = x[1]

        x_rgb = inputs_rgb.contiguous().view((-1, ) + inputs_rgb.shape[2:])
        x_event = inputs_event.contiguous().view((-1, ) + inputs_event.shape[2:])
        
        x_rgb_stem_conv1 = self.ResNetTSM.conv1(x_rgb)     
        x_rgb_stem = self.ResNetTSM.maxpool(x_rgb_stem_conv1)   
        
        x_event_stem_conv1 = self.ResNetTSM_event.conv1(x_event)     
        x_event_stem = self.ResNetTSM_event.maxpool(x_event_stem_conv1)    
        
        out_rgb_layer1 = self.ResNetTSM.layer1(x_rgb_stem)            
        out_rgb_layer2 = self.ResNetTSM.layer2(out_rgb_layer1) 
        out_rgb_layer3 = self.ResNetTSM.layer3(out_rgb_layer2)   
        
        out_event_layer1 = self.ResNetTSM_event.layer1(x_event_stem)   
        out_event_layer2 = self.ResNetTSM_event.layer2(out_event_layer1)  
        out_event_layer3 = self.ResNetTSM_event.layer3(out_event_layer2)   
        
        nt_rgb, c, h, w   = out_rgb_layer3.shape       
        nt_event = out_event_layer3.shape[0]
        nt = nt_rgb + nt_event
        
        reshape_rgb_layer3 = out_rgb_layer3.view(-1, num_seg, c, h, w)       
        reshape_event_layer3 = out_event_layer3.view(-1, num_seg, c, h, w)  
        
        out_res_layer3 = torch.cat((reshape_rgb_layer3, reshape_event_layer3), 1)  
        out_res_layer3 = out_res_layer3.view(-1, c, h, w)           
        
        z = self.token.repeat(nt, 1, 1)                        
        z_hid3 = self.Res2former3(out_res_layer3, z)           
        z_out3 = self.former3(z_hid3)       
                             
        z_out_layer3 = z_out3.view(nt_rgb, -1)                   
        z_out_layer3 = self.fc_layer3(z_out_layer3)                   
        z_out_layer3 = z_out_layer3.view(nt_rgb, -1, h, w)

        input_rgb_layer4   = torch.cat((out_rgb_layer3, z_out_layer3), 1)      
        input_event_layer4 = torch.cat((out_event_layer3, z_out_layer3), 1)    
        
        input_rgb_layer4   = self.conv_1x1_rgb_layer3(input_rgb_layer4)              
        input_event_layer4 = self.conv_1x1_event_layer3(input_event_layer4)          
        
        out_rgb_layer4 = self.ResNetTSM.layer4(input_rgb_layer4)   
        out_event_layer4 = self.ResNetTSM_event.layer4(input_event_layer4)    
        
        b, c_4, h_4, w_4 = out_rgb_layer4.shape
        reshape_rgb_layer4 = out_rgb_layer4.view(-1, num_seg, c_4, h_4, w_4)
        reshape_event_layer4 = out_event_layer4.view(-1, num_seg, c_4, h_4, w_4)
        
        out_res_layer4 = torch.cat((reshape_rgb_layer4, reshape_event_layer4), 1)
        out_res_layer4 = out_res_layer4.view(-1, c_4, h_4, w_4)
    
        z_hid4 = self.Res2former4(out_res_layer4, z_out3)              
        z_out4 = self.former4(z_hid4)                                  
        
        z_out_layer4 = z_out4.view(nt_rgb, -1)                   
        z_out_layer4 = self.fc_layer4(z_out_layer4)                    
        z_out_layer4 = z_out_layer4.view(nt_rgb, -1, h_4, w_4)

        out_r   = torch.cat((out_rgb_layer4, z_out_layer4), 1)      
        out_e = torch.cat((out_event_layer4, z_out_layer4), 1)    
        
        out_r = self.conv_1x1_rgb_layer4(out_r)            
        out_e = self.conv_1x1_event_layer4(out_e)          
        
        out_r = out_r.view(-1, num_seg, c_4, h_4, w_4)
        out_e = out_e.view(-1, num_seg, c_4, h_4, w_4)
        
        out = torch.cat((out_r, out_e), 1)
        out = out.view(-1, c_4, h_4, w_4)
        
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout_res(out)
        
        z_out = z_out4.view(nt, -1)
        out_all = torch.cat((out, z_out), 1)
        
        
        return out_all
