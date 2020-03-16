from model import common
import torch.nn as nn


class Modify_RCAN(nn.Module):
    def __init__(self,net,args,checkpoint,all_params=False):
        super().__init__()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        kernel_size=3
        self.args=args

        
        
        
        
        self.head=net.model.head
        
#         self.tail=net.model.tail
        modules_head = [common.default_conv(args.channels, args.n_feats, kernel_size)]
            
        rgb_mean_pr=[0.004534]
        rgb_std_pr = [1.0]
        self.sub_mean_pr = common.MeanShift(600, rgb_mean_pr, rgb_std_pr,1)
    
    
    
        if all_params:
            modules_tail = [
            common.Upsampler(common.default_conv, args.scale[0],args.n_feats, act=False),
    #             common.default_conv(args.n_feats, 1, kernel_size)
                common.default_conv(args.n_feats, args.channels, kernel_size)

            ]
            self.add_mean = common.MeanShift(500, rgb_mean_pr, rgb_std_pr,args.channels,1)
        else:
            modules_tail = [
            common.Upsampler(common.default_conv, args.scale[0],args.n_feats, act=False),
                common.default_conv(args.n_feats, 1, kernel_size)
#                 common.default_conv(args.n_feats, args.channels, kernel_size)

            ]
            self.add_mean = common.MeanShift(600, rgb_mean_pr, rgb_std_pr,1,1)
        
#         rgb_mean = (0.0020388064770,0.0020388064770,0.0020388064770)
#         if args.channels==1:
#             rgb_mean = [0.0020388064770]
#             rgb_std = [1.0]
#         if args.channels==2:
#             rgb_mean = [0.0020388064770,0.0020388064770]
#             rgb_std = [1.0,1.0]
#         if args.channels==3:
#             rgb_mean = [0.0020388064770,0.0020388064770,0.0020388064770]
#             rgb_std = [1.0,1.0,1.0]

        
        rgb_mean_dem=[0.05986051]
        rgb_std_dem = [1.0]
        self.sub_mean_dem = common.MeanShift(2228.3303, rgb_mean_dem, rgb_std_dem,1)    
        
        rgb_mean_psl=[0.980945]
        rgb_std_psl = [1.0]
        self.sub_mean_psl = common.MeanShift(103005.8125, rgb_mean_psl, rgb_std_psl,1) 
 ########################################################################################

        rgb_mean_zg=[0.04429504,0.04453959, 0.04467554, 0.04476025 ,0.04485082 ,0.04495631,0.04496927, 0.04500141, 0.04497658, 0.04496762, 0.04475294, 0.04429719,0.04421607, 0.04418965, 0.04413029, 0.04394101, 0.04353212 ,0.04325647,0.04286277, 0.04167133 ,0.03755078, 0.01161219]#22dim
        rgb_std_zg = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.sub_mean_zg = common.MeanShift(49000., rgb_mean_zg, rgb_std_zg,22) 
        
        rgb_mean_tasmax=[0.8990] 
        rgb_std_tasmax = [1.0]
        self.sub_mean_tasmax = common.MeanShift(340, rgb_mean_tasmax, rgb_std_tasmax,1)
        
        rgb_mean_tasmin=[0.964896]
        rgb_std_tasmin = [1.0]
        self.sub_mean_tasmin = common.MeanShift(308.69238, rgb_mean_tasmin, rgb_std_tasmin,1) 
        
        


        self.body=net.model.body
        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
#         self.add_mean=net.model.add_mean
#         self.sub_mean=net.model.sub_mean
#         self.body = nn.Sequential(
#                 net.model.head,
#                 net.model.body,
#                 net.model.tail
#         )

    def forward(self, pr,dem=None,psl=None,zg=None,tasmax=None,tasmin=None):
        
        x = self.sub_mean_pr(pr)
        if self.args.dem:
            dem = self.sub_mean_dem(dem)
            x=torch.cat((x,dem),dim=1)
            
        if self.args.psl:
            psl = self.sub_mean_psl(psl)
            x=torch.cat((x,psl),dim=1)
            
        if self.args.zg:
            zg = self.sub_mean_zg(zg)
            x=torch.cat((x,zg),dim=1)
            
        if self.args.tasmax:
            tasmax = self.sub_mean_tasmax(tasmax)
            x=torch.cat((x,tasmax),dim=1)
            
        if self.args.tasmin:
            tasmin = self.sub_mean_tasmin(tasmin)
            x=torch.cat((x,tasmin),dim=1)        

        x = self.head(x)
        res = self.body(x)
        res += x
        x=self.tail(res)
        x = self.add_mean(x)
        return x
