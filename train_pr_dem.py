#pr dem chennel
import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
from args_parameter import args
from PrepareData import ACCESS_BARRA_v2_0,ACCESS_BARRA_v2_1,ACCESS_BARRA_v2_pr_dem
import torch
import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

# from PIL import Image
import time
from sklearn.model_selection import StratifiedShuffleSplit
import model
from model import my_model
import utility
from tqdm import tqdm
import math
import xarray as xr
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr,compare_mse
import platform
from torch.autograd import Variable


def write_log(log):
    print(log)
    if not os.path.exists("./model/save/"+args.train_name+"/"):
        os.mkdir("./model/save/"+args.train_name+"/")
    my_log_file=open("./model/save/"+args.train_name + '/train.txt', 'a')
#     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


def main():
    
#     pre_train_path="./model/save/temp01/"+0+".pth"

    
    init_date=date(1970, 1, 1)
    start_date=date(1990, 1, 2)
    end_date=date(2011,12,25)
#     end_date=date(2012,12,25) #if 929 is true we should substract 1 day    
    sys = platform.system()
    args.file_ACCESS_dir="../data/"
    args.file_BARRA_dir="../data/barra_aus/"
#     if sys == "Windows":
#         init_date=date(1970, 1, 1)
#         start_date=date(1990, 1, 2)
#         end_date=date(1990,12,15) #if 929 is true we should substract 1 day   
#         args.cpu=True
# #         args.file_ACCESS_dir="E:/climate/access-s1/"
# #         args.file_BARRA_dir="C:/Users/JIA059/barra/"
#         args.file_DEM_dir="../DEM/"
#     else:
#         args.file_ACCESS_dir_pr="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
#         args.file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
#         # training_name="temp01"
#         args.file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"

    args.channels=0
    if args.pr:
        args.channels+=1
    if args.zg:
        args.channels+=1
    if args.psl:
        args.channels+=1
    if args.tasmax:
        args.channels+=1
    if args.tasmin:
        args.channels+=1
    if args.dem:
        args.channels+=1
    access_rgb_mean= 2.9067910245780248e-05*86400
    pre_train_path="./model/save/"+args.train_name+"/last_"+str(args.channels)+".pth"
    leading_time=217
    args.leading_time_we_use=1
    args.ensemble=1


    print(access_rgb_mean)

    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s"%args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d"%args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d"%args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d"%args.batch_size)
    print("  ------------------------------")
    print("  using cpu only？ | %5d"%args.cpu)

    ############################################################################################

    train_transforms = transforms.Compose([
    #     transforms.Resize(IMG_SIZE),
    #     transforms.RandomResizedCrop(IMG_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
        transforms.ToTensor()
    #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

#     data_set=ACCESS_BARRA_v2_0(start_date,end_date,transform=train_transforms,args=args)
    data_set=ACCESS_BARRA_v2_pr_dem(start_date,end_date,transform=train_transforms,args=args)

    train_data,val_data=random_split(data_set,[int(len(data_set)*0.8),len(data_set)-int(len(data_set)*0.8)])


    print("Dataset statistics:")
    print("  ------------------------------")
    print("  total | %5d"%len(data_set))
    print("  ------------------------------")
    print("  train | %5d"%len(train_data))
    print("  ------------------------------")
    print("  val   | %5d"%len(val_data))

    ###################################################################################set a the dataLoader
    train_dataloders =DataLoader(train_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                num_workers=args.n_threads)
    val_dataloders =DataLoader(val_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                              num_workers=args.n_threads)
    ##
    def prepare( l, volatile=False):
        def _prepare(tensor):
            if args.precision == 'half': tensor = tensor.half()
            if args.precision == 'single': tensor = tensor.float()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint = utility.checkpoint(args)
    net = model.Model(args, checkpoint)
#     net.load("./model/RCAN_BIX4.pt", pre_train="./model/RCAN_BIX4.pt", resume=args.resume, cpu=True)
    net=my_model.Modify_RCAN(net,args,checkpoint)

#     net.load("./model/RCAN_BIX4.pt", pre_train="./model/RCAN_BIX4.pt", resume=args.resume, cpu=args.cpu)
    
    args.lr=0.00001
    criterion = nn.L1Loss()
    optimizer_my = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer_my, step_size=7, gamma=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_my, gamma=0.9)
    # torch.optim.lr_scheduler.MultiStepLR(optimizer_my, milestones=[20,80], gamma=0.1)
    
#     if args.resume==1:
#         print("continue last train")
#         model_checkpoint = torch.load(pre_train_path,map_location=device)
#     else:
#         print("restart train")
#         model_checkpoint = torch.load("./model/save/"+args.train_name+"/first_"+str(args.channels)+".pth",map_location=device)

#     my_net.load_state_dict(model_checkpoint['model'])
#     optimizer_my.load_state_dict(model_checkpoint['optimizer'])
#     epoch = model_checkpoint['epoch']
    
    if torch.cuda.device_count() > 1:
        write_log("Let's use"+str(torch.cuda.device_count())+"GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net,range(torch.cuda.device_count()))

    else:
        write_log("Let's use"+str(torch.cuda.device_count())+"GPUs!")

#     my_net = torch.nn.DataParallel(my_net)
    net.to(device)
    
    ##########################################################################training

    write_log("start")
    max_error=np.inf
    for e in range(args.epochs):
        #train
        scheduler.step()
        net.train()
        loss=0
        start=time.time()
        for batch, (pr,dem,hr,_,_) in enumerate(train_dataloders):
            
            start=time.time()
            pr,dem,hr= prepare([pr,dem,hr])

            optimizer_my.zero_grad()
            with torch.set_grad_enabled(True):
                sr = net(pr,dem,0)
                running_loss =criterion(sr, hr)
                running_loss.backward()
                optimizer_my.step()
                
            loss+=running_loss #.copy()?
            if batch%10==0:
                if not os.path.exists("./model/save/"+args.train_name):
                    os.mkdir("./model/save/"+args.train_name)
                state = {'model': net.state_dict(), 'optimizer': optimizer_my.state_dict(), 'epoch': e}
                torch.save(state, "./model/save/"+args.train_name+"/last.pth")
                write_log("Train done,train time cost %f s,loss: %f"%(start-time.time(),running_loss.item()  ))
            start=time.time()

        #validation
        net.eval()
        start=time.time()
        with torch.no_grad():
            eval_psnr=0
            eval_ssim=0
#             tqdm_val = tqdm(val_dataloders, ncols=80)
            for idx_img, (lr,dem,hr,_,_) in enumerate(val_dataloders):
                lr,dem,hr = prepare([lr,dem,hr])
                sr = net(lr,dem,0)
                val_loss=criterion(sr, hr)
                for ssr,hhr in zip(sr,hr):
                    eval_psnr+=compare_psnr(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() )
                    eval_ssim+=compare_ssim(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() )

        write_log("epoche: %d,time cost %f s, lr: %f, train_loss: %f,validation loss:%f "%(
                  e,
                  time.time()-start,
                  optimizer_my.state_dict()['param_groups'][0]['lr'],
                  loss.item()/len(train_data),
                  val_loss
             ))
#         print("epoche: %d,time cost %f s, lr: %f, train_loss: %f,validation loss:%f "%(
#                   e,
#                   time.time()-start,
#                   optimizer_my.state_dict()['param_groups'][0]['lr'],
#                   loss.item()/len(train_data),
#                   val_loss
#              ))

        if running_loss<max_error:
            max_error=running_loss
    #         torch.save(net,train_loss"_"+str(e)+".pkl")
            if not os.path.exists("./model/save/"+args.train_name+"/"):
                os.mkdir("./model/save/"+args.train_name+"/")
            write_log("saving")
            state = {'model': net.state_dict(), 'optimizer': optimizer_my.state_dict(), 'epoch': e}
            torch.save(state, "./model/save/temp01/"+str(e)+".pth")
            
        scheduler.step()


            
if __name__=='__main__':
    main()
            




