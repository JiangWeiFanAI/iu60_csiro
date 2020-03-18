import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch,torchvision
import numpy as np
import random

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
# from sklearn.model_selection import StratifiedShuffleSplit

# file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
# file_BARRA_dir="/g/data/ma05/BARRA_R/analysis/acum_proc"

# ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
# ensemble=[]
# for i in range(args.ensemble):
#     ensemble.append(ensemble_access[i])
    
# ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']

# leading_time=217
# leading_time_we_use=31


# init_date=date(1970, 1, 1)
# start_date=date(1990, 1, 1)
# end_date=date(1990,12,31) #if 929 is true we should substract 1 day
# dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

# domain = [111.975, 156.275, -44.525, -9.975]

# domain = [111.975, 156.275, -44.525, -9.975]

import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch,torchvision
import numpy as np

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import random

class ACCESS_BARRA_v2_1(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
        lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
        if self.args.channels==1:
            lr=np.repeat(lr,3,axis=2)
         
        if self.transform:#channel 数量需要整理！！
            if self.args.channels==1:
                return self.transform(lr),self.transform(label),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

class ACCESS_BARRA_v2_0(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
        lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)
            lr=np.concatenate((lr,self.mapping(lr_zg)),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)
            lr=np.concatenate((lr,self.mapping(lr_tasmax)),axis=2)

        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
        if self.args.channels==1:
            lr=np.repeat(lr,3,axis=2)
         
        if self.transform:#channel 数量需要整理！！
            return self.transform(lr),self.transform(label),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)


class ACCESS_BARRA_v4(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,en,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        lr=dpt.read_access_data(access_filename_pr,idx=time_leading).data[82:144,134:188]*86400
#         lr=dpt.map_aust(lr,domain=self.args.domain,xrarray=False)
        lr=np.expand_dims(dpt.interp_tensor_2d(lr,self.shape),axis=2)
        lr.dtype="float32"

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)

        if self.args.zg:
            access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading).data[:][83:145,135:188]
            lr_zg=dpt.interp_tensor_3d(lr_zg,self.shape)
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_psl=dpt.read_access_data(access_filename_psl,var_name="psl",idx=time_leading).data[82:144,134:188]
            lr_psl=dpt.interp_tensor_2d(lr_psl,self.shape)

        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmax=dpt.read_access_data(access_filename_tasmax,var_name="tasmax",idx=time_leading).data[82:144,134:188]
            lr_tasmax=dpt.interp_tensor_2d(lr_tasmax,self.shape)
            
        if self.args.tasmin:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmin=dpt.read_access_data(access_filename_tasmin,var_name="tasmin",idx=time_leading).data[82:144,134:188]
            lr_tasmin=dpt.interp_tensor_2d(lr_tasmin,self.shape)

            
#         if self.args.dem:
# #             print("add dem data")
#             lr=np.concatenate((lr,np.expand_dims(self.dem_data,axis=2)),axis=2)

            
#         print("end loading one data,time cost %f"%(time.time()-t))


        if self.transform:#channel 数量需要整理！！
            if self.args.channels==27:
                return self.transform(lr),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_zg),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            if self.args.channels==2:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))

###################################################################
class ACCESS_BARRA_v4_test(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,en,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        lr=dpt.read_access_data(access_filename_pr,idx=time_leading).data[82:144,134:188]
#         lr=dpt.map_aust(lr,domain=self.args.domain,xrarray=False)
        lr=np.expand_dims(dpt.interp_tensor_2d(lr,self.shape),axis=2)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)

        if self.args.zg:
#             access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_zg="../data/da_zg_19900101_e01.nc"

            lr_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading).data[:][83:145,135:188]
            lr_zg=dpt.interp_tensor_3d(lr_zg,self.shape)
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_psl="../data/da_psl_19900101_e01.nc"

            lr_psl=dpt.read_access_data(access_filename_psl,var_name="psl",idx=time_leading).data[82:144,134:188]
            lr_psl=dpt.interp_tensor_2d(lr_psl,self.shape)

        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_tasmax="../data/da_tasmax_19900101_e01.nc"

            lr_tasmax=dpt.read_access_data(access_filename_tasmax,var_name="tasmax",idx=time_leading).data[82:144,134:188]
            lr_tasmax=dpt.interp_tensor_2d(lr_tasmax,self.shape)
            
        if self.args.tasmin:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_tasmin="../data/da_tasmin_19900101_e01.nc"
            lr_tasmin=dpt.read_access_data(access_filename_tasmin,var_name="tasmin",idx=time_leading).data[82:144,134:188]
            lr_tasmin=dpt.interp_tensor_2d(lr_tasmin,self.shape)

            
#         if self.args.dem:
# #             print("add dem data")
#             lr=np.concatenate((lr,np.expand_dims(self.dem_data,axis=2)),axis=2)

            
#         print("end loading one data,time cost %f"%(time.time()-t))


        if self.transform:#channel 数量需要整理！！
            if self.args.channels==27:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_zg),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            elif self.args.channels==5:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            if self.args.channels==2:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

#######################################################################
class ACCESS_BARRA_v1(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_1_documention: the data we use is raw data that store at NCI
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        if args is None:
            exit(0)
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=7

        
#         if regin=="AUS":
#             self.shape=(314,403,1,1)
#             self.domain=[111.975, 156.275, -44.525, -9.975]
#         else:
#             self.shape=(768,1200,1,1)
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        _,date_for_BARRA,time_leading=self.filename_list[0]

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]        
        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        for en in ensemble:
            for date in self.dates:
                filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        path=[access_path]
                        
#                         barra_path=file_BARRA_dir+"/accum_prcp-an-spec-PT0H-BARRA_R-v1-"+((date+timedelta(i)).strftime("%Y%m%d"))
                        barra_date=date+timedelta(i)
#                         self.data_dir+date.strftime('%m')+"/accum_prcp-an-spec-PT0H-BARRA_R-v1-"\
#                         +date.strftime('%Y%m%d')+"T"+enum[i]+"Z.nc"
                        path.append(barra_date)
                        path.append(i)
#                         print(path)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        print(len(_files))
        if args.nine2nine and args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        #read_data filemame[idx]
        access_filename_pr,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        data_low=dpt.read_access_data(access_filename_pr,idx=time_leading)
        lr_raw=dpt.map_aust(data_low,domain=args.domain,xrarray=False)
        
#         domain = [train_data.lon.data.min(), train_data.lon.data.max(), train_data.lat.data.min(), train_data.lat.data.max()]
#         print(domain)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=args.domain,xrarray=False)#,domain=domain)
        lr=dpt.interp_tensor_2d(lr_raw,(78,100))
        if self.transform:#channel 数量需要整理！！
            if self.args.channels==27:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_zg),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            elif self.args.channels==5:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            if self.args.channels==2:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))
######################################################################################################################################
class ACCESS_BARRA_v3(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver.3, I extend the demention of the input data DEM.and change the domain to fit the size of dem. the shape also can be divided by 4
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,en,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        lr=dpt.read_access_data(access_filename_pr,idx=time_leading).data[82:144,134:188]
#         lr=dpt.map_aust(lr,domain=self.args.domain,xrarray=False)
        lr=np.expand_dims(dpt.interp_tensor_2d(lr,self.shape),axis=2)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)

        
        if self.args.zg:
            access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading).data[:][83:145,135:188]
#             lr_zg=dpt.map_aust(lr_zg,data_name="zg",xrarray=False)
            lr_zg=dpt.interp_tensor_3d(lr_zg,self.shape)
            lr=np.concatenate((lr,lr_zg),axis=2)
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_psl=dpt.read_access_data(access_filename_psl,var_name="psl",idx=time_leading).data[82:144,134:188]
#             lr_psl=dpt.map_aust(lr_psl,xrarray=False)
            lr_psl=dpt.interp_tensor_2d(lr_psl,self.shape)
            lr=np.concatenate((lr,np.expand_dims(lr_psl,axis=2)),axis=2)
        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmax=dpt.read_access_data(access_filename_tasmax,var_name="tasmax",idx=time_leading).data[82:144,134:188]
#             data_tasmax_aus=dpt.map_aust(data_tasmax,xrarray=False)
            lr_tasmax=dpt.interp_tensor_2d(lr_tasmax,self.shape)
            lr=np.concatenate((lr,np.expand_dims(lr_tasmax,axis=2)),axis=2)

            
        if self.args.tasmin:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmin=dpt.read_access_data(access_filename_tasmin,var_name="tasmin",idx=time_leading).data[82:144,134:188]
#             data_tasmin_aus=dpt.map_aust(data_tasmin,xrarray=False)
            lr_tasmin=dpt.interp_tensor_2d(lr_tasmin,self.shape)
            lr=np.concatenate((lr,np.expand_dims(lr_tasmin,axis=2)),axis=2)
            
        if self.args.dem:
#             print("add dem data")
            lr=np.concatenate((lr,np.expand_dims(self.dem_data,axis=2)),axis=2)

            
#         print("end loading one data,time cost %f"%(time.time()-t))


        if self.transform:#channel 数量需要整理！！
            return self.transform(lr*86400),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))



##########################################################################################################################################



class ACCESS_BARRA_v2(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_2_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver.2, I add extend the demention of the input data ,using zg etc.
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]

        
#         print("Dataset statistics:")
#         print("  ------------------------------")
#         print("  total | %5d"%len(self.filename_list))

#         print("  ------------------------------")
        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
                filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
#         t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        data_low=dpt.read_access_data(access_filename_pr,idx=time_leading)
        lr_raw=dpt.map_aust(data_low,domain=self.args.domain,xrarray=False)
        
#         domain = [train_data.lon.data.min(), train_data.lon.data.max(), train_data.lat.data.min(), train_data.lat.data.max()]
#         print(domain)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)
        lr=dpt.interp_tensor_2d(lr_raw,(78,100))
        
        if self.args.zg:
            access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            data_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading)
            data_zg_aus=map_aust(data_zg,xrarray=False)
            lr_zg=dpt.interp_tensor_3d(data_zg_aus,(78,100))
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            data_psl=dpt.read_access_data(access_filename_psl,idx=time_leading)
            data_psl_aus=map_aust(data_psl,xrarray=False)
            lr_psl=dpt.interp_tensor_2d(data_psl_aus,(78,100))
        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            data_tasmax=dpt.read_access_data(access_filename_tasmax,idx=time_leading)
            data_tasmax_aus=map_aust(data_tasmax,xrarray=False)
            lr_tasmax=dpt.interp_tensor_2d(data_tasmax_aus,(78,100))
            
        if self.args.tasmax:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            data_tasmin=dpt.read_access_data(access_filename_tasmin,idx=time_leading)
            data_tasmin_aus=map_aust(data_tasmin,xrarray=False)
            lr_tasmin=dpt.interp_tensor_2d(data_tasmin_aus,(78,100))
            
            
#         print("end loading one data,time cost %f"%(time.time()-t))

        if self.transform:#channel 数量需要整理！！
            return self.transform(lr*86400),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))





    
class ACCESS_v1(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_1_documention: the data we use is raw data that store at NCI
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=31

        
#         if regin=="AUS":
#             self.shape=(314,403,1,1)
#             self.domain=[111.975, 156.275, -44.525, -9.975]
#         else:
#             self.shape=(768,1200,1,1)
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir)
        _,date_for_BARRA,time_leading=self.filename_list[0]

#         data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
#         data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
#         self.lat=data_exp["lat"]
#         self.lon=data_exp["lon"]        
#         
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        for en in ensemble:
            for date in self.dates:
                filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        path=[access_path]
                        barra_date=date+timedelta(i)
                        path.append(barra_date)
                        path.append(i)
#                         print(path)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if args.nine2nine and args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        #read_data filemame[idx]
        access_filename,date_for_BARRA,time_leading=self.filename_list[idx]
        data_low=dpt.read_access_data(access_filename,idx=time_leading)
        lr_raw=dpt.map_aust(data_low,domain=args.domain,xrarray=False)
        
        lr=dpt.interp_tensor_2d(lr_raw,(78,100))
        
        if self.transform:
            return self.transform( np.expand_dims(lr,axis=3)*86400)
        else:
            return np.expand_dims(lr,axis=3)*86400,np.expand_dims(label,axis=3),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))

    


