import argparse

parser = argparse.ArgumentParser(description='BARRA_R and ACCESS-S!')
parser.add_argument('--args_test', type=int, default=0,
                        help='testing parameters input')
parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications

parser.add_argument('--pr', type=bool, 
                default=True,
                help='add-on pr?')

parser.add_argument('--dem', action='store_true',
                help='add-on dem?') 
parser.add_argument('--psl', action='store_true',
                help='add-on psl?') 
parser.add_argument('--zg', action='store_true',
                help='add-on zg?') 
parser.add_argument('--tasmax', action='store_true',
                help='add-on tasmax?') 
parser.add_argument('--tasmin', action='store_true',
                help='add-on tasmin?')

# parser.add_argument('--pr', type=bool, 
#                 default=True,
#                 help='add-on pr?')

# parser.add_argument('--dem', type=bool, 
#                 default=False,
#                 help='add-on dem?') 
# parser.add_argument('--psl', type=bool, 
#                 default=False,
#                 help='add-on psl?') 
# parser.add_argument('--zg', type=bool, 
#                 default=False,
#                 help='add-on zg?') 
# parser.add_argument('--tasmax', type=bool, 
#                 default=False,
#                 help='add-on tasmax?') 
# parser.add_argument('--tasmin', type=bool, 
#                 default=False,
#                 help='add-on tasmin?')

parser.add_argument('--leading_time_we_use', type=int, 
                default=7,
                help='add-on tasmin?')




parser.add_argument('--ensemble', type=int, 
                default=2,
                help='total ensambles is 11') 


parser.add_argument('--channels', type=float, 
                    default=0,
                    help='channel of data_input must') 
#[111.85, 155.875, -44.35, -9.975]
parser.add_argument('--domain', type=list, 
                    default=[112.9, 154.25, -43.7425, -9.0],
                    help='dataset directory')    


parser.add_argument('--file_ACCESS_dir', type=str, 
                    default="../data/",

                    help='dataset directory')
parser.add_argument('--file_BARRA_dir', type=str, 
                    default="../data/barra_aus/",
                    help='dataset directory')

parser.add_argument('--file_DEM_dir', type=str, 
                    default="../DEM/",
                    help='dataset directory')

parser.add_argument('--nine2nine', type=bool, 
                    default=True,
                    help='whether rainfall acculate from 9am to 9am')
parser.add_argument('--date_minus_one', type=int, 
                    default=1,
                    help='whether rainfall acculate from yesterday(1)/today(0) 9am to tody/tomorrow 9am')


parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
#     parser.add_argument('--data_train', type=str, default='BARRA_R',
#                         help='train dataset name')
#     parser.add_argument('--data_test', type=str, default='DIV2K',
#                         help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=10,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
#??????????????????????????????????????????????????
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='RCAN',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half','double'),
                    help='FP precision for test (single | half)')

# Training specifications

parser.add_argument('--train_name', type=str, default='temp01',
                    help='the trainning name of the set')
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='My_RCAN',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

# New options
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')
parser.add_argument('--degradation', type=str, default='BI',
                    help='degradation model: BI, BD')

# import platform 
# sys = platform.system()
# if sys == "Windows":
#     args = parser.parse_args(args=[])
# else:
#     args = parser.parse_args()

args = parser.parse_args()


#     template.set_template(args)

# args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False











# def set_parser():
    
    
#     parser.add_argument('--args_test', type=int, default=0,
#                         help='testing parameters input')
#     parser.add_argument('--debug', action='store_true',
#                         help='Enables debug mode')
#     parser.add_argument('--template', default='.',
#                         help='You can set various templates in option.py')

#     # Hardware specifications
#     parser.add_argument('--n_threads', type=int, default=0,
#                         help='number of threads for data loading')
#     parser.add_argument('--cpu', action='store_true',
#                         help='use cpu only')
#     parser.add_argument('--n_GPUs', type=int, default=1,
#                         help='number of GPUs')
#     parser.add_argument('--seed', type=int, default=1,
#                         help='random seed')

#     # Data specifications
#     parser.add_argument('--pr', type=bool, 
#                     default=True,
#                     help='add-on pr?')     
#     parser.add_argument('--psl', type=bool, 
#                     default=False,
#                     help='add-on psl?') 
#     parser.add_argument('--zg', type=bool, 
#                     default=False,
#                     help='add-on zg?') 
#     parser.add_argument('--tasmax', type=bool, 
#                     default=False,
#                     help='add-on tasmax?') 
#     parser.add_argument('--tasmin', type=bool, 
#                     default=False,
#                     help='add-on tasmin?')
    
#     parser.add_argument('--leading_time_we_use', type=int, 
#                     default=7,
#                     help='add-on tasmin?')
    
    
    
    
#     parser.add_argument('--ensemble', type=int, 
#                     default=2,
#                     help='total ensambles is 11') 

    
#     parser.add_argument('--channels', type=float, 
#                         default=0,
#                         help='channel of data_input must') 
    
#     parser.add_argument('--domain', type=list, 
#                         default=[111.85, 155.875, -44.35, -9.975],
#                         help='dataset directory')    
    
    
#     parser.add_argument('--file_ACCESS_dir', type=str, 
#                         default="F:/climate/access-s1/pr/daily/",
    
#                         help='dataset directory')
#     parser.add_argument('--file_BARRA_dir', type=str, 
#                         default="C:/Users/JIA059/barra/",
#                         help='dataset directory')
#     parser.add_argument('--nine2nine', type=bool, 
#                         default=True,
#                         help='whether rainfall acculate from 9am to 9am')
#     parser.add_argument('--date_minus_one', type=int, 
#                         default=1,
#                         help='whether rainfall acculate from yesterday(1)/today(0) 9am to tody/tomorrow 9am')
    
    
#     parser.add_argument('--dir_demo', type=str, default='../test',
#                         help='demo image directory')
# #     parser.add_argument('--data_train', type=str, default='BARRA_R',
# #                         help='train dataset name')
# #     parser.add_argument('--data_test', type=str, default='DIV2K',
# #                         help='test dataset name')
#     parser.add_argument('--benchmark_noise', action='store_true',
#                         help='use noisy benchmark sets')
#     parser.add_argument('--n_train', type=int, default=800,
#                         help='number of training set')
#     parser.add_argument('--n_val', type=int, default=10,
#                         help='number of validation set')
#     parser.add_argument('--offset_val', type=int, default=800,
#                         help='validation index offest')
#     parser.add_argument('--ext', type=str, default='sep',
#                         help='dataset file extension')
#     parser.add_argument('--scale', default='4',
#                         help='super resolution scale')
#     parser.add_argument('--patch_size', type=int, default=96,
#                         help='output patch size')
#     #??????????????????????????????????????????????????
#     parser.add_argument('--rgb_range', type=int, default=400,
#                         help='maximum value of RGB')
#     parser.add_argument('--n_colors', type=int, default=1,
#                         help='number of color channels to use')
#     parser.add_argument('--noise', type=str, default='.',
#                         help='Gaussian noise std.')
#     parser.add_argument('--chop', action='store_true',
#                         help='enable memory-efficient forward')

#     # Model specifications
#     parser.add_argument('--model', default='RCAN',
#                         help='model name')

#     parser.add_argument('--act', type=str, default='relu',
#                         help='activation function')
#     parser.add_argument('--pre_train', type=str, default='.',
#                         help='pre-trained model directory')
#     parser.add_argument('--extend', type=str, default='.',
#                         help='pre-trained model directory')
#     parser.add_argument('--n_resblocks', type=int, default=16,
#                         help='number of residual blocks')
#     parser.add_argument('--n_feats', type=int, default=64,
#                         help='number of feature maps')
#     parser.add_argument('--res_scale', type=float, default=1,
#                         help='residual scaling')
#     parser.add_argument('--shift_mean', default=True,
#                         help='subtract pixel mean from the input')
#     parser.add_argument('--precision', type=str, default='single',
#                         choices=('single', 'half'),
#                         help='FP precision for test (single | half)')

#     # Training specifications
    
#     parser.add_argument('--train_name', type=str, default='temp01',
#                         help='the trainning name of the set')
#     parser.add_argument('--reset', action='store_true',
#                         help='reset the training')
#     parser.add_argument('--test_every', type=int, default=1000,
#                         help='do test per every N batches')
#     parser.add_argument('--epochs', type=int, default=300,
#                         help='number of epochs to train')
#     parser.add_argument('--batch_size', type=int, default=16,
#                         help='input batch size for training')
#     parser.add_argument('--split_batch', type=int, default=1,
#                         help='split the batch into smaller chunks')
#     parser.add_argument('--self_ensemble', action='store_true',
#                         help='use self-ensemble method for test')
#     parser.add_argument('--test_only', action='store_true',
#                         help='set this option to test the model')
#     parser.add_argument('--gan_k', type=int, default=1,
#                         help='k value for adversarial loss')

#     # Optimization specifications
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help='learning rate')
#     parser.add_argument('--lr_decay', type=int, default=200,
#                         help='learning rate decay per N epochs')
#     parser.add_argument('--decay_type', type=str, default='step',
#                         help='learning rate decay type')
#     parser.add_argument('--gamma', type=float, default=0.5,
#                         help='learning rate decay factor for step decay')
#     parser.add_argument('--optimizer', default='ADAM',
#                         choices=('SGD', 'ADAM', 'RMSprop'),
#                         help='optimizer to use (SGD | ADAM | RMSprop)')
#     parser.add_argument('--momentum', type=float, default=0.9,
#                         help='SGD momentum')
#     parser.add_argument('--beta1', type=float, default=0.9,
#                         help='ADAM beta1')
#     parser.add_argument('--beta2', type=float, default=0.999,
#                         help='ADAM beta2')
#     parser.add_argument('--epsilon', type=float, default=1e-8,
#                         help='ADAM epsilon for numerical stability')
#     parser.add_argument('--weight_decay', type=float, default=0,
#                         help='weight decay')

#     # Loss specifications
#     parser.add_argument('--loss', type=str, default='1*L1',
#                         help='loss function configuration')
#     parser.add_argument('--skip_threshold', type=float, default='1e6',
#                         help='skipping batch that has large error')

#     # Log specifications
#     parser.add_argument('--save', type=str, default='RCAN',
#                         help='file name to save')
#     parser.add_argument('--load', type=str, default='.',
#                         help='file name to load')
#     parser.add_argument('--resume', type=int, default=0,
#                         help='resume from specific checkpoint')
#     parser.add_argument('--print_model', action='store_true',
#                         help='print model')
#     parser.add_argument('--save_models', action='store_true',
#                         help='save all intermediate models')
#     parser.add_argument('--print_every', type=int, default=100,
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save_results', action='store_true',
#                         help='save output results')

#     # New options
#     parser.add_argument('--n_resgroups', type=int, default=10,
#                         help='number of residual groups')
#     parser.add_argument('--reduction', type=int, default=16,
#                         help='number of feature maps reduction')
#     parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
#                         help='dataset directory for testing')
#     parser.add_argument('--testset', type=str, default='Set5',
#                         help='dataset name for testing')
#     parser.add_argument('--degradation', type=str, default='BI',
#                         help='degradation model: BI, BD')

#     args = parser.parse_args(args=[])
#     # args = parser.parse_args()
# #     template.set_template(args)

#     args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
#     if args.epochs == 0:
#         args.epochs = 1e8

#     for arg in vars(args):
#         if vars(args)[arg] == 'True':
#             vars(args)[arg] = True
#         elif vars(args)[arg] == 'False':
#             vars(args)[arg] = False
#     return args

# args=set_parser()
# # args.template.find("DDBPN")
