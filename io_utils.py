import numpy as np
import os
import glob
import argparse
import backbone as backbone

model_dict = dict(
            ResNet10_Newv3 = backbone.ResNet10_Newv3,ResNet10 = backbone.ResNet10, ResNet10_Newv2 = backbone.ResNet10_Newv2 ,ResNet10_New = backbone.ResNet10_New, ResNet10_FW = backbone.ResNet10_FW, ResNet18 = backbone.ResNet18)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='training base model')
    parser.add_argument('--test_dataset'     , default='',        help='test dataset')
    parser.add_argument('--unsupervised'     , default='',        help='unsupervised dataset')
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture') 
    parser.add_argument('--method'      , default='baseline',   help='baseline/protonet') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    parser.add_argument('--both'   , action='store_true',  help='use both tuned and untuned model ')
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning') 
    parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['miniImageNet', 'caltech256', 'DTD', 'cifar100', 'CUB'], help='pretained model to use')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='fine-tune each model before selection') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--fine_tune_epoch', default=3, type=int,help ='number of epochs to finetune')
    parser.add_argument('--gen_examples', default=17, type=int,help ='number of examples to generate (data augmentation)')
    parser.add_argument('--ablation'       , default='no_ablation',      help='set the ablation study we want to perform') 
    parser.add_argument('--num_FT_block' , default=2, type=int,  help='number of blocks to finetune')
    parser.add_argument('--load_features'   , action='store_true',  help='whether we want to load features') 
    parser.add_argument('--parallel'   , action='store_true',  help='whether we want to use data parallelerism') 
    parser.add_argument('--change_FT_dir'   , default=-1, type=int,  help='change finetune directory to load from') 
    parser.add_argument('--optimization'   , default="Adam",  help='change optimization between options during training') 
    if script == 'train':
        parser.add_argument('--fine_tune'   , action='store_true',  help='fine tuning during training ') 
        parser.add_argument('--aug_episodes', action='store_true',      help='augmentation epsidoes during fine tuning ') 
        parser.add_argument('--maml_gnn', action='store_true',      help='augmentation epsidoes during fine tuning ') 
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
        
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        #parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        #parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--unsup'  , action='store_true', help='unsupervised learning or not')
        parser.add_argument('--unsup_cluster'  , action='store_true', help='unsupervised learning with clustering or not')
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
