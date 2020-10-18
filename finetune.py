import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import copy
import backbone
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet

from methods import dampnet
from methods import dampnet_full


from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def finetune_linear(liz_x,y, state_in, save_it, linear = False, flatten = True, n_query = 15, ds= False, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    ###############################################################################################
    # load pretrained model on miniImageNet
    pretrained_model = model_dict[params.model](flatten = flatten)

     
    
    state_temp = copy.deepcopy(state_in)

    state_keys = list(state_temp.keys())



    if torch.cuda.device_count() > 1:
      for _, key in enumerate(state_keys):
          if "feature." in key and "num_batches_tracked" not in key:
              newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
              state_temp[newkey] = state_temp.pop(key)
          else:
              state_temp.pop(key)
    else:
      for _, key in enumerate(state_keys):
          if "feature." in key:
              newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
              state_temp[newkey] = state_temp.pop(key)
          else:
              state_temp.pop(key)

    
    
    ###############################################################################################

    classifier = Classifier(pretrained_model.final_feat_dim, n_way)

    ###############################################################################################
    pretrained_model.load_state_dict(state_temp)
    if torch.cuda.device_count() > 1:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        pretrained_model = DataParallelPassthrough(pretrained_model, device_ids = list(range(torch.cuda.device_count())))
        classifier = DataParallelPassthrough(classifier, device_ids = list(range(torch.cuda.device_count())))


   

    x = liz_x[0] ### non-changed one
    n_query = x.size(1) - n_support
    x = x.to(device)
    x_var = Variable(x)


    batch_size = 5
    support_size = n_way * n_support 
    
    y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device) # (25,)

    x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
    #x_inn = x_var.view(n_way* (n_support + n_query), *x.size()[2:])
    
    ### to load all the changed examples
    x_a_i = torch.cat((x_a_i, x_a_i), dim = 0) ##oversample the first one
    y_a_i = torch.cat((y_a_i, y_a_i), dim = 0)
    for x_aug in liz_x[1:]:
      x_aug = x_aug.to(device)
      x_aug = Variable(x_aug)
      x_a_aug = x_aug[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])
      y_a_aug = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) ))
      x_a_i = torch.cat((x_a_i, x_a_aug), dim = 0)
      y_a_i = torch.cat((y_a_i, y_a_aug.to(device)), dim = 0)
    
    
    #print(y_a_i)
    
    
    ###############################################################################################
    loss_fn = nn.CrossEntropyLoss().to(device)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01, weight_decay=0.001)
    
    names = []
    for name, param in pretrained_model.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:glob_num_FT_layers] 

    for name, param in pretrained_model.named_parameters():
      if name in names_sub:
        param.requires_grad = False

    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)


    
    pretrained_model.to(device)
    classifier.to(device)
    ###############################################################################################
    

    if freeze_backbone is False:
        pretrained_model.train()
    else:
        pretrained_model.eval()
    
    classifier.train()

    for epoch in range(20):
        rand_id = np.random.permutation(support_size)

        for j in range(0, support_size, batch_size):
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).to(device)
            
            z_batch = x_a_i[selected_id].to(device)
            y_batch = y_a_i[selected_id] 
            #####################################

            output = pretrained_model(z_batch)
            output = classifier(output)
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()

            classifier_opt.step()
            
            if freeze_backbone is False:
                delta_opt.step()

    pretrained_model.eval()
    classifier.eval()

    output = pretrained_model(x_b_i.to(device))
    score = classifier(output).detach()
    score = torch.nn.functional.softmax(score, dim = 1).detach()
    return score







def finetune(liz_x,y, model, state_in, save_it, linear = False, flatten = True, n_query = 15, ds= False, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    ###############################################################################################
    # load pretrained model on miniImageNet
    pretrained_model = model_dict[params.model](flatten = flatten)
    
    state_temp = copy.deepcopy(state_in)

    state_keys = list(state_temp.keys())

    ### modify key names to fit
    for _, key in enumerate(state_keys):
        if "module." in key and "feature." in key:
            newkey = key.replace("module.feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state_temp[newkey] = state_temp.pop(key)
        elif "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state_temp[newkey] = state_temp.pop(key)
        else:
            state_temp.pop(key)


    pretrained_model.load_state_dict(state_temp)

    model = model.to(device)
    
    ###############################################################################################

    classifier = Classifier(pretrained_model.final_feat_dim, n_way)

    ###############################################################################################
    if torch.cuda.device_count() > 1:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        pretrained_model = DataParallelPassthrough(pretrained_model, device_ids = list(range(torch.cuda.device_count())))
        classifier = DataParallelPassthrough(classifier, device_ids = list(range(torch.cuda.device_count())))

    pretrained_model.to(device)
    classifier.to(device)

    x = liz_x[0] ### non-changed one
    
    model.n_query = n_query
    x = x.to(device)
    x_var = Variable(x)


    batch_size = 5
    support_size = n_way * n_support 
    
    y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device) # (25,)

    x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
    x_inn = x_var.view(n_way* (n_support + n_query), *x.size()[2:])
    
    ### to load all the changed examples

    x_a_i = torch.cat((x_a_i, x_a_i), dim = 0) ##oversample the first one
    y_a_i = torch.cat((y_a_i, y_a_i), dim = 0)
    for x_aug in liz_x[1:]:
      x_aug = x_aug.to(device)
      x_aug = Variable(x_aug)
      x_a_aug = x_aug[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])
      y_a_aug = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device)
      x_a_i = torch.cat((x_a_i, x_a_aug), dim = 0)
      y_a_i = torch.cat((y_a_i, y_a_aug.to(device)), dim = 0)
    
    #print(y_a_i)
    
    
    ###############################################################################################
    loss_fn = nn.CrossEntropyLoss().to(device) ##change this code up ## dorop n way
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01, weight_decay = 0.001)
    #optimizer = torch.optim.Adam(model.parameters())
    names = []
    for name, param in pretrained_model.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
   names_sub = names[:glob_num_FT_layers] 

    for name, param in pretrained_model.named_parameters():
      if name in names_sub:
        param.requires_grad = False

    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01, weight_decay= 0.001)


    pretrained_model.to(device)
    classifier.to(device)
    ###############################################################################################
    total_epoch = params.fine_tune_epoch

    if freeze_backbone is False:
        pretrained_model.train()
    else:
        pretrained_model.eval()
    #pretrained_model_fixed = copy.deepcopy(pretrained_model)
    #pretrained_model_fixed.eval()
    classifier.train()
    lengt = len(liz_x) +1
    for epoch in range(total_epoch):
        rand_id = np.random.permutation(support_size * lengt)

        for j in range(0, support_size * lengt, batch_size):
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)]).to(device)
            
            z_batch = x_a_i[selected_id].to(device)
            y_batch = y_a_i[selected_id] 
            #####################################

            output = pretrained_model(z_batch)
            if flatten == False:
              avgpool = nn.AvgPool2d(7)
              flat = backbone.Flatten()
              output = flat(avgpool(output))
            scores  = classifier(output)
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()

            classifier_opt.step()
            
            if freeze_backbone is False:
                delta_opt.step()

    #pretrained_model.eval() ## for transduction
    classifier.eval()
    if not linear:
      #model.eval() ## evaluation mode ## comment for transduction learning
      if flatten == True:
        output_all = pretrained_model(x_inn.to(device)).view(n_way, n_support + n_query, -1).detach()
        output_query = pretrained_model(x_b_i.to(device)).view(n_way,n_query,-1)
      else:
        output_all = pretrained_model(x_inn).view(n_way, n_support + n_query, pretrained_model.final_feat_dim[0], pretrained_model.final_feat_dim[1], pretrained_model.final_feat_dim[2]).detach()
        output_query_original = pretrained_model(x_b_i.to(device))
        output_query = output_query_original.view(n_way, n_query, pretrained_model.final_feat_dim[0], pretrained_model.final_feat_dim[1], pretrained_model.final_feat_dim[2])
      model.n_query = n_query
      if ds == True:
        score = model.set_forward(output_all, is_feature = True, domain_shift = True)
      else:
        score = model.set_forward(output_all, is_feature = True)
      score = torch.nn.functional.softmax(score, dim = 1).detach()
    elif linear:
      output_query_original = pretrained_model(x_b_i.to(device))    
      if flatten == False:
        output_query_original = flat(avgpool(output_query_original))
      score = classifier(output_query_original).detach()
      
    #score = torch.nn.functional.softmax(score, dim = 1)

    #print(score.shape)

    return score


def nofinetune(x,y, model, state_in, save_it, flatten = True, n_query = 15, ds= False, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5, linear = False): 
    ###############################################################################################
    # load pretrained model on miniImageNet
    pretrained_model = model_dict[params.model](flatten = flatten)
    
    state_temp = copy.deepcopy(state_in)

    state_keys = list(state_temp.keys())
    for _, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state_temp[newkey] = state_temp.pop(key)
        else:
            state_temp.pop(key)

    model = model.to(device)
    pretrained_model.load_state_dict(state_temp)

    pretrained_model.to(device)
    n_query = x.size(1) - n_support
    x = x.to(device)
    x_var = Variable(x)

    support_size = n_way * n_support 
    
    y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device) # (25,)

    #x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
    #x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
    x_inn = x_var.view(n_way* (n_support + n_query), *x.size()[2:])
    
    if flatten == True:

      #inn = torch.cat((x_a_i.to(device), x_b_i.to(device)), dim = 0)
      output_all = pretrained_model(x_inn).view(n_way, n_support + n_query, -1).detach()
      
    else:
      output_all = pretrained_model(x_inn).view(n_way, n_support + n_query, pretrained_model.final_feat_dim[0], pretrained_model.final_feat_dim[1], pretrained_model.final_feat_dim[2]).detach()
     
      

    #output_all = torch.cat((output_support, output_query), dim =1)



    model.n_query = n_query
    
    if ds == True:
      if linear == True:
        model.train()
        score2 = model.set_forward_adaptation_full(output_all, is_feature = True)
      model.eval()
      score = model.set_forward(output_all, is_feature = True, domain_shift = True)
      model.train()
      
    else:
      if linear == True:
        model.train()
        score2 = model.set_forward_adaptation(output_all, is_feature = True)
      model.eval()
      score = model.set_forward(x_var)
      model.train()
      
      
       

    #m1 = torch.unsqueeze(torch.mean(scores, dim = 1), 1) ##predict each query example independently
    #s1 = torch.unsqueeze(torch.std(scores, dim = 1), 1)

    #scores  = (scores - m1) / s1

    #m1_2 = torch.unsqueeze(torch.mean(scores2, dim = 1), 1)
    #s1_2 = torch.unsqueeze(torch.std(scores2, dim = 1), 1)

    #scores2  = (scores2 - m1_2) / s1_2
    
    #scores = (scores + scores2)/2

    score = torch.nn.functional.softmax(score, dim = 1)
    if linear == True:
      score2 = torch.nn.functional.softmax(score2, dim = 1) /2
      out = score + score2
    else:
      out = score
    
    
    return out





######################
if __name__=='__main__':
  np.random.seed(10)
  params = parse_args('train')

  ##################################################################
  image_size = 224
  iter_num = 600
  pretrained_dataset = "miniImageNet"
  ds = False

  n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 



  if params.method in ["gnnnet", "gnnnet_maml"]:
      model           = GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'relationnet':
        feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
  
  elif params.method in ["dampnet_full_class"]:
        model           = dampnet_full_class.DampNet( model_dict[params.model], **few_shot_params)
  elif params.method == "baseline":
        checkpoint_dir_b = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, "baseline")
        if params.train_aug:
            checkpoint_dir_b += '_aug'

        if params.save_iter != -1:
            modelfile_b   = get_assigned_file(checkpoint_dir_b, 400) ### always get the 400 iteration of it
        elif params.method in ['baseline', 'baseline++'] :
            modelfile_b   = get_resume_file(checkpoint_dir_b)
        else:
            modelfile_b  = get_best_file(checkpoint_dir_b)
        
        tmp_b = torch.load(modelfile_b)
        state_b = tmp_b['state']
  
  elif params.method == "all":
        
        model_2           = GnnNet( model_dict[params.model], **few_shot_params )
        checkpoint_dir2 = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, "gnnnet")
        
        
        ### Now we load the baseline model
        checkpoint_dir_b = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, "baseline")
        if params.train_aug:
            checkpoint_dir_b += '_aug'

        if params.save_iter != -1:
            modelfile_b   = get_assigned_file(checkpoint_dir_b, 400)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile_b   = get_resume_file(checkpoint_dir_b)
        else:
            modelfile_b  = get_best_file(checkpoint_dir_b)
        
        tmp_b = torch.load(modelfile_b)
        state_b = tmp_b['state']

        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model_2 = DataParallelPassthrough(model_2, device_ids = list(range(torch.cuda.device_count())))

  


  if params.method != "all" and params.method != "baseline":
      checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, params.method)
      
      
      if params.train_aug:
          checkpoint_dir += '_aug'

      if not params.method in ['baseline'] :
          checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
          if params.fine_tune and not params.num_FT_block == 1:
            checkpoint_dir += "_" + str(params.num_FT_block)

          if params.save_iter != -1:
              modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
          else:
              modelfile   = get_best_file(checkpoint_dir)
              
          if modelfile is not None:
              tmp = torch.load(modelfile)
              state = tmp['state']
              state_keys = list(state.keys())
              for _, key in enumerate(state_keys):
                  if "feature2." in key:
                      state.pop(key)
                  if "feature3." in key:
                      state.pop(key)
              model.load_state_dict(state)
  elif params.method == "all":
        
                
        if params.method == "all":
            checkpoint_dir2 += '_aug'
            if not params.method in ['baseline'] :
              checkpoint_dir2 += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
            if not params.method in ['baseline'] : 
              modelfile2   = get_assigned_file(checkpoint_dir2,params.save_iter)
              
              modelfile2_o   = get_assigned_file(checkpoint_dir2,params.save_iter)
                

              if modelfile2 is not None:
                tmp2 = torch.load(modelfile2)
                state2 = tmp2['state']
                state_keys = list(state2.keys())
                for _, key in enumerate(state_keys):
                    if "feature2." in key:
                        state2.pop(key)
                    if "feature3." in key:
                        state2.pop(key)
                model_2.load_state_dict(state2)
                
                model_2.num_FT_block = params.num_FT_block

                if params.num_FT_block % 2 == 0:
                   glob_num_FT_layers = (-9 * math.floor(params.num_FT_block / 2))
                else:
                   glob_num_FT_layers = (-9 * math.floor(params.num_FT_block / 2)) - 6
                
                ## clear some files
                del tmp2
                del modelfile2

              
                
  
              

  freeze_backbone = params.freeze_backbone
  ##################################################################
  pretrained_dataset = "miniImageNet"

  
  if params.test_dataset == "ISIC":
    print ("Loading ISIC")
    datamgr             =  ISIC_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)

  
  elif params.test_dataset == "EuroSAT":
    print ("Loading EuroSAT")
    datamgr             =  EuroSAT_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)
  
  
  elif params.test_dataset == "CropDisease":
    print ("Loading CropDisease")
    datamgr             =  CropDisease_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)


  elif params.test_dataset == "ChestX":
    print ("Loading ChestX")
    datamgr             =  Chest_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???

  ## uncomment code below to see if code is same across loaders
  

  #########################################################################
  
  acc_all = []
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  print (freeze_backbone)
  


  # replace finetine() with your own method
  iter_num = 600
  #print(iter_num)
  #print(len(novel_loader))
  n_way = 5
  n_query = 15
 
  
  if params.method != "all":
      
    for idx, (elem) in enumerate(novel_loader):
      
      leng = len(elem)
      
      ## uncomment below assertion to check that same images are shown - the same image is selected despite using two different loaders
      ## as we set the seed to be fixed
      ## for more checking, you can save the images and print them to see if they seem like crops / data augmentation of the underlying data
      assert(torch.all(torch.eq(elem[0][0] , elem[1][0])) )
      _, y = elem[0]
      liz_x = [x for (x,y) in elem]
      #for i in range(leng):
        #if i >= 1:
          #assert(torch.all(torch.eq(elem[i][1] , elem[i-1][1])) ) ##assertion check
      
  
      if params.method == "relationnet":
        scores = nofinetune(liz_x[0], y, model, state, flatten = False, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.method == "baseline":
        scores = finetune_linear(liz_x, y, state_in = state_b, linear = True, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.method == "gnnnet":
        scores = finetune(liz_x,y, model, state, ds = ds, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
        #scores += nofinetune(liz_x[0],y, model, state, ds = ds, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)

      n_way = 5
      n_query = 15

      y_query = np.repeat(range(n_way ), n_query )
      topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      
      top1_correct = np.sum(topk_ind[:,0] == y_query)
      correct_this, count_this = float(top1_correct), len(y_query)
      print (correct_this/ count_this *100)
      acc_all.append((correct_this/ count_this *100))
  else:
    model_2.n_query = n_query
    
    model_2.to(device)
    for idx, (elem) in enumerate(novel_loader):
      leng = len(elem)
      ## uncomment below assertion to check randomness - the same image is selected despite using two different loaders
      ## as we set the seed to be fixed
      assert(torch.all(torch.eq(elem[0][0] , elem[1][0])) )
      
      _, y = elem[0]
      liz_x = [x for (x,y) in elem]
      #for i in range(leng):
        #if i >= 1:
          #assert(torch.all(torch.eq(elem[i][1] , elem[i-1][1])) ) ##assertion check
      
      ### check for ablation
      if params.ablation == "simple_FT":
        scores_out = finetune(liz_x, y, model_2, state2, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.ablation == "no_ablation":
        scores_out = finetune_linear(liz_x, y, state_in = state_b, linear = True, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
        scores_out += finetune(liz_x, y, model_2, state2, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.ablation == "ablation_NL":
        scores_out = finetune(liz_x, y, model_2, state2, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
    
      if idx == 0:
          print(params.save_iter)

      y_query = np.repeat(range(n_way ), n_query )
      topk_scores, topk_labels = scores_out.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      
      top1_correct = np.sum(topk_ind[:,0] == y_query)
      correct_this, count_this = float(top1_correct), len(y_query)
      if idx % 100 == 0:
          print(idx)
          print(correct_this/ count_this *100)
      acc_all.append((correct_this/ count_this *100))
      
      

          
          


          
          
          ###############################################################################################

  acc_all  = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std  = np.std(acc_all)
  print(params.test_dataset)
  print(params.n_shot)
  print(params.save_iter)
  print(params.fine_tune_epoch)
  print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
