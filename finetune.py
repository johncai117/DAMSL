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
import backbone as backbone
import configs
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.gnnnet import GnnNet
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods import damsl_v1
from methods import damsl_v1_proto
from methods import damsl_v2
from methods import damsl_v2_gnn
from methods import damsl_v2_proto
from methods import damsl_v2_ss_lab
from methods import damsl_v2_ss
from methods.protonet import euclidean_dist
from self_supervised_label import *

#configs.save_dir = 'logs_final_train' ##override
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, DTD_few_shot, CUB_few_shot, cifar_few_shot, caltech256_few_shot, cars_few_shot, plantae_few_shot, places_few_shot

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def finetune_classify(liz_x,y, model, state_in, save_it, linear = False, flatten = True, n_query = 15, ds= False, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    ###############################################################################################
    # load pretrained model on miniImageNet
    if "damsl_v1" in params.method or params.method == "gnnnet":
      pretrained_model = model_dict[params.model](flatten = flatten)
      state_temp = copy.deepcopy(state_in)
      state_keys = list(state_temp.keys())
      for _, key in enumerate(state_keys):
          if "feature." in key:
              newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
              state_temp[newkey] = state_temp.pop(key)
          else:
              state_temp.pop(key)
      pretrained_model.load_state_dict(state_temp)
    else:
      pretrained_model = copy.deepcopy(model.feature_baseline2)
    
    
    model = model.to(device)
    model.train()
    
    ###############################################################################################

    classifier = Classifier(model.feature_baseline.final_feat_dim, model.n_way)

    ###############################################################################################
    
    x = liz_x[0] ### non-changed one
    #print(x.shape)
    #print(n_query)
    model.n_query = n_query
    #x = x
    x_var = Variable(x)
    
    support_size = n_way * n_support 
    batch_size = 8

    y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device) # (25,)

    x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]).to(device)
    x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
    x_a_i_original = x_a_i.to(device)
    x_inn = x_var.view(n_way* (n_support + n_query), *x.size()[2:]).to(device)
    
    ### to load all the changed examples

    x_a_i = torch.cat((x_a_i, x_a_i), dim = 0) ##oversample the first one
    y_a_i = torch.cat((y_a_i, y_a_i), dim = 0)
    for x_aug in liz_x[1:]:
      #x_aug = x_aug
      x_aug = Variable(x_aug)
      x_a_aug = x_aug[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])
      y_a_aug = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).to(device)
      x_a_i = torch.cat((x_a_i, x_a_aug), dim = 0)
      y_a_i = torch.cat((y_a_i, y_a_aug.to(device)), dim = 0)
    
    ###############################################################################################
    
    #optimizer = torch.optim.Adam(model.parameters())
    names = []
    for name, param in pretrained_model.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-9] ### last Resnet block can adapt

    for name, param in pretrained_model.named_parameters():
      if name in names_sub:
        param.requires_grad = False

    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)

    loss_fn = nn.CrossEntropyLoss().to(device) ##change this code up ## dorop n way
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01) ##try it with weight_decay
    pretrained_model.to(device)
    classifier.to(device)
    ###############################################################################################
    total_epoch = params.fine_tune_epoch

    if freeze_backbone is False:
        pretrained_model.train()
    
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
            loss = loss_fn(scores, y_batch)

            #####################################
            loss.backward()

            classifier_opt.step()
            
            if freeze_backbone is False:
                delta_opt.step()

    
    #output_support = pretrained_model(x_a_i_original.to(device)).view(n_way, n_support, -1)
    #output_query = pretrained_model(x_b_i.to(device)).view(n_way,n_support+n_query,-1)

  

    if not params.ablation == "linear":
      output_all = pretrained_model(x_inn.to(device)).view(n_way, n_support + n_query, -1).detach()
      if params.method == "damsl_v1" or params.method == "damsl_v2" or "damsl_v2_ss" in params.method:
        final = classifier(output_all)
        batchnorm = model.batchnorm
        final = torch.transpose(batchnorm(torch.transpose(final, 1,2)),1,2).contiguous()
        z = model.fc2(final.view(-1, *final.size()[2:]))
        z = z.view(model.n_way, -1, z.size(1))
      elif params.method == "damsl_v2_gnn":
        z = model.fc2(output_all.view(-1, *output_all.size()[2:]))
        z = z.view(model.n_way, -1, z.size(1))
      elif params.method == "damsl_v2_proto":
        final = classifier(output_all)
        final = torch.transpose(model.batchnorm(torch.transpose(final, 1,2)),1,2).contiguous()
        #z = model.fc_deep(final.view(-1, *final.size()[2:])) ## use fc deep for deep embedding network
        #z = z.view(model.n_way, -1, z.size(1))
    else:
      output_all = pretrained_model(x_b_i.to(device)).detach()
      final = classifier(output_all)
      final = torch.nn.functional.softmax(final, dim = 1).detach()
    #z = model.fc2(final.view(-1, *final.size()[2:]))
    #z = z.view(model.n_way, -1, z.size(1))
    

    if "sbmtl" or "damsl" in params.method:
        #### copy baseline feature and instantiate classifer
        baseline_feat = copy.deepcopy(model.feature_baseline)
        classifier_baseline = Classifier(model.feature_baseline.final_feat_dim, model.n_way) ##instantiate classifier
        classifier_baseline.to(device)
    
        ### freeze layers of baseline feat
        names_b = []
        for name, param in baseline_feat.named_parameters():
            if param.requires_grad:
                #print(name)
                names.append(name)
        
        names_sub_b = names_b[:-9] ### last Resnet block can adapt

        for name, param in baseline_feat.named_parameters():
            if name in names_sub_b:
                param.requires_grad = False   

        delta_opt_b = torch.optim.Adam(filter(lambda p: p.requires_grad, baseline_feat.parameters()), lr = 0.01)
        classifier_opt_b = torch.optim.Adam(classifier_baseline.parameters(), lr = 0.01)

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size * lengt)

            for j in range(0, support_size * lengt, batch_size):
                classifier_opt_b.zero_grad()
                
                delta_opt_b.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)]).to(device)
                z_batch = x_a_i[selected_id].to(device)
                y_batch = y_a_i[selected_id] 
                #####################################

                output = baseline_feat(z_batch)
                score  = classifier_baseline(output)
                loss_b = loss_fn(score, y_batch)
                #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

                #####################################
                loss_b.backward() ### think about how to compute gradients and achieve a good initialization

                classifier_opt_b.step()
                delta_opt_b.step()
        
        #output_support_b = baseline_feat(x_a_i_original.to(device)).view(n_way, n_support, -1)
        #output_query_b = baseline_feat(x_b_i.to(device)).view(n_way,n_query,-1)

        if not params.ablation == "linear":

          output_all_b = baseline_feat(x_inn.to(device)).view(n_way, n_support + n_query, -1).detach()
          if params.method == "damsl_v1" or params.method == "damsl_v2" or "damsl_v2_ss" in params.method:
            final_b = classifier_baseline(output_all_b).detach() ##initial baseline scores
            final_b = torch.transpose(model.batchnorm2(torch.transpose(final_b, 1,2)),1,2).contiguous()
            z_b = model.fc2(final_b.view(-1, *final_b.size()[2:]))
            z_b = z_b.view(model.n_way, -1, z_b.size(1))
          elif params.method == "damsl_v2_gnn":
            z_b = model.fc2(output_all_b.view(-1, *output_all_b.size()[2:]))
            z_b = z_b.view(n_way, -1, z_b.size(1))
          elif params.method == "damsl_v2_proto":
            final_b = classifier_baseline(output_all_b).detach()
            final_b = torch.transpose(model.batchnorm2(torch.transpose(final_b, 1,2)),1,2).contiguous()
            #z_b = model.fc_deep(final_b.view(-1, *final_b.size()[2:]))
            #z_b = z_b.view(n_way, -1, z_b.size(1))

            #z = torch.cat([z, z_b], dim = 2)

            final = torch.cat([final, final_b], dim = 2)
            z = model.fc_deep(final.view(-1, *final.size()[2:])) ## use fc deep for deep embedding network
            z = z.view(n_way, -1, z.size(1))
    
            z_support = z[:,:model.n_support,:].contiguous()
            z_query = z[:,model.n_support:,:].contiguous()
            
            z_proto     = z_support.view(model.n_way, model.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
            z_query     = z_query.contiguous().view(model.n_way* model.n_query, -1 )

            dists = euclidean_dist(z_query, z_proto)
            scores = -dists
            return scores ## early return

        
        #final = torch.cat([final, final_b], dim = 2)
        elif params.ablation == "linear":
          

          output_all_b = baseline_feat(x_b_i.to(device)).detach()
          final_b = classifier_baseline(output_all_b).detach() ##initial baseline scores
          final_b = torch.nn.functional.softmax(final_b, dim = 1).detach()

          
          return final + final_b
        
        

       

        
        z = torch.cat([z, z_b], dim = 2)
        #
        #z = model.fc_new(final.view(-1, *final.size()[2:]))
        #z = z.view(model.n_way, -1, z.size(1))
    else:
        z = model.fc2(final.view(-1, *final.size()[2:]))
        z = z.view(model.n_way, -1, z.size(1))
        #z = torch.cat([z, z_b], dim = 2) ##concatenate

    if not "damsl_v2_ss" in params.method:
      #z_stack = [torch.cat([z[:, :model.n_support], z[:, model.n_support + i:model.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(n_query)]
      #score = model.forward_gnn(z_stack)
      model.start_ss()
      score = model.forward_gnn_ss(z)
    else:
      z_stack = [torch.cat([z[:, :model.n_support], z[:, model.n_support + i:model.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(n_query)]
      model.original_lab()
      score = model.forward_gnn(z_stack)
      score = torch.nn.functional.softmax(score, dim = 1).detach()
      final_class, total_indices = ss_lab_cb(score, n_way, n_query)
      model.load_pseudo_support(final_class, total_indices)

      new_score = model.forward_gnn_ss(z)
      new_score = torch.nn.functional.softmax(new_score, dim = 1).detach()
      return new_score

    


    score = torch.nn.functional.softmax(score, dim = 1).detach()

    return score










######################
if __name__=='__main__':
  np.random.seed(10)
  params = parse_args('train')

  ##################################################################
  image_size = 224
  iter_num = 200
  n_way  = 5
  pretrained_dataset = "miniImageNet"
  ds = False

  n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 



  if params.method == "gnnnet": ## damsl_v1_gnn (this is the version)
      model           = GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v1':
        model           = damsl_v1.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v1_proto':
        model           = damsl_v1_proto.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2':
        model           = damsl_v2.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_gnn':
        model           = damsl_v2_gnn.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_proto': ## proper name damsl_v2_proto
        model           = damsl_v2_proto.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_ss': ## proper name damsl_v2_proto
        model           = damsl_v2_ss.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_ss_lab': ## proper name damsl_v2_proto
        model           = damsl_v2_ss_lab.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'relationnet':
        feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
  elif params.method == "baseline":
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
  
  elif params.method == "all":
        
        model_2           = GnnNet( model_dict[params.model], **few_shot_params )
        checkpoint_dir2 = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, "gnnnet")
        
        

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

  


  if params.method != "all" and params.method != "baseline":
      checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, params.method)
      if params.train_aug:
          checkpoint_dir += '_aug'

      if not params.method in ['baseline'] :
          if params.optimization != "SGD":
            checkpoint_dir += "_" + params.optimization
          checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
      print(checkpoint_dir)
      if not params.method in ['baseline'] : 
          if params.save_iter != -1:
              modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
          else:
              modelfile   = get_best_file(checkpoint_dir)
          classifier_found = False
          if modelfile is not None:
              if torch.cuda.device_count() == 1:
                map_loc = torch.device('cuda:0')
              else:
                map_loc = device
              tmp = torch.load(modelfile, map_location = map_loc)
              state = tmp['state']
              state_keys = list(state.keys())
              for _, key in enumerate(state_keys):
                  if "feature2." in key:
                      state.pop(key)
                  if "feature3." in key:
                      state.pop(key)
                  if "classifier2." in key:
                      classifier_found = True
                      state.pop(key)
                  if "classifier3." in key:
                      classifier_found = True
                      state.pop(key)
              model.classifier = Classifier(model.feat_dim, n_way)
              model.batchnorm = nn.BatchNorm1d(5, track_running_stats=False)
              if "damsl" in params.method:
                  model.instantiate_baseline(params)
              model.load_state_dict(state)
              model.to(device)
     
  elif params.method == "all":
        
                
        if params.method == "all":
            checkpoint_dir2 += '_aug'
            if not params.method in ['baseline'] :
              checkpoint_dir2 += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
            if not params.method in ['baseline'] : 
              modelfile2   = get_assigned_file(checkpoint_dir2,600)
              
              modelfile2_o   = get_assigned_file(checkpoint_dir2,600)
                
                
                
              

              if modelfile2 is not None:
                tmp2 = torch.load(modelfile2)
                tmp2_o  = torch.load(modelfile2_o)
                state2 = tmp2['state']
                state2_o = tmp2_o['state']
                state_keys = list(state2.keys())
                for _, key in enumerate(state_keys):
                    if "feature2." in key:
                        state2.pop(key)
                    if "feature3." in key:
                        state2.pop(key)
                model_2.load_state_dict(state2)
                
                
                
                
                del tmp2
                del modelfile2

              
                
  
              

  freeze_backbone = params.freeze_backbone
  ##################################################################
  pretrained_dataset = "miniImageNet"

  ### Loading datasets below
  
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


  elif params.test_dataset == "DTD":
    print ("Loading DTD")
    datamgr             =  DTD_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "CUB":
    print ("Loading CUB")
    datamgr             =  CUB_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Caltech":
    print ("Loading Caltech")
    datamgr             =  caltech256_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "Cifar":
    print ("Loading Cifar")
    datamgr             =  cifar_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Cars":
    print ("Loading Cars")
    datamgr             =  cars_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Places":
    print ("Loading Places")
    datamgr             =  places_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "Plantae":
    print ("Loading Plantae")
    datamgr             =  plantae_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  ## uncomment code below to see if code is same across loaders
  

  #########################################################################
  
  acc_all = []
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  print (freeze_backbone)
  


  # replace finetine() with your own method
  iter_num = iter_num
  print(params.n_shot)
  #print(iter_num)
  #print(len(novel_loader))
  print(params.ablation)
  
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
      elif params.method in ["gnnnet"] or "damsl" in params.method:
        scores = finetune_classify(liz_x,y, model, state, ds = ds, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)

      n_way = 5
      n_query = 15

      y_query = np.repeat(range(n_way ), n_query )

      topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      
      top1_correct = np.sum(topk_ind[:,0] == y_query)
      correct_this, count_this = float(top1_correct), len(y_query)
      acc_all.append((correct_this/ count_this *100))
      if idx % 1 == 0:
        print(idx)
        print(correct_this/ count_this *100)
        #print(sum(acc_all) / len(acc_all))
  
      ###############################################################################################

  acc_all  = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std  = np.std(acc_all)
  print(params.test_dataset)
  
  print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
  print(params.model)
