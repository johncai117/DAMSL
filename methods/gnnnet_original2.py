import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
from torch.autograd import Variable
import backbone
import copy
from backbone_original import SimpleBlock_New

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class GnnNet(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support):
    super(GnnNet, self).__init__(model_func, n_way, n_support)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()
    self.first = True

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 64), nn.BatchNorm1d(64, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 64), backbone.BatchNorm1d_fw(64, track_running_stats=False))
    self.fc_new = nn.Sequential(nn.Linear(n_way, 64), nn.BatchNorm1d(64, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(n_way, 64), backbone.BatchNorm1d_fw(64, track_running_stats=False))
    self.additional_block = SimpleBlock_New(self.feat_dim, 5, half_res = True, last = True)
    self.gnn = GNN_nl(64 + self.n_way, 48, self.n_way)
    self.method = 'GnnNet'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.support_label = self.support_label.cuda()
    self.additional_block.cuda()
    self.fc_new.cuda()

    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores

  def train_loop_finetune(self, epoch, train_loader, optimizer ):
        print_freq = 10
        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss_finetune( x )
            loss.backward()
            optimizer.step()
            self.MAML_update()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

  def MAML_update(self):
    names = []
    for name, param in self.feature.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-9]
    if not self.first:
      for (name, param), (name1, param1), (name2, param2) in zip(self.additional_block.named_parameters(), self.additional_block2.named_parameters(), self.additional_block3.named_parameters()):
        if name not in names_sub:
          dat_change = param2.data - param1.data ### Y - X
          new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
          param.data.copy_(new_dat)

  
  def set_forward_finetune(self,x,is_feature=False):
    x = x.cuda()

    
    # get feature using encoder
    batch_size = 4
    support_size = self.n_way * self.n_support 

    for name, param  in self.additional_block.named_parameters():
      param.requires_grad = True
    
    for name, param  in self.feature.named_parameters():
      param.requires_grad = True

    x_var = Variable(x)
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)

    #print(y_a_i)
    self.MAML_update() ## call MAML update
    
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) # (25, 3, 224, 224)
    block_network = copy.deepcopy(self.additional_block)
    feat_network = copy.deepcopy(self.feature)
    block_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, block_network.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)

    for name, param in feat_network.named_parameters():
        param.requires_grad = False    
  
      
    total_epoch = 15
    block_network.train()
    feat_network.train()

    block_network.cuda()
    feat_network.cuda()

    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)

          for j in range(0, support_size, batch_size):
              
              block_opt.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
              
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = feat_network(z_batch)
              scores  = block_network(output)
              loss = loss_fn(scores, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss.backward() ### think about how to compute gradients and achieve a good initialization

              block_opt.step()
    

    #feat_network.eval() ## fix this
    #classifier.eval()
    #self.train() ## continue training this!
    if self.first == True:
      self.first = False
    self.additional_block2 = copy.deepcopy(self.additional_block)
    self.additional_block3 = copy.deepcopy(block_network) ## before the new state_dict is copied over
    self.additional_block.load_state_dict(block_network.state_dict())
    
    for name, param  in self.feature.named_parameters():
        param.requires_grad = True
    
    output_support = self.additional_block(self.feature(x_a_i.cuda())).view(self.n_way, self.n_support, -1)
    output_query = self.additional_block(self.feature(x_b_i.cuda())).view(self.n_way,self.n_query,-1)

    final = torch.cat((output_support, output_query), dim =1).cuda()
    #print(x.size(1))
    #print(x.shape)
    assert(final.size(1) == self.n_support + 16) ##16 query samples in each batch
    z = self.fc_new(final.view(-1, *final.size()[2:]))
    z = z.view(self.n_way, -1, z.size(1))

    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    
    scores = self.forward_gnn(z_stack)
    
    return scores

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return loss

  def set_forward_loss_finetune(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward_finetune(x)
    loss = self.loss_fn(scores, y_query)
    return loss
