import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchnet.meter as meter
from model import Net
import time
import copy

device='cuda' if t.cuda.is_available() else 'cpu'

def cal_F1(confusion_matrix):
    
    # a function to calculate F1 scores from confusion matrix
    
    cm_value=confusion_matrix.value()
    
    TN=cm_value[0][0]
    FP=cm_value[0][1]
    FN=cm_value[1][0]
    
    precision=TN/(TN+FN)
    recall=TN/(TN+FP)
    
    # F1 score for label '0'('negative')
    F1_score=2*precision*recall/(precision+recall)
    
    return F1_score

# a function to save some checkpoints models
def save_checkpoints(model,optimizer,epoch,iteration,path):
    
    state_dict={
        'model':model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration':iteration
    }
    
    torch.save(state_dict,path)

# define a function of validation for the use in training
def val(model,dataloader):
    
    # validation mode
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    running_loss=0.0
    dataset_size=0
    confusion_matrix=meter.ConfusionMeter(2)
    for data in dataloader:
        
        # input info:
        
        inputs, labels=data
        # inputs has batchsize number of images w 3 channels
        
        val_inputs=inputs.to(device)
        val_labels=labels.to(device)
        # Variable has also grad info
        
        with t.set_grad_enabled(False):
        
            outputs=model(val_inputs)
            
            val_loss=criterion(outputs,val_labels)
            
            dataset_size+=val_inputs.size(0)
            running_loss+=val_loss.item()*val_inputs.size(0)
            
            confusion_matrix.add(outputs.data.squeeze(),labels)
        
    epoch_loss=running_loss/dataset_size
        
    # resume the training mode for the model
    model.train()
    
    F1_score=cal_F1(confusion_matrix)
    
    return confusion_matrix, epoch_loss, F1_score

# training

def train(model, dataloader, valloader, max_epoch=10):
    
    today=time.strftime('%Y%m%d')
    if not os.path.exists(today):
        os.mkdir(today)
    
    val_F1_plot=[]
    val_loss_plot=[]
    train_F1_plot=[]
    train_loss_plot=[]
    
    best_model_wts=copy.deepcopy(model.state_dict())
    best_val_F1=0.0
    
    # define an optimizer

    criterion = nn.CrossEntropyLoss()
    LR=0.001
    optimizer=optim.Adam(model.parameters(),lr=LR)#, lr=0.001, momentum=0.9)
    
    train_lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    # train_lr_scheduler=optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
    
    confusion_matrix=meter.ConfusionMeter(2)
    
    for epoch in range(max_epoch):
    
        running_loss=0.0
        dataset_size=0
        
        since=time.time()
        
        print('Epoch {}/{}'.format(epoch+1,max_epoch))
        print('-'*10)
        
        # set the model to be in training mode
        model.train()
    
        confusion_matrix.reset()
        
        for inputs, labels in dataloader:
            
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            with t.set_grad_enabled(True):
            
                # clear the gradient
                optimizer.zero_grad()
            
                # forward & backward
                outputs=model(inputs)

                train_loss=criterion(outputs, labels)
                train_loss.backward()
            
                # update the parameters
                optimizer.step()
                
                
            # gather statistic score and visualize things
            dataset_size+=inputs.size(0)
            running_loss+=train_loss.item()*inputs.size(0)
            confusion_matrix.add(
                outputs.data.squeeze(), labels.data)
        
        
        val_cm,val_loss,val_F1=val(model,valloader)
        
        print('Validation F1 score is ',val_F1)
        val_F1_plot+=[val_F1]
        val_loss_plot+=[val_loss]
        print('Validation Loss is ',val_loss)

        # lower lr if loss is not getting down
        # lr_decay=0.95
        # lr=0.001
        # if train_loss_meter.value()[0]>previous_loss:
            # lr=lr*lr_decay
            # for param_group in optimizer.param_groups:
                # param_group['lr']=lr
                
        #update learning rate
        train_lr_scheduler.step()

        previous_loss=running_loss/dataset_size
        train_loss_plot+=[previous_loss]
        
        train_F1=cal_F1(confusion_matrix)
        train_F1_plot+=[train_F1]
        
        if val_F1>best_val_F1:
            best_val_F1=val_F1.copy()
            best_model_wts=copy.deepcopy(model.state_dict())
            prefix=today+'/'
            t.save(best_model_wts,prefix+'best_model.pth')
        
        time_elapsed=time.time()-since
        print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed//60, time_elapsed%60))
        print('Best F1 score: {:4f}'.format(best_val_F1))
        print('')
        
    #take the best model
    model.load_state_dict(best_model_wts)
        
    return model,train_F1_plot,train_loss_plot,val_F1_plot,val_loss_plot


