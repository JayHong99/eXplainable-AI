import torch
import numpy as np

def run_batch(phase, model, criterion, optimizer, X, label, device) : 
    X = X.to(device, non_blocking = True)
    label = label.to(device , non_blocking= True)
    output = model(X)
    loss = criterion(output, label)
    if phase == 'train' : 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (
        model,
        optimizer,
        output,
        loss
    )

def run_epoch(phase, dataloader, model, optimizer, criterion, device, scheduler) : 
    if phase == 'train' : 
        model.train()
    else : 
        model.eval()
    
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []
    with torch.set_grad_enabled(phase == 'train') : 
        for X, label in dataloader : 
            model, optimizer, output, loss = run_batch( phase,
                                                        model,
                                                        criterion,
                                                        optimizer, 
                                                        X, 
                                                        label,  
                                                        device)
            if phase == 'train' : 
                scheduler.step()
            epoch_loss += loss.item() 
            epoch_outputs.extend(output.detach().cpu().numpy())
            epoch_labels.extend(label.detach().cpu().numpy())
    return model, optimizer, epoch_loss / len(dataloader), np.array(epoch_outputs), np.array(epoch_labels)

def train_model(dataloader, model, criterion, optimizer, device, scheduler) : 
    
    model, optimizer, epoch_loss, epoch_output, epoch_label = run_epoch('train', dataloader, model, 
                                            optimizer, criterion, device, scheduler)
    epoch_acc = sum(np.argmax(epoch_output, axis = 1) == epoch_label)/ len(epoch_output)
    
    return model, optimizer, epoch_loss, epoch_acc

def valid_model(dataloader, model, criterion, optimizer, device, scheduler) : 
    model, optimizer, epoch_loss, epoch_output, epoch_label = run_epoch('valid', dataloader, model, 
                                                                        optimizer, criterion, device, None)
    epoch_acc = sum(np.argmax(epoch_output, axis = 1) == epoch_label)/ len(epoch_output)
    
    return model, optimizer, epoch_loss, epoch_acc

def test_model(model, dataloader, device) :
    outputs = []
    model.eval()
    for X in dataloader : 
        X = X.to(device)
        outputs.extend(model(X).detach().cpu().numpy())
    return np.argmax(outputs, axis = 1)