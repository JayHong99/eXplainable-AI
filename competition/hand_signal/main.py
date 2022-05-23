import pandas as pd
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
torch.backends.cudnn.benchmark = True
#torch.set_default_dtype(torch.float32)

from src.calculate import calculate_normalize
from src.preprocess import Preprocess
from src.model import return_ResNet
from src.training import train_model, valid_model, test_model
from src.earlystopping import EarlyStopping



root_path = Path('./data')
train_ratio = 0.75
batch_size = 256
random_seed = 20220522
num_classes = 11
num_epoch = 1000
device = 'cuda:0'
criterion = nn.CrossEntropyLoss().to(device)
decay_epoch = [20, 30]
learning_rate = 5e-2
patience = 40
model_path = 'model/best_model.ckpt'
save_path = Path('submissions/20220522_1.csv')


def main() :
    # normalize_mean, normalize_std = calculate_normalize(root_path)

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(normalize_mean, normalize_std)
                                ])

    preprocessor = Preprocess(  
                                root_path = root_path, 
                                transforms= transform, 
                                train_ratio= train_ratio, 
                                batch_size= batch_size, 
                                random_seed= random_seed
                                )

    train_loader, valid_loader, test_loader, label_dict = preprocessor.process()

    model = return_ResNet(num_classes).to(device)
    optimizer = SGD(model.parameters(), lr = learning_rate)
    scheduler = MultiStepLR(optimizer, decay_epoch, 0.5)
    early_stopping = EarlyStopping(
                                    patience = patience,
                                    verbose = False,
                                    delta = 0,
                                    path = model_path,
                                    )
    

    for epoch in range(num_epoch) : 
        model, optimizer, epoch_loss, epoch_acc = train_model(train_loader, model, criterion, optimizer, device, scheduler)
        print(f'{epoch}|\tTRAIN LOSS [{str(round(epoch_loss,3)).zfill(5)}] | ACC [{round(epoch_acc* 100,1)}]', end ='')

        model, optimizer, epoch_loss, epoch_acc = valid_model(valid_loader, model, criterion, optimizer, device, None)
        print(f'|\tVALID : LOSS [{str(round(epoch_loss,3)).zfill(5)}] | ACC [{round(epoch_acc* 100,1)}]')

        early_stopping(epoch_loss, model)
        if early_stopping.early_stop : 
            print("Early Stopping")
            break;

    # Test Phase
    reverse_label_dict = {value : key for key, value in label_dict.items()}
    model.load_state_dict(torch.load(model_path))
    outputs = test_model(model, test_loader, device)
    outputs = [reverse_label_dict.get(x) for x in outputs]
    
    sample = pd.read_csv(root_path.joinpath('sample_submission.csv'))
    sample['label'] = outputs
    sample.to_csv(save_path, index=False)



if __name__ == "__main__" : 
    main()
