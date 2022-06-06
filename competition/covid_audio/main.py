from pathlib import Path
from torchsummary import summary

from src.fix_seed import seed_everything
from src.logger import logger
from src.preprocess import preprocess
from src.model import AudioClassifier

# Setting Environment
random_seed = 2022
seed_everything(random_seed)
log_path = Path('./log')
print_logger = logger(log_path)
data_path = Path('./data')

# Hyper Parameter
valid_ratio = 0.2
batch_size = 16

# Model

def main() : 

    print_logger('Initiated')
    preprocessor = preprocess(
                                data_path = data_path,
                                valid_ratio = valid_ratio,
                                random_seed = random_seed,
                                batch_size = batch_size,
                                log_func = print_logger,
                                )
    preprocessor.set_dataset()
    train_dataloader, valid_dataloader, test_dataloader = preprocessor.set_dataloader()
    input_shape = next(iter(train_dataloader))[0].shape
    print_logger(f"\tBatch Input Shape : {input_shape}")

    model = AudioClassifier()
    summary(model, input_size = (2,64,344), device = 'cpu')



    return

if __name__ == "__main__" : 
    main()