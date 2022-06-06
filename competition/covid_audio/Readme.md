## Following Code from
https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5


## Pipeline

Build Dataset
    - Load Data
    - Rechannel
    - Fix Length
    - rechannel
    - pad_trunc
    - time_shift
    - spectro_gram
    - spectro_augment

Build DataLoader
    - Train : Valid = 8 : 2 (Stratified)
    - batch_size : 16

Model
