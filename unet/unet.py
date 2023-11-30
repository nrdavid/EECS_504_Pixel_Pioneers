import torch
import numpy as np
import segmentation_models_pytorch as smp
from dataset import Dataset
import segmentation_models_pytorch.utils
from torch.utils.data import DataLoader

ENCODER = "vgg19"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["Matrix", "Austenite", "Martensite/Austenite", "Precipitate", "Defect"]
DEVICE = 'cuda'

def main():
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        in_channels=3,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # generate train, validation, and test directories
    with open("data/MetalDAM/train.txt", 'r') as f:
        train_splits = [line.strip() for line in f]
    with open("data/MetalDAM/test.txt", 'r') as f:
        test_splits = [line.strip() for line in f]
    with open("data/MetalDAM/val.txt", 'r') as f:
        val_splits = [line.strip() for line in f]
    x_dir = "data/MetalDAM/cropped_images"
    y_dir = "data/MetalDAM/cropped_labels"
    train_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=train_splits,
        augmentation=None, 
        preprocessing=None,
        classes=CLASSES,
    )
    print(train_dataset[0][0].shape)
    valid_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=val_splits,
        augmentation=None, 
        preprocessing=None,
        classes=CLASSES,
    )
    print(valid_dataset[0][0].shape)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    epochs = 1
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == "__main__":
    main()

# model = smp.Unet(
#     encoder_name="vgg19",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=5,                      # model output channels (number of classes in your dataset)
# )

# preprocess_input = get_preprocessing_fn('vgg19', pretrained='imagenet')

