import torch
import numpy as np
import segmentation_models_pytorch as smp
from dataset import Dataset
import segmentation_models_pytorch.utils
from torch.utils.data import DataLoader
import augmentations as aug

ENCODER = "vgg19"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["Matrix", "Austenite", "Martensite/Austenite", "Precipitate", "Defect"]
DEVICE = 'cuda'

def train(epochs=1):
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        in_channels=3,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # generate train and validation lists
    with open("data/MetalDAM/train.txt", 'r') as f:
        train_splits = [line.strip() for line in f]
    with open("data/MetalDAM/val.txt", 'r') as f:
        val_splits = [line.strip() for line in f]
    x_dir = "data/MetalDAM/cropped_images"
    y_dir = "data/MetalDAM/cropped_labels"
    train_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=train_splits,
        augmentation=None, 
        preprocessing=aug.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    valid_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=val_splits,
        augmentation=None, 
        preprocessing=aug.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, shuffle=False, num_workers=4)
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
            torch.save(model, 'unet/best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


def test(viz_preds=False):
    best_model = torch.load('unet/best_model.pth')
    x_dir = "data/MetalDAM/cropped_images"
    y_dir = "data/MetalDAM/cropped_labels"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    with open("data/MetalDAM/test.txt", 'r') as f:
        test_splits = [line.strip() for line in f]
    test_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=test_splits,
        augmentation=None, 
        preprocessing=aug.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    test_dataloader = DataLoader(test_dataset)
    # evaluate model on test set
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    logs = test_epoch.run(test_dataloader)

    if viz_preds:
        # test dataset without transformations for image visualization
        test_dataset_vis = Dataset(
            x_dir, y_dir, 
            classes=CLASSES,
        )
        for i in range(5):
            n = np.random.choice(len(test_dataset))
            
            image_vis = test_dataset_vis[n][0].astype('uint8')
            image, gt_mask = test_dataset[n]
            
            gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                
            Dataset.visualize(
                image=image_vis, 
                ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )

def main():
    #train()
    test(True)


if __name__ == "__main__":
    main()

