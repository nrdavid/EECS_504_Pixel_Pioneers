import torch
import numpy as np
import segmentation_models_pytorch as smp
from dataset import Dataset
import segmentation_models_pytorch.utils
from torch.utils.data import DataLoader
import augmentations as aug
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd


#ITER = 0
ARCH = "Unet"
ENCODER = "vgg19"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["Matrix", "Austenite", "Martensite/Austenite", "Precipitate", "Defect"]
DEVICE = 'cuda'
ACTIVATION = 'softmax2d'
TRAIN_FILE = f"data/MetalDAM/splits/train.txt"
TEST_FILE = f"data/MetalDAM/splits/test.txt"
VAL_FILE = f"data/MetalDAM/splits/val.txt"
OUTPUT_DIR = f"models/output/{ARCH}"
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def train(iter, epochs=1):
    # create segmentation model with pretrained encoder
    model = smp.create_model(
        arch=ARCH,
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        in_channels=3,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # generate train and validation lists
    with open(f"data/MetalDAM/splits/train{iter}.txt", 'r') as f:
        train_splits = [line.strip() for line in f]
    with open(f"data/MetalDAM/splits/val{iter}.txt", 'r') as f:
        val_splits = [line.strip() for line in f]
    x_dir = "data/MetalDAM/cropped_images"
    y_dir = "data/MetalDAM/cropped_labels"
    train_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=train_splits,
        augmentation=aug.get_training_augmentation(), 
        preprocessing=aug.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    valid_dataset = Dataset(
        x_dir, 
        y_dir,
        split_list=val_splits,
        augmentation=aug.get_validation_augmentation(), 
        preprocessing=aug.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # for batch_data, batch_targets in train_loader:
    #     print(batch_targets[:, 0, :, :])
    #     gaga
    loss = smp.utils.losses.DiceLoss()
    # metrics = [
    #     smp.metrics.iou_score(reduction="weighted", class_weights=[31.86, 58.26, 8.96, 0.24, 0.68])
    # ]
    metrics = [smp.utils.metrics.Accuracy()]
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
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['accuracy']:
            max_score = valid_logs['accuracy']
            torch.save(model, 'unet/best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
    return max_score, model


def test(iter, best_model, viz_preds=False):
    #best_model = torch.load('unet/best_model_unetplusplus.pth')
    x_dir = "data/MetalDAM/cropped_images"
    y_dir = "data/MetalDAM/cropped_labels"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    with open(f"data/MetalDAM/splits/val{iter}.txt", 'r') as f:
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
    loss = loss = smp.utils.losses.DiceLoss()
    # metrics = [
    #     smp.metrics.iou_score(reduction="weighted", class_weights=[31.86, 58.26, 8.96, 0.24, 0.68])
    # ]
    metrics = [smp.utils.metrics.Accuracy(),
               smp.utils.metrics.IoU(threshold=0.5),
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1,2,3,4]), #iou class 0
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,2,3,4]),
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,3,4]),
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,4]),
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,1,2,3])]
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    for i in range(2, 7):
        test_epoch.metrics[i].__name__=f"IoU_Class{i-2}"
    logs = test_epoch.run(test_dataloader)
    if viz_preds:
        # test dataset without transformations for image visualization
        test_dataset_vis = Dataset(
            x_dir, y_dir,
            split_list=test_splits,
            classes=CLASSES
        )
        for n in range(len(test_dataset)):
            # n = np.random.choice(len(test_dataset))
            
            image_vis = test_dataset_vis[n][0].astype('uint8')
            image, gt_mask = test_dataset[n]
            # print(image_vis.shape)
            # print(image.shape)
            
            gt_mask = gt_mask.squeeze()
            # print(gt_mask.shape)
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model.predict(x_tensor)
            unique = np.unique(pr_mask.squeeze().cpu().numpy().round().astype('uint8'))
            #print(unique)
            #pr_mask = (pr_mask.squeeze().cpu().numpy().round().astype('uint8'))
            pr_mask = pr_mask.squeeze().cpu().numpy()
            max_l_indices = np.argmax(pr_mask, axis=0)
            mask = np.eye(pr_mask.shape[0])[max_l_indices]
            result = pr_mask*mask.transpose(2, 0, 1)
            pr_mask = np.where(result != 0, 1, 0)
            #pr_mask = pr_mask.astype('uint8')

            #print("Ground Truth Mask Shape: ", gt_mask.shape)
            #print("Predicted Mask Shape: ", pr_mask.shape)
            
            # change it to a regular implementation
            plt.figure(figsize=(16, 5))
            # assign different colors in different classes, use cropped labels
            # images = (image_vis, gt_mask, pr_mask)
            plt.subplot(1, 3, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Original Image')
            plt.imshow(image_vis)
            # cropped labels, assigning
            label_colors = {
                0: 'red',    # Label 0 colored red
                1: 'green',  # Label 1 colored green
                2: 'blue',    # Label 2 colored blue
                3: 'yellow',   # Label 3 colored yellow
                4: 'magenta'   # Label 4 colored magenta
            }
            colors = ['red', 'green', 'blue', 'yellow', 'purple']
            colored_image = np.zeros((gt_mask.shape[1], gt_mask.shape[2], 3), dtype='uint8')
            pr_mask_plt = np.zeros((pr_mask.shape[1], pr_mask.shape[2], 3), dtype='uint8')
            total_classified_gt, total_classified_pr = np.zeros(5, dtype=int), np.zeros(5, dtype=int)
            pr_mask_plt[:, :, 0] = 255
            pr_mask_plt[:, :, 1] = 255
            pr_mask_plt[:, :, 2] = 255

            for img_index in range(gt_mask.shape[0]):
                # Create an RGB image where each pixel's color corresponds to its label
                rgb = np.array(mcolors.to_rgb(colors[img_index])) * 255
                total_classified_gt[img_index]= (gt_mask[img_index, :, :] == 1).sum()
                #print("gt_mask: ", (gt_mask[img_index, :, :] == 1).sum())
                colored_image[gt_mask[img_index, :, :] == 1] = rgb.astype('uint8')
                #print("pr_mask_plt: ", (pr_mask[img_index, :, :] == 1).sum())
                total_classified_pr[img_index]= (pr_mask[img_index, :, :] == 1).sum()
                pr_mask_plt[pr_mask[img_index, :, :] == 1] = rgb.astype('uint8')
            
            #print("gt_mask: ", total_classified_gt)
            #print("pr_mask: ", total_classified_pr)
            plt.subplot(1, 3, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('Ground Truth')
            plt.imshow(colored_image)
            plt.subplot(1, 3, 3)
            plt.xticks([])
            plt.yticks([])
            plt.title('Prediction')
            plt.imshow(pr_mask_plt)
            #
            plt.savefig(f'{OUTPUT_DIR}/test_images/test{iter}_{n}.png', dpi = 500)
            plt.close()
    return logs

def main():
    best_acc = 0
    log_list = []
    epochs = 20
    for i in range(5):
        max_score, model = train(i, epochs)
        if max_score > best_acc:
            best_acc = max_score
            torch.save(model, f'{OUTPUT_DIR}/best_train_model.pth')
            print('Model saved!')
        model = torch.load(f'{OUTPUT_DIR}/best_train_model.pth')
        logs = test(i, model, True)
        log_list.append(logs)
    log_df = pd.DataFrame(log_list)
    log_df.to_csv(f'{OUTPUT_DIR}/test_logs.csv')



if __name__ == "__main__":
    main()

