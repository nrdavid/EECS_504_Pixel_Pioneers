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


# Global parameters
ARCH = "UnetPlusPlus"
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
    # train the model using the specifications above
    # iter is an int for the iteration through random samples
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
    # Generate train and validation data
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
    # Put into dataloader for pass to model
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # Dice loss and training accuracy
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.Accuracy()]
    # Adam optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # setup for training and validation epochs
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
    # Save max score
    max_score = 0
    train_logs_ret = {'dice_loss': [], 'accuracy': []}
    valid_logs_ret = {'dice_loss': [], 'accuracy': []}
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        for key in train_logs.keys():
            train_logs_ret[key].append(train_logs[key])
            valid_logs_ret[key].append(valid_logs[key])
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['accuracy']:
            max_score = valid_logs['accuracy']
            torch.save(model, 'unet/best_model.pth')
            print('Model saved!')
    # return the max score and model for info and testing later
    return max_score, model, train_logs_ret, valid_logs_ret


def test(iter, best_model, viz_preds=False):
    # runs model on test set
    # iter = int for random sample test set
    # best_model is the best model saved from pefore
    # viz_preds decides whether or not to visualize our predictions
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
    loss = smp.utils.losses.DiceLoss()
    # class wise and total IoU
    metrics = [smp.utils.metrics.Accuracy(),
               smp.utils.metrics.IoU(threshold=0.5),
               smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1,2,3,4]),
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
            
            image_vis = test_dataset_vis[n][0].astype('uint8')
            image, gt_mask = test_dataset[n]
            
            gt_mask = gt_mask.squeeze()
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model.predict(x_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy()
            max_l_indices = np.argmax(pr_mask, axis=0) # get most probable
            mask = np.eye(pr_mask.shape[0])[max_l_indices]
            result = pr_mask*mask.transpose(2, 0, 1)
            pr_mask = np.where(result != 0, 1, 0) # change nonzeros to 1s for plotting from argmax
            plt.figure(figsize=(16, 5))
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
            # check to make sure we're classifying everything
            total_classified_gt, total_classified_pr = np.zeros(5, dtype=int), np.zeros(5, dtype=int)

            for img_index in range(gt_mask.shape[0]):
                # Create an RGB image where each pixel's color corresponds to its label
                rgb = np.array(mcolors.to_rgb(colors[img_index])) * 255
                total_classified_gt[img_index]= (gt_mask[img_index, :, :] == 1).sum()
                colored_image[gt_mask[img_index, :, :] == 1] = rgb.astype('uint8')
                total_classified_pr[img_index]= (pr_mask[img_index, :, :] == 1).sum()
                pr_mask_plt[pr_mask[img_index, :, :] == 1] = rgb.astype('uint8')
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
            plt.savefig(f'{OUTPUT_DIR}/test_images/test{iter}_{n}.png', dpi = 500)
            plt.close()
    # return logs for model info
    return logs

def main():
    # keep track of accuracy and log_list for report
    best_acc = 0
    log_list = []
    epochs = 20
    for i in range(5):
        max_score, model, train_logs, valid_logs = train(i, epochs)
        if i == 0:
            pd.DataFrame(train_logs).to_csv(f'{OUTPUT_DIR}/train_logs.csv')
            pd.DataFrame(valid_logs).to_csv(f'{OUTPUT_DIR}/valid_logs.csv')
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


# %%
