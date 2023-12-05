import matplotlib.pyplot as plt
import pandas as pd

# Sample DataFrames
df_train = pd.read_csv("models/output/UnetPlusPlus/train_logs.csv")

df_val = pd.read_csv("models/output/UnetPlusPlus/valid_logs.csv")

epochs = [i for i in range(1, 21)]

# Plotting
plt.figure(figsize=(12, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, df_train['dice_loss'], label='Training', marker='o', linestyle='-')
plt.plot(epochs, df_val['dice_loss'], label='Validation', marker='o', linestyle='-')
plt.title('Dice Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, df_train['accuracy'], label='Training', marker='o', linestyle='-')
plt.plot(epochs, df_val['accuracy'], label='Validation', marker='o', linestyle='-')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig("models/output/UnetPlusPlus/loss_acc.png")
plt.close()
