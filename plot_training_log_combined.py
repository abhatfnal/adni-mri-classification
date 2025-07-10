import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find all training log CSVs
log_files = sorted(glob.glob('training_log*.csv'))

plt.figure(figsize=(10, 6))

for log_file in log_files:
    df = pd.read_csv(log_file)
    # X axis: epoch column if present, else index
    if 'epoch' in df.columns:
        x = df['epoch']
    else:
        x = df.index + 1

    # Derive label base by stripping the 'training_log_' prefix
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    label_base = base_name.replace('training_log_', '')

    # Plot training and validation loss
    plt.plot(x, df['train_loss'], linestyle='--', alpha=0.7, label=f'{label_base} Train', color='black')
    plt.plot(x, df['val_loss'], linestyle=':', alpha=0.7, label=f'{label_base} Val')

    # Determine smoothed validation loss
    if 'val_loss_smooth' in df.columns:
        smooth = df['val_loss_smooth']
    elif 'smooth' in df.columns:
        smooth = df['smooth']
    else:
        smooth = df['val_loss'].rolling(window=5, min_periods=1).mean()

    plt.plot(x, smooth, linewidth=2, label=f'{label_base} Val (smoothed)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of All Training Campaigns')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('all_training_campaigns.png', dpi=300)
print("✅ Plot saved as all_training_campaigns.png")
