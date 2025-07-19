import pandas as pd
import sys
import matplotlib.pyplot as plt

# Load CSV

if len(sys.argv) <= 1:
    print("Usage: python plot_training_log.py <path>")
    exit()
    
log_file = sys.argv[1]
df = pd.read_csv(log_file)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig('loss_plot.png', dpi=300)
print("✅ Plot saved as loss_plot.png")

# Optional: Show plot (uncomment if running interactively)
# plt.show()
