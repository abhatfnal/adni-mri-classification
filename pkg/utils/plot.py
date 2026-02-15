import matplotlib.pyplot as plt

def plot_metrics(filepath, metrics):

    # Create figure
    plt.figure()

    # Plot curves 
    for k,v in metrics.items():
        plt.plot([i for i in range(1, len(v)+1)], v, label=k)

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Metrics')
    plt.legend()

    # Save and close
    plt.savefig(filepath)
    plt.close()