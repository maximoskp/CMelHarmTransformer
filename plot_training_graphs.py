import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

save_path = 'results/figs/'
os.makedirs(save_path, exist_ok=True)

models = ['bart', 'gpt']

# List of CSV files
csv_files = ["ChordSymbolTokenizer.csv", "PitchClassTokenizer.csv"]
labels = [s.split('.')[0] for s in csv_files]  # Labels for the legend

# Read the epoch column from A.csv (assuming all have the same epochs)
df_A = pd.read_csv("results/bart/ChordSymbolTokenizer.csv")
epochs = df_A["epoch"]

# Metrics to plot
metrics = ["train_loss", "train_acc", "val_loss", "val_acc", "val_ppl", "val_te"]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

matplotlib.rcParams.update({'font.size': 20})

# Plot each metric
for model in models:
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        for file, label in zip(csv_files, labels):
            df = pd.read_csv('results/' + model + '/' + file)
            if metric in df.keys():
                plt.plot(epochs, df[metric], label=label)

        # plt.title(f"{metric} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

        # Save figure
        filename = f"{model}_{metric}_comparison.png"
        plt.savefig('results/figs/' + filename, dpi=300)
        print(f"Saved {filename}")

        plt.close()  # Close the figure to free memory