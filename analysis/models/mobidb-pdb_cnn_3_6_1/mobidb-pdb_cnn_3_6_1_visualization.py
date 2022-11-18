"""Plot results of CNN training."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

df = pd.read_table('out/history.tsv')

plt.plot(df['epoch'], df['loss'], label='train')
plt.plot(df['epoch'], df['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('out/line_loss.png')
plt.close()

params = [('binary_accuracy', 'Accuracy'),
          ('recall', 'Recall'),
          ('precision', 'Precision')]
fig, axs = plt.subplots(len(params), 1, figsize=(12.8, 7.2))
for ax, (column_name, ylabel) in zip(axs, params):
    ax.plot(df['epoch'], df['binary_accuracy'], label='train')
    ax.plot(df['epoch'], df[f'val_{column_name}'], label='validation')
    ax.set_ylabel(ylabel)
axs[-1].set_xlabel('Epoch')
fig.legend([Line2D([], [], color='C0'), Line2D([], [], color='C1')], ['train', 'validation'],
           ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0))
plt.savefig('out/line_metrics.png')
plt.close()
