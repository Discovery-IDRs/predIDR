"""Metrics for model_1-00."""

import os

import matplotlib.pyplot as plt
import pandas as pd


if not os.path.exists("out/"):
    os.mkdir("out/")

metrics_path = "out/metrics.tsv"
metrics_df = pd.read_csv(metrics_path, sep='\t')

plt.plot(metrics_df['loss'], label='training generator loss')
plt.plot(metrics_df['val_loss'], label='validation generator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('out/metrics_loss_model1-00.png')
plt.close()

plt.plot(metrics_df['categorical_accuracy'], label='training accuracy')
plt.plot(metrics_df['val_categorical_accuracy'], label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('out/metrics_accuracy_model1-00.png')
plt.close()
