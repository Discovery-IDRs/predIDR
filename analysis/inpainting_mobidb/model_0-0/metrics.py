"""Metrics for model_0-0"""

import pandas as pd
import os
import matplotlib.pyplot as plt

if not os.path.exists("out/"):
    os.mkdir("out/")

metrics_path = "out/metrics.tsv"
metrics_df = pd.read_csv(metrics_path, sep='\t')

plt.plot(metrics_df['train generator loss'], label='training generator loss')
plt.plot(metrics_df['train discriminator loss'], label='training discriminator loss')
plt.plot(metrics_df['valid generator loss'], label='validation generator loss')
plt.plot(metrics_df['valid discriminator loss'], label='validation discriminator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('out/metrics_loss_model0-0.png')
plt.close()

plt.plot(metrics_df['train accuracy'], label='training accuracy')
plt.plot(metrics_df['valid accuracy'], label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('out/metrics_accuracy_model0-0.png')
plt.close()
