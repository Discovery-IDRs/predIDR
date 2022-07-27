"""Metrics for model_0-0"""

import pandas as pd
import os
import matplotlib.pyplot as plt

if not os.path.exists("out/"):
    os.mkdir("out/")

metrics_path = "out/metrics.tsv"
metrics_df = pd.read_csv(metrics_path, sep='\t')

plt.plot(metrics_df['accuracy'], label='accuracy')
plt.plot(metrics_df['generator loss'], label='generator loss')
plt.plot(metrics_df['discriminator loss'], label='discriminator loss')
plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend()
plt.savefig('out/metrics_model0-0.png')
plt.close()