"""Metrics for model_0-00."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from src.inpainting_mobidb.utils import load_data, OHE_to_seq


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
plt.savefig('out/metrics_loss_model0-00.png')
plt.close()

plt.plot(metrics_df['train accuracy'], label='training accuracy')
plt.plot(metrics_df['valid accuracy'], label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('out/metrics_accuracy_model0-00.png')
plt.close()

model = tf.keras.models.load_model('out/generator_model.h5')
valid_seq_path = '../split_data/out/validation_seqs.fasta'
valid_label_path = '../split_data/out/validation_labels.fasta'

alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_count_valid = {aa: 0 for aa in alphabet}
aa_count_generated = {aa: 0 for aa in alphabet}
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}
idx2sym = {idx: sym for idx, sym in enumerate(alphabet)}

OHE_valid, _ = load_data(valid_seq_path, valid_label_path, sym2idx)
seq_valid = OHE_to_seq(OHE_valid, idx2sym)
for aa in seq_valid:
    aa_count_valid[aa] += 1


generated_data = model.predict(OHE_valid, batch_size=30)
seq_generated = []
for data in generated_data:
    seq = OHE_to_seq(data, idx2sym)
    seq_generated.append(seq)

for aa in seq_generated:
    aa_count_generated[aa] += 1

df_aa_count_valid = pd.DataFrame(aa_count_valid)
df_aa_count_generated = pd.DataFrame(aa_count_generated)
plt.plot(df_aa_count_valid)
plt.plot(df_aa_count_generated)

