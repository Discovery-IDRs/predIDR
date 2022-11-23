"""Metrics for model_0-00."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from src.inpainting_mobidb.utils import load_data, OHE_to_seq, get_context_weight


if not os.path.exists("out/"):
    os.mkdir("out/")

metrics_path = "out/metrics.tsv"
metrics_df = pd.read_csv(metrics_path, sep='\t')

plt.plot(metrics_df['loss'], label='training generator loss')
plt.plot(metrics_df['val_loss'], label='validation generator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title("Loss of Generator")
plt.savefig('out/metrics_loss_model1-00.png', dpi=600)
plt.close()

plt.plot(metrics_df['categorical_accuracy'], label='training accuracy')
plt.plot(metrics_df['val_categorical_accuracy'], label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.title("Accuracy of Generator")
plt.savefig('out/metrics_accuracy_model1-00.png', dpi=600)
plt.close()

model = tf.keras.models.load_model('out/generator_model.h5')
valid_seq_path = '../split_data/out/validation_seqs.fasta'
valid_label_path = '../split_data/out/validation_labels.fasta'

train_seq_path = '../split_data/out/train_seqs.fasta'
train_label_path = '../split_data/out/train_labels.fasta'

alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_count_valid = {aa: 0 for aa in alphabet}
aa_count_generated = {aa: 0 for aa in alphabet}
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}
idx2sym = {idx: sym for idx, sym in enumerate(alphabet)}


valid_seq, valid_label = load_data(valid_seq_path, valid_label_path, sym2idx)

valid_context, valid_weight = get_context_weight(valid_seq, valid_label)
valid_generated = model.predict(valid_context, batch_size=30)


with open("out/model_output_epoch300.txt", "w") as file:
    for i, (seq, label, generated) in enumerate(zip(valid_seq, valid_label, valid_generated)):
        seq = OHE_to_seq(seq, idx2sym)
        seqstring_seq = " ".join(seq) + '\n'

        seqstring_label = " ".join([str(l) for l in label]) + '\n'

        generated = OHE_to_seq(generated, idx2sym)
        seqstring_generated = " ".join(generated) + '\n'

        file.write(f'>{i}\n' + seqstring_label + seqstring_seq + seqstring_generated)


# making graph comparing amino acid from predicted vs actual

train_seq, train_label = load_data(train_seq_path, train_label_path,sym2idx)
train_context, train_weight = get_context_weight(train_seq, train_label)
train_generated = model.predict(train_context, batch_size=30)


counts = np.zeros((len(alphabet), len(alphabet)))
for seq, label, generated in zip(train_seq, train_label, train_generated):
    actual_seq_lst = list(OHE_to_seq(seq, idx2sym))
    generated_seq_lst = list(OHE_to_seq(generated, idx2sym))
    label_lst = label.tolist()
    for actual_aa, generated_aa, label_aa in zip(actual_seq_lst, generated_seq_lst, label_lst):
        if label_aa == 1:
            counts[sym2idx[actual_aa], sym2idx[generated_aa]] += 1


x = np.arange(len(alphabet))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, counts.sum(axis=1), width, label='true amino acid composition')
ax.bar(x + width/2, counts.sum(axis=0), width, label='predicted amino acid composition')
ax.set_xticks(x)
ax.set_xticklabels(alphabet)
fig.tight_layout()
ax.legend()
ax.set_xlabel("Amino Acid")
ax.set_ylabel("Number of Amino Acids")
plt.title("Count of Overall Amino Acids in True vs. Predicted Amino Acids")
fig.set_tight_layout(True)
plt.savefig('out/model_aa_comp.png', dpi=600)
plt.close()

y = np.arange(len(alphabet))
plt.imshow(counts)
plt.xticks(x, alphabet)
plt.xlabel("True Amino Acid Composition")
plt.ylabel("Predicted Amino Acid Composition")
plt.yticks(y, alphabet)
plt.title("Comparison of True and Predicted Pairs in Model Outputs")
plt.savefig('out/heatmap_dist.png', dpi=600)
plt.close()