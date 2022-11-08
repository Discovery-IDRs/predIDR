"""Calculate metrics for all predictors against the MobiDB dataset."""

import src.metrics as metrics

y_label_paths = [('aucpred_seq', '../aucpreds_mobidb/out/aucpreds_labels.fasta'),
                 ('deepcnfd', '../deepcnfd_mobidb/out/deepcnfd_labels.fasta'),
                 ('espritz_profile', '../espritzp_mobidb/out/espritzp_labels.fasta'),
                 ('espritz_seq', '../espritzs_mobidb/out/espritzs_labels.fasta'),
                 ('iupred2a', '../iupred2a_mobidb/out/iupred2a_labels.fasta')]
y_score_paths = [('aucpred_seq', '../aucpreds_mobidb/out/aucpreds_scores.fasta'),
                 ('deepcnfd', '../deepcnfd_mobidb/out/deepcnfd_scores.fasta'),
                 ('espritz_profile', '../espritzp_mobidb/out/espritzp_scores.fasta'),
                 ('espritz_seq', '../espritzs_mobidb/out/espritzs_scores.fasta'),
                 ('iupred2a', '../iupred2a_mobidb/out/iupred2a_scores.fasta')]
metrics.main('../../mobidb_validation/format_seqs/out/mobidb_labels.fasta', '>([A-Z0-9]+)|',
             y_label_paths=y_label_paths, y_score_paths=y_score_paths, visual=True)
