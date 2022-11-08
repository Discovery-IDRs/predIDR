"""Calculate metrics for all predictors against the DisProt dataset."""

import src.metrics as metrics

y_label_paths = [('aucpred_profile', '../aucpredp_disprot/out/aucpredp_labels.fasta'),
                 ('aucpred_seq', '../aucpreds_disprot/out/aucpreds_labels.fasta'),
                 ('deepcnfd', '../deepcnfd_disprot/out/deepcnfd_labels.fasta'),
                 ('espritz_profile', '../espritzp_disprot/out/espritzp_labels.fasta'),
                 ('espritz_seq', '../espritzs_disprot/out/espritzs_labels.fasta'),
                 ('iupred2a', '../iupred2a_disprot/out/iupred2a_labels.fasta'),
                 ('disopred3', '../disopred3_disprot/out/disopred3_labels.fasta')]
y_score_paths = [('aucpred_profile', '../aucpredp_disprot/out/aucpredp_scores.fasta'),
                 ('aucpred_seq', '../aucpreds_disprot/out/aucpreds_scores.fasta'),
                 ('deepcnfd', '../deepcnfd_disprot/out/deepcnfd_scores.fasta'),
                 ('espritz_profile', '../espritzp_disprot/out/espritzp_scores.fasta'),
                 ('espritz_seq', '../espritzs_disprot/out/espritzs_scores.fasta'),
                 ('iupred2a', '../iupred2a_disprot/out/iupred2a_scores.fasta'),
                 ('disopred3', '../disopred3_disprot/out/disopred3_scores.fasta')]
metrics.main('../../disprot_validation/format_seqs/out/disprot_labels.fasta', '>disprot_id:(DP[0-9]+)|',
             y_label_paths=y_label_paths, y_score_paths=y_score_paths, visual=True)
