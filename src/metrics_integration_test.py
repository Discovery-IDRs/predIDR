from src.metrics import *
import unittest

class TestMetrics(unittest.TestCase):
    def test_main_IUPRED_binary_metrics_only(self):
        y_pred_fasta_files = ["iupred_short_binary_output.fasta", "iupred_long_binary_output.fasta"]
        y_true_fasta_file = "disprot_2020_06.fasta"
        main(y_pred_fasta_files, y_true_fasta_file, by_protein=False, visual_list=[], all_visual=False)
    
    def test_main_IUPRED_binary_by_protein_metrics_only(self):
        y_pred_fasta_files = ["iupred_short_binary_output.fasta", "iupred_long_binary_output.fasta"]
        y_true_fasta_file = "disprot_2020_06.fasta"
        main(y_pred_fasta_files, y_true_fasta_file, by_protein=True, visual_list=[], all_visual=False)

    def test_main_IUPRED_Short_binary_by_protein_metrics_only(self):
        y_pred_fasta_files = ["iupred_short_binary_output.fasta"]
        y_true_fasta_file = "disprot_2020_06.fasta"
        main(y_pred_fasta_files, y_true_fasta_file, by_protein=True, visual_list=[], all_visual=False)
    
    def test_main_IUPRED_Short_binary_visuals_only(self):
        y_pred_fasta_files = ["iupred_short_binary_output.fasta"]
        y_true_fasta_file = "disprot_2020_06.fasta"
        main(y_pred_fasta_files, y_true_fasta_file, by_protein=False, visual_list=[], all_visual=True)

    def test_main_IUPRED_binary_by_protein(self):
        y_pred_fasta_files = ["iupred_short_binary_output.fasta", "iupred_long_binary_output.fasta"]
        y_true_fasta_file = "disprot_2020_06.fasta"
        main(y_pred_fasta_files, y_true_fasta_file, by_protein=True, visual_list=[], all_visual=True)



if __name__ == '__main__':
    unittest.main()