"""Functions to test metrics calculation."""

import math
import unittest

from src.metrics import *


class TestMetrics(unittest.TestCase):
    def test_get_confusion_matrix1(self):
        actual = get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]).tolist()
        expected = [[2, 3], [4, 1]]
        return self.assertEqual(actual, expected)

    def test_get_confusion_matrix2(self):
        actual = get_confusion_matrix([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]).tolist()
        expected = [[3, 2], [3, 5]]
        return self.assertEqual(actual, expected)

    def test_get_accuracy(self):
        actual = get_accuracy([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = (1+2)/(1+2+3+4)
        return self.assertEqual(actual, expected)

    def test_get_MCC(self):
        actual = get_MCC([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = ((1*2)-(3*4))/math.sqrt((1+3)*(1+4)*(2+3)*(2+4))
        return self.assertEqual(actual, expected)

    def test_get_sensitivity(self):
        actual = get_sensitivity([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = 1/(1+4)
        return self.assertEqual(actual, expected)

    def test_get_specificity(self):
        actual = get_specificity([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = 2/(3+2)
        return self.assertEqual(actual, expected)

    def test_get_precision(self):
        actual = get_precision([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = 1/(1+3)
        return self.assertEqual(actual, expected)

    def test_get_f1(self):
        actual = get_f1([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        precision = 1/(1+3)
        sensitivity = 1/(1+4)
        expected = 2 * (precision * sensitivity) / (precision + sensitivity)
        self.assertEqual(actual, expected)
        return

    def test_check_binary(self):
        vals = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        actual = check_binary(vals)
        expected = True
        self.assertEqual(actual, expected)

        vals = [0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        actual = check_binary(vals)
        expected = False
        self.assertEqual(actual, expected)
        return

    def test_check_inputs_valid(self):
        normalseq = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        nonbinaryseq = [0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        emptyseq = []
        shortseq = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

        # case 1: both inputs valid
        actual = check_inputs_valid(normalseq, normalseq)
        expected = True
        self.assertEqual(actual, expected)

        # case 2: inputs of different lengths
        with self.assertRaises(ValueError) as context:
            check_inputs_valid(normalseq, shortseq)
        self.assertTrue('y_true and y_pred_bin must be the same length' in str(context.exception))

        # case 3: y_true input is empty
        with self.assertRaises(ValueError) as context:
            check_inputs_valid(emptyseq, normalseq)
        self.assertTrue('y_true must be non-empty' in str(context.exception))

        # case 4: y_pred_bin input is empty
        with self.assertRaises(ValueError) as context:
            check_inputs_valid(normalseq, emptyseq)
        self.assertTrue('y_pred_bin must be non-empty' in str(context.exception))

        # case 5: y_true input is non-binary
        with self.assertRaises(ValueError) as context:
            check_inputs_valid(nonbinaryseq, normalseq)
        self.assertTrue('y_true must only contain 1s and 0s' in str(context.exception))

        # case 6: y_pred_bin input is non-binary
        with self.assertRaises(ValueError) as context:
            check_inputs_valid(normalseq, nonbinaryseq)
        self.assertTrue('y_pred_bin must only contain 1s and 0s' in str(context.exception))

    def test_get_confusion_matrix_invalid_inputs(self):
        normalseq = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        nonbinaryseq = [0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        emptyseq = []
        shortseq = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

        # case 1: inputs of different lengths
        with self.assertRaises(ValueError) as context:
            get_confusion_matrix(normalseq, shortseq)
        self.assertTrue('y_true and y_pred_bin must be the same length' in str(context.exception))

        # case 2: y_true input is empty
        with self.assertRaises(ValueError) as context:
            get_confusion_matrix(emptyseq, normalseq)
        self.assertTrue('y_true must be non-empty' in str(context.exception))

        # case 3: y_pred_bin input is empty
        with self.assertRaises(ValueError) as context:
            get_confusion_matrix(normalseq, emptyseq)
        self.assertTrue('y_pred_bin must be non-empty' in str(context.exception))

        # case 4: y_true input is non-binary
        with self.assertRaises(ValueError) as context:
            get_confusion_matrix(nonbinaryseq, normalseq)
        self.assertTrue('y_true must only contain 1s and 0s' in str(context.exception))

        # case 5: y_pred_bin input is non-binary
        with self.assertRaises(ValueError) as context:
            get_confusion_matrix(normalseq, nonbinaryseq)
        self.assertTrue('y_pred_bin must only contain 1s and 0s' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
