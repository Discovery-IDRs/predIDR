"""Functions to test metrics calculation."""

import math
import unittest

from src.metrics import *


class TestMetrics(unittest.TestCase):
    def test_get_confusion_matrix1(self):
        actual = get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
        expected = {'tp': 1, 'fp': 3, 'tn': 2, 'fn': 4}
        return self.assertEqual(actual, expected)

    def test_get_confusion_matrix2(self):
        actual = get_confusion_matrix([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
        expected = {'tp': 5, 'fp': 2, 'tn': 3, 'fn': 3}
        return self.assertEqual(actual, expected)

    def test_get_accuracy(self):
        actual = get_accuracy(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        expected = (1+2)/(1+2+3+4)
        return self.assertEqual(actual, expected)

    def test_get_MCC(self):
        actual = get_MCC(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        expected = ((1*2)-(3*4))/math.sqrt((1+3)*(1+4)*(2+3)*(2+4))
        return self.assertEqual(actual, expected)

    def test_get_sensitivity(self):
        actual = get_sensitivity(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        expected = 1/(1+4)
        return self.assertEqual(actual, expected)

    def test_get_specificity(self):
        actual = get_specificity(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        expected = 2/(3+2)
        return self.assertEqual(actual, expected)

    def test_get_precision(self):
        actual = get_precision(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        expected = 1/(1+3)
        return self.assertEqual(actual, expected)

    def test_get_F1(self):
        actual = get_F1(get_confusion_matrix([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]))
        precision = 1/(1+3)
        sensitivity = 1/(1+4)
        expected = 2 * (precision * sensitivity) / (precision + sensitivity)
        return self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
