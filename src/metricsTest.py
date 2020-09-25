
import unittest
import math
from src.metrics import *

"""Functions to test metrics calculation."""

class TestStringMethods(unittest.TestCase):

    def test_get_confusion_matrix1(self):
        seq = [1, 0, 1, 0]
        ref = [1, 1, 0, 0]
        cmatrix = get_confusion_matrix1(seq, ref)
        expected = dict([("TP", 1), ("TN", 1), ("FP", 1), ("FN", 1)])
        return self.assertEquals(cmatrix,expected)

    def test_get_confusion_matrix2(self):
        seq = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
        ref = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
        cmatrix = get_confusion_matrix2(seq, ref)
        expected = dict([("TP", 3), ("TN", 3), ("FP", 3), ("FN", 3)])
        return self.assertEquals(cmatrix, expected)

    def test_get_accuracy(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_accuracy(cmatrix)
        expected = (1+2)/(1+2+3+4)
        return self.assertEquals(actual, expected)

    def test_get_MCC(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_MCC(cmatrix)
        expected = ((1*2)-(3*4))/math.sqrt((1+3)*(1+4)*(2+3)*(2+4))
        return self.assertEquals(actual, expected)

    def test_get_sensitivity(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_sensitivity(cmatrix)
        expected = (1/(1+4))*100
        return self.assertEquals(actual, expected)

    def test_get_specificity(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_specificity(cmatrix)
        expected = (2/(3+2))*100
        return self.assertEquals(actual, expected)