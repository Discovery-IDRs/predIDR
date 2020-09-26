from src.metrics import *
import unittest
import math
from src.metrics import *

"""Functions to test metrics calculation."""

class TestStringMethods(unittest.TestCase):

    def test_get_confusion_matrix1(self):
        seq = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
        ref = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        cmatrix = get_confusion_matrix1(seq, ref)
        expected = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        return self.assertEqual(cmatrix,expected)

    def test_get_confusion_matrix2(self):
        seq = [[1, 0, 1, 0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]]
        ref = [[1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]]
        cmatrix = get_confusion_matrix2(seq, ref)
        expected = dict([("TP", 3), ("TN", 6), ("FP", 9), ("FN", 12)])
        return self.assertEqual(cmatrix, expected)

    def test_get_accuracy(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_accuracy(cmatrix)
        expected = (1+2)/(1+2+3+4)
        return self.assertEqual(actual, expected)

    def test_get_MCC(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_MCC(cmatrix)
        expected = ((1*2)-(3*4))/math.sqrt((1+3)*(1+4)*(2+3)*(2+4))
        return self.assertEqual(actual, expected)

    def test_get_sensitivity(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_sensitivity(cmatrix)
        expected = (1/(1+4))
        return self.assertEqual(actual, expected)

    def test_get_specificity(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_specificity(cmatrix)
        expected = (2/(3+2))
        return self.assertEqual(actual, expected)

    def test_get_precision(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        actual = get_precision(cmatrix)
        expected = (1 / (1 + 3))
        return self.assertEqual(actual, expected)

    def test_get_f1(self):
        cmatrix = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        precision = get_precision(cmatrix)
        recall = get_sensitivity(cmatrix)
        b1 = 0.5
        actual1 = get_f1(cmatrix, b1)
        expected1 = (1 + b1 ** 2) * ((precision * recall) / (b1 ** 2 * precision + recall))
        self.assertEqual(actual1, expected1)
        b2 = 2
        actual2 = get_f1(cmatrix, b2)
        expected2 = (1 + b2 ** 2) * ((precision * recall) / (b2 ** 2 * precision + recall))
        self.assertEqual(actual2, expected2)
        return

    def test_get_confusion_matrix1_integration(self):
        seq = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
        ref = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        cmatrix = get_confusion_matrix1(seq, ref)
        expected = dict([("TP", 1), ("TN", 2), ("FP", 3), ("FN", 4)])
        self.assertEqual(cmatrix,expected)
        actualAccuracy = get_accuracy(cmatrix)
        expectedAccuracy = (1+2)/(1+2+3+4)
        self.assertEqual(actualAccuracy, expectedAccuracy)
        actualMCC = get_MCC(cmatrix)
        expectedMCC = ((1*2)-(3*4))/math.sqrt((1+3)*(1+4)*(2+3)*(2+4))
        self.assertEqual(actualMCC, expectedMCC)
        actualSensitivity = get_sensitivity(cmatrix)
        expectedSensitivity = (1/(1+4))
        self.assertEqual(actualSensitivity, expectedSensitivity)
        actualSpecificty = get_specificity(cmatrix)
        expectedSpecificity = (2/(3+2))
        self.assertEqual(actualSpecificty, expectedSpecificity)
        actualPrecision = get_precision(cmatrix)
        expectedPrecision = (1/(1+3))
        self.assertEqual(actualPrecision, expectedPrecision)
        actualF1 = get_f1(cmatrix, 0.5)
        expectedF1 = (1 + 0.5 ** 2) * ((actualPrecision * actualSensitivity) / (0.5 ** 2 * actualPrecision + actualSensitivity))
        self.assertEqual(actualF1, expectedF1)
        return

    def test_wiki_get_confusion_matrix1(self):
        seq = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        ref = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        cmatrix = get_confusion_matrix1(seq, ref)
        expected = dict([("TP", 5), ("TN", 3), ("FP", 2), ("FN", 3)])
        return self.assertEqual(cmatrix, expected)

    def test_wiki_get_confusion_matrix2(self):
        seq = [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1] , [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]]
        ref = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] , [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] , [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
        cmatrix = get_confusion_matrix2(seq, ref)
        expected = dict([("TP", 15), ("TN", 9), ("FP", 6), ("FN", 9)])
        return self.assertEqual(cmatrix, expected)
