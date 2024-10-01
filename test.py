import unittest
import torch
from torch import tensor
from logistic import count_pixel_percent, count_contrasting_pixels
from logistic import extract_pixel_pattern_features
from logistic import LogisticRegressionModel, LogisticRegressionParameterSpace
from logistic import precision, recall
from uncool import extract_image_features


SWISS_FLAG = tensor([[0,0,0,0,0],
                     [0,0,255,0,0],
                     [0,255,255,255,0],
                     [0,0,255,0,0],
                     [0,0,0,0,0]])

WEIRD_FLAG = tensor([[0,0,0,0,0],
                     [0,0,255,0,0],
                     [0,255,255,255,0],
                     [0,0,255,0,0],
                     [110,100,80,0,0],
                     [110,0,40,150,0],
                     [0,100,110,0,120]])   


def compare_tensors(expected, actual, precision=3):
    if expected.shape != actual.shape:
        msg = f"\nExpected tensor of shape: {expected.shape}\n"
        msg += f"Actual tensor had shape:  {actual.shape}\n"
        raise Exception(msg)
    else:
        expected = expected.round(decimals=precision)
        actual = actual.round(decimals=precision)
        if not torch.all(expected.eq(actual)):
            msg = f"\nExpected:\n{expected}\n\nActual:\n{actual}"
            raise Exception(msg)

def compare_int_tensors(expected, actual):
    if expected.shape != actual.shape:
        msg = f"\nExpected tensor of shape: {expected.shape}\n"
        msg += f"Actual tensor had shape:  {actual.shape}\n"
        raise Exception(msg)
    else:
        if not torch.all(expected.eq(actual)):
            msg = f"\nExpected:\n{expected}\n\nActual:\n{actual}"
            raise Exception(msg)

class Q1(unittest.TestCase):

    def test_feature_extractor(self):
        feats = {'bias': lambda img: 1.0,
                 'black': lambda img: count_pixel_percent(img, lambda pixel: pixel < 50),
                 'white': lambda img: count_pixel_percent(img, lambda pixel: pixel > 160),
                 'contrast': lambda img: count_contrasting_pixels(img, 2, 130)}
        image = SWISS_FLAG
        d = extract_image_features(image, feats)
        actual = {k: round(d[k], 2) for k in d}        
        expected = {'bias': 1.0, 'black': 0.8, 
                    'white': 0.2, 'contrast': 0.4} 
        self.assertEqual(expected, actual)

    def test_feature_extractor2(self):
        feats = {'bias': lambda img: 1.0,
                 'black': lambda img: count_pixel_percent(img, lambda pixel: pixel < 50),
                 'white': lambda img: count_pixel_percent(img, lambda pixel: pixel > 160),
                 'contrast': lambda img: count_contrasting_pixels(img, 2, 130)}
        image = WEIRD_FLAG
        d = extract_image_features(image, feats)
        actual = {k: round(d[k], 2) for k in d} 
        expected = {'bias': 1.0, 'black': 0.63, 
                    'white': 0.14, 'contrast': 0.33}
        self.assertEqual(expected, actual)


class Q2(unittest.TestCase):
            
    def test_lr_model(self):
        theta = torch.tensor([[0., 0., 0.], 
                              [2., 3., 2.5]])
        model = LogisticRegressionModel(theta)
        X = torch.tensor([[-1., -1., -2.,  3.],
                          [-1., -1., -1.,  4.],
                          [ 2.,  3.,  2.,  2.]])
        expected = tensor([0.5000, 0.9241, 0.1192, 1.0000])
        compare_tensors(expected, model.predict_probs(X))
        classes = model.classify(X, thres = 0.4)
        expected = tensor([1, 1, 0, 1])
        compare_int_tensors(expected, classes)
        accuracy = model.evaluate(X, torch.tensor([1, 0, 1, 0]), thres = 0.4)
        self.assertAlmostEqual(accuracy, 0.25, 2)
        accuracy = model.evaluate(X, torch.tensor([1, 0, 0, 1]), thres = 0.4)
        self.assertAlmostEqual(accuracy, 0.75, 2)

    def test_lr_model2(self):
        theta = torch.tensor([[-1., 1.2, -0.1], 
                              [0.7, 1.5,  2.1]])
        model = LogisticRegressionModel(theta)
        X = torch.tensor([[-1., -1., -2., -3.],
                          [-1., -1., -1.,  2.],
                          [ 2.,  3.,  2.,  2.]])
        expected = tensor([0.9170, 0.9900, 0.6680, 0.4750])
        compare_tensors(expected, model.predict_probs(X))
        classes = model.classify(X, thres = 0.7)
        expected = tensor([1, 1, 0, 0])
        compare_int_tensors(expected, classes)
        accuracy = model.evaluate(X, torch.tensor([1, 0, 1, 0]), thres = 0.7)
        self.assertAlmostEqual(accuracy, 0.5, 2)
        accuracy = model.evaluate(X, torch.tensor([0, 0, 1, 1]), thres = 0.7)
        self.assertAlmostEqual(accuracy, 0.0, 2)


class Q3(unittest.TestCase):

    def test_gradient(self):
        X = torch.tensor([[ 1.,  1.,  1.,  1.],
                          [-1.,  3., -2., -1.],
                          [ 2., -1.,  2.,  4.]])
        y = tensor([1, 0, 0, 1])
        pspace = LogisticRegressionParameterSpace(X, y)
        theta = tensor([[0., 1.0, 0.2], 
                        [0.5, 0., -0.3]])
        expected = tensor([[ 0.0632,  0.2775,  1.7290],
                           [-0.0632, -0.2775, -1.7290]])
        compare_tensors(expected, pspace.gradient(theta))


class Q4(unittest.TestCase):

    def test_pixel_pattern_extractor(self):
        image = SWISS_FLAG
        feats = extract_pixel_pattern_features(image, 130,
                                               [(0,0), (0,1), (1,0), (1,1)])
        expected = {'p_0000': 4, 'p_0001': 2, 'p_0010': 2, 'p_0011': 0, 
                    'p_0100': 2, 'p_0101': 0, 'p_0110': 0, 'p_0111': 1, 
                    'p_1000': 2, 'p_1001': 0, 'p_1010': 0, 'p_1011': 1, 
                    'p_1100': 0, 'p_1101': 1, 'p_1110': 1, 'p_1111': 0}        
        self.assertEqual(expected, feats)
        feats = extract_pixel_pattern_features(image, 130,
                                               [(0,0), (0,2), (2,1)])
        expected = {'p_000': 0, 'p_001': 4, 'p_010': 2, 'p_011': 0, 
                    'p_100': 2, 'p_101': 0, 'p_110': 1, 'p_111': 0}
        self.assertEqual(expected, feats)
        
    def test_pixel_pattern_extractor2(self):
        image = WEIRD_FLAG
        feats = extract_pixel_pattern_features(image, 130,
                                               [(0,0), (0,1), (1,0), (1,1)])
        expected = {'p_0000': 8, 'p_0001': 3, 'p_0010': 3, 'p_0011': 0, 
                    'p_0100': 3, 'p_0101': 0, 'p_0110': 0, 'p_0111': 1, 
                    'p_1000': 3, 'p_1001': 0, 'p_1010': 0, 'p_1011': 1, 
                    'p_1100': 0, 'p_1101': 1, 'p_1110': 1, 'p_1111': 0}                
        self.assertEqual(expected, feats)        
        feats = extract_pixel_pattern_features(image, 130,
                                               [(0,0), (0,2), (2,1)])
        expected = {'p_000': 4, 'p_001': 4, 'p_010': 3, 'p_011': 0, 
                    'p_100': 2, 'p_101': 1, 'p_110': 1, 'p_111': 0}
        self.assertEqual(expected, feats)

class Q5(unittest.TestCase):
  
    def test_pr(self):
        weights = torch.tensor([[0., 0., 0.], [2., 3., 2.5]])
        model = LogisticRegressionModel(weights)
        X = torch.tensor([[-3., -1., -1., -1., -2.],
                          [-1.,  4., -1., -1., -1.],
                          [ 2.,  3.,  2.,  3.,  2.]])
        y = torch.tensor([0, 1, 0, 0, 1])
        self.assertAlmostEqual(precision(model, X, y, thres = 0.4), 0.33, 2)
        self.assertAlmostEqual(recall(model, X, y, thres = 0.4), 0.5, 2)
        self.assertAlmostEqual(precision(model, X, y, thres = 0.8), 0.5, 2)
        self.assertAlmostEqual(recall(model, X, y, thres = 0.05), 1.0, 2)
        y = torch.tensor([1, 1, 1, 1, 1])
        self.assertAlmostEqual(precision(model, X, y, thres = 0.4), 1.0, 2)
        self.assertAlmostEqual(recall(model, X, y, thres = 0.4), 0.6, 2)
        y = torch.tensor([0, 0, 0, 0, 0])
        self.assertAlmostEqual(precision(model, X, y, thres = 0.4), 0.0, 2)
        self.assertAlmostEqual(recall(model, X, y, thres = 0.4), 1.0, 2)


if __name__ == "__main__":
    unittest.main() # run all tests