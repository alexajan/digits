from digits import example

import unittest

class SoftmaxTestSuite(unittest.TestCase):
    """Make sure softmax on MNIST data has the expected accuracy. (0.91 <= accuracy <= 0.93)"""

    def test_softmax_digits(self):
        result = example.softmax_digits()
        assert result >= 0.91 and result <=0.93
