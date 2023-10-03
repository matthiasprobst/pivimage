import pathlib
import unittest

import pivtestdata as ptd

import pivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestPivImagePair(unittest.TestCase):

    def setUp(self) -> None:
        pc_1B = ptd.piv_challenge.pc_1B
        sample_folder_pc_1B = pc_1B.download()
        self.filenames = sorted(sample_folder_pc_1B.glob('*.tif*'))

    def test_PIVImagePair(self):
        pivpair = pivimage.PIVImagePair(self.filenames[0], self.filenames[1])
        self.assertTrue(pivpair.A._img is None)
        imgA = pivpair.A[:]
        self.assertFalse(pivpair.A._img is None)
        self.assertIsInstance(pivpair._A, pivimage.PIVImage)
        self.assertIsInstance(pivpair.A, pivimage.PIVImage)
        self.assertEqual(pivpair._A.filename, self.filenames[0])
        self.assertIsInstance(pivpair._B, pivimage.PIVImage)
        self.assertIsInstance(pivpair.B, pivimage.PIVImage)
        self.assertEqual(pivpair._B.filename, self.filenames[1])
