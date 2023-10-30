import pathlib
import unittest

import numpy as np
import pivtestdata as ptd

import pivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestPivImages(unittest.TestCase):

    def setUp(self) -> None:
        pc_1B = ptd.piv_challenge.pc_1B
        sample_folder_pc_1B = pc_1B.download()
        self.filenames = sorted(sample_folder_pc_1B.glob('*.tif*'))

    def test_PIVImages(self):
        pivimgs = pivimage.PIVImages(self.filenames)
        self.assertEqual(len(pivimgs), len(self.filenames))
        self.assertEqual(pivimgs[0].filename, self.filenames[0])

    def test_background(self):
        pivimgs = pivimage.PIVImages(self.filenames)
        bg = pivimgs.compute_background(np.min)
        bg.rot180()
