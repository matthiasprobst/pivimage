import pathlib
import unittest

import pivtestdata as ptd

import pivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestMetadata(unittest.TestCase):

    def setUp(self) -> None:
        pc_1B = ptd.piv_challenge.pc_1B
        sample_folder_pc_1B = pc_1B.download()
        self.filenames = sorted(sample_folder_pc_1B.glob('*.tif*'))

    def test_dict(self):
        pivimg = pivimage.PIVImage(self.filenames[0],
                                   meta={'name': 'test image'})
        self.assertIsInstance(pivimg.meta, pivimage.meta.Metadata)
        pivimg.meta.save(filename='test.json')
        filename = pivimg.meta.save()
        self.assertTrue(filename.exists())
        filename.unlink(missing_ok=True)

        no_filename_pivimg = pivimage.PIVImage(None)
        with self.assertRaises(ValueError):
            filename = no_filename_pivimg.meta.save()

        pathlib.Path('test.json').unlink(missing_ok=True)
