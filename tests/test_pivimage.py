import cv2
import numpy as np
import pathlib
import pivtestdata as ptd
import unittest
import xarray as xr

import pivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestPivImage(unittest.TestCase):

    def setUp(self) -> None:
        pc_1B = ptd.piv_challenge.pc_1B
        sample_folder_pc_1B = pc_1B.download()
        self.filenames = sorted(sample_folder_pc_1B.glob('*.tif*'))

    def test_array(self):
        pivimg = pivimage.PIVImage(self.filenames[0])
        self.assertEqual(pivimg._img, None)
        self.assertIsInstance(pivimg.__array__(), np.ndarray)

        pivimg = pivimage.PIVImage(self.filenames[0])
        self.assertIsInstance(np.asarray(pivimg), np.ndarray)

        xrpivimg = pivimg.asxarray()
        self.assertIsInstance(xrpivimg[:], xr.DataArray)

    def test_PIVImage(self):
        pivimg_none = pivimage.PIVImage(None)
        self.assertEqual(pivimg_none._filename, None)
        with self.assertRaises(ValueError):
            pivimg_none.get()

        with self.assertRaises(ValueError):
            _ = pivimage.PIVImage(self.filenames[0], pco=True)
        pivimg = pivimage.PIVImage(self.filenames[0],
                                   pco=False,
                                   is_first_image=True)
        self.assertEqual(pivimg._img, None)
        self.assertIsInstance(pivimg._filename, pathlib.Path)
        self.assertEqual(pivimg._filename, self.filenames[0])
        self.assertEqual(pivimg.filename, self.filenames[0])
        ref_img = pivimage.loadimg(self.filenames[0])
        self.assertFalse(pivimg.pco)
        self.assertEqual(pivimg[:].shape, ref_img.shape)
        np.testing.assert_array_equal(pivimg[:], ref_img)
        np.testing.assert_array_equal(pivimg[0:10, 0:4:2], ref_img[0:10, 0:4:2])
        self.assertEqual(pivimg.shape, ref_img.shape)
        self.assertEqual(pivimg.ndim, ref_img.ndim)
        pivimg.clear()
        self.assertEqual(pivimg._img, None)
        self.assertEqual(pivimg.ndim, ref_img.ndim)
        pivimg.clear()
        self.assertEqual(pivimg.shape, ref_img.shape)
        pivimg.clear()
        np.testing.assert_array_equal(pivimg[0:10, 0:4:2], ref_img[0:10, 0:4:2])

        pivimg = pivimage.PIVImage.from_array(ref_img)
        self.assertEqual(pivimg._filename, None)
        self.assertEqual(pivimg.filename, None)
        self.assertEqual(pivimg[:].shape, ref_img.shape)

        with self.assertRaises(ValueError):
            _ = pivimage.PIVImage.from_array(ref_img, pco=True)
        pivimg = pivimage.PIVImage.from_array(ref_img)
        self.assertFalse(pivimg._img is None)
        np.testing.assert_array_equal(pivimg[:], ref_img[:])

        pivimg = pivimage.PIVImage.from_array(ref_img,
                                              pco=True,
                                              is_first_image=True)
        ny, nx = ref_img.shape
        np.testing.assert_array_equal(pivimg[:], ref_img[:ny // 2, :])

        pivimg = pivimage.PIVImage.from_array(ref_img,
                                              pco=True,
                                              is_first_image=False)
        ny, nx = ref_img.shape
        np.testing.assert_array_equal(pivimg[:], ref_img[ny // 2:, :])

    def test_max_min(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        self.assertEqual(pivimg.max(), ref_img.max())
        self.assertEqual(pivimg.min(), ref_img.min())

    def test_mask(self):
        for fill_value in [0, 50.4, 100]:
            pivimg = pivimage.PIVImage(self.filenames[0])
            mask = np.zeros_like(pivimg[:], dtype=bool)
            mask[0:100, 0:40] = True
            masked_pivimg = pivimg.apply_mask(mask, fill_value)
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(masked_pivimg[:], pivimg[:])
            np.testing.assert_array_equal(masked_pivimg[0:100, 0:40], int(fill_value) * np.ones((100, 40)))

            masked_pivimg2 = pivimg.apply_mask(mask, fill_value, inplace=True)
            self.assertTrue(masked_pivimg2 is pivimg)

    def test_sub(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        with self.assertRaises(TypeError):
            _ = pivimg - 1
        new_pivimg = pivimg - ref_img
        np.testing.assert_array_equal(new_pivimg._img, np.zeros_like(ref_img))

    def test_arraywrapper(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        self.assertIsInstance(np.asarray(pivimg), np.ndarray)

    def test_grayscale(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg_gray = pivimg.grayscale()
        self.assertFalse(pivimg is pivimg_gray)

        rgb = np.random.randint(255, size=(100, 50, 3), dtype=np.uint8)
        pivimg = pivimage.PIVImage.from_array(rgb)
        pivimg_gray = pivimg.grayscale()
        self.assertFalse(pivimg is pivimg_gray)
        self.assertEqual(3, rgb.ndim)
        self.assertEqual(3, pivimg.ndim)
        self.assertEqual(2, pivimg_gray.ndim)
        self.assertEqual((100, 50, 3), pivimg.shape)
        self.assertEqual((100, 50), pivimg_gray.shape)

        pivimg = pivimage.PIVImage.from_array(rgb, pco=True, is_first_image=True)
        pivimg_gray = pivimg.grayscale()
        self.assertFalse(pivimg is pivimg_gray)
        self.assertEqual(3, rgb.ndim)
        self.assertEqual(3, pivimg.ndim)
        self.assertEqual(2, pivimg_gray.ndim)
        self.assertEqual((50, 50, 3), pivimg.shape)
        self.assertEqual((50, 50), pivimg_gray.shape)

    def test_smooth(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg_smooth = pivimg.smooth(kernel_size=5)
        self.assertFalse(pivimg is pivimg_smooth)
        with self.assertRaises(cv2.error):
            pivimg.smooth(kernel_size=0)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg[:], pivimg_smooth[:])

    def test_normalize(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        self.assertTrue(pivimg[:].max() > 1)
        pivimg_norm = pivimg.normalize()
        self.assertFalse(pivimg is pivimg_norm)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg_norm[:], pivimg[:])
        self.assertEqual(pivimg_norm[:].max(), 1.0)
        self.assertEqual(pivimg_norm[:].min(), 0.0)

        max_value = pivimg.max()
        assert max_value > 1
        pivimg.normalize(inplace=True)
        self.assertEqual(pivimg[:].max(), 1.0)

    def test_rot90(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg_rot = pivimg.rot90()
        self.assertFalse(pivimg is pivimg_rot)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg_rot[:], pivimg[:])
        np.testing.assert_array_equal(pivimg_rot[:], np.rot90(ref_img, 1))

        pivimg_rot2 = pivimg_rot.rot90(inplace=True)
        self.assertTrue(pivimg_rot2 is pivimg_rot)

    def test_rot180(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg_rot = pivimg.rot180()
        self.assertFalse(pivimg is pivimg_rot)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg_rot[:], pivimg[:])
        np.testing.assert_array_equal(pivimg_rot[:], np.rot90(ref_img, 2))
        pivimg_rot2 = pivimg_rot.rot180()
        np.testing.assert_array_equal(pivimg_rot2[:], pivimg[:])

        pivimg_rot2 = pivimg_rot.rot180(inplace=True)
        self.assertTrue(pivimg_rot2 is pivimg_rot)

    def test_flip(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg_fliplr = pivimg.fliplr()
        self.assertFalse(pivimg is pivimg_fliplr)
        self.assertEqual(pivimg.shape, pivimg_fliplr.shape)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg_fliplr[:], pivimg[:])
        np.testing.assert_array_equal(pivimg_fliplr[:], np.fliplr(ref_img))

        pivimg_flipud = pivimg.flipud()
        self.assertFalse(pivimg is pivimg_flipud)
        self.assertEqual(pivimg.shape, pivimg_flipud.shape)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(pivimg_flipud[:], pivimg[:])
        np.testing.assert_array_equal(pivimg_flipud[:], np.flipud(ref_img))

        pivimg_flip2 = pivimg.fliplr(inplace=True)
        self.assertTrue(pivimg_flip2 is pivimg)

    def test_to_tiff(self):
        ref_img = pivimage.loadimg(self.filenames[0])
        pivimg = pivimage.PIVImage.from_array(ref_img)
        pivimg.to_tiff('test.tif')

        pivimg2 = pivimage.loadimg('test.tif')
        np.testing.assert_array_equal(pivimg2[:], pivimg[:])
        pathlib.Path('test.tif').unlink()
