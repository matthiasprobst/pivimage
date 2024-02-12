import abc
import copy
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import warnings
from cv2 import imread as cv2_imread
from pathlib import Path
from pco_tools import pco_reader
from typing import Union, List, Tuple, Dict

from .file_wrapper import FileWrapper
from .meta import Metadata

keep_meta = False
logger = logging.getLogger(__package__)


class _PIVImage(abc.ABC, FileWrapper):

    def __init__(self,
                 filename,
                 meta: Union[Dict, Metadata] = None,
                 **kwargs):
        super().__init__(filename, **kwargs)
        self._img = None

        attrs = kwargs.get('attrs', None)
        if attrs is not None:
            warnings.warn(f'Parameter "attrs" is deprecated. Please use "meta" instead.')
            if meta is not None:
                raise ValueError(f'Both "meta" and "attrs" are given. In future only "meta" will be available. '
                                 f'Make sure to only pass one of it at this stage, preferably "meta".')

        if isinstance(meta, dict):
            self._meta = Metadata(self)
        else:
            self._meta = Metadata(self)

    @property
    def meta(self):
        """Return metadata object"""
        return self._meta

    @meta.setter
    def meta(self, data: Dict):
        if not isinstance(data, dict):
            raise TypeError('Only allow a dictionary')
        self._meta._data = data

    def __getitem__(self, item):
        return self.get().__getitem__(item)

    def asxarray(self):
        """Use xarray instead of numpy array. Requires xarray to be installed.
        A new object with xarray.DataArray for the iamge array is returned.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError('xarray is not installed. Please install it first.')
        ny, nx = self.shape
        new_obj = copy.deepcopy(self)
        try:
            attrs = self.meta.data
        except NotImplementedError as e:
            attrs = {}
        new_obj._img = xr.DataArray(new_obj.get(),
                                    dims=('iy', 'ix'),
                                    coords={'iy': np.arange(ny),
                                            'ix': np.arange(nx)},
                                    attrs=attrs)
        return new_obj

    def unlink(self, missing_ok: bool = True):
        """Deletes the image file and the meta file (if exists/associated)"""
        self.filename.unlink(missing_ok=missing_ok)
        self.meta.unlink(missing_ok=missing_ok)

    @property
    def filename(self):
        """Return filename"""
        return self._filename

    def __array__(self) -> np.ndarray:
        """returns numpy array if np.asarray() or e.g. np.roll() is called on this object"""
        return np.asarray(self.get())

    @classmethod
    @abc.abstractmethod
    def from_array(cls, img: np.ndarray, meta: Union[Dict, Metadata] = None):
        """Initialize from array"""

    @property
    def ndim(self):
        """Return ndim of array"""
        return self.get().ndim

    @property
    def dtype(self):
        return self.get().dtype

    @property
    def shape(self):
        """Return shape of array"""
        return self.get().shape

    def clear(self):
        """free the (RAM), aka unset _img"""
        self._img = None

    def get(self) -> np.ndarray:
        """Return the image. If not yet loaded (self._img is None), load it."""
        if self._img is None:
            if self._filename is None:
                raise ValueError('No filename set!')
            self._img = loadimg(self._filename)
        return self._img

    def max(self):
        """Return maximum value of the image"""
        return self.get().max()

    def min(self):
        """Return minimum value of the image"""
        return self.get().min()

    def mean(self, *args, **kwargs):
        """Return the mean value of the image"""
        return self.get().mean(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Return the median value of the image"""
        return np.median(self.get(), *args, **kwargs)

    def std(self, *args, **kwargs):
        """Return the standard deviation value of the image"""
        return np.std(self.get(), *args, **kwargs)

    def grayscale(self) -> "PIVImage":
        """make image grayscale if ndim==3"""
        if self.ndim == 2:
            return copy.deepcopy(self)
        new_piv_image = PIVImage.from_array(self._img.copy(), is_first_image=self._is_a, pco=self.pco)
        new_piv_image._img = cv2.cvtColor(self.get(), cv2.COLOR_BGR2GRAY)
        return new_piv_image

    def smooth(self, kernel_size) -> "PIVImage":
        """smooth the image"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
        smoothed_image = cv2.filter2D(self.get(), -1, kernel)
        return PIVImage.from_array(smoothed_image, meta=self.meta.copy() if keep_meta else {})

    def normalize(self, inplace: bool = False) -> "PIVImage":
        """normalize the image"""
        _img = self.get()
        _min = np.nanmin(_img)
        _max = np.nanmax(_img)
        norm_img = (_img - _min) / (_max - _min)
        if inplace:
            self._img = norm_img
            return self
        return self.from_array(norm_img, meta=self.meta.copy() if keep_meta else {})

    def rot90(self, inplace: bool = False) -> "PIVImage":
        """rotate image by 90 degrees"""
        _img = self.get()
        if inplace:
            self._img = np.rot90(_img)
            return self
        return self.from_array(np.rot90(_img), meta=self.meta.copy() if keep_meta else {})

    def fliplr(self, inplace: bool = False) -> "PIVImage":
        """flip image left-right"""
        _img = self.get()
        if inplace:
            self._img = np.fliplr(_img)
            return self
        return self.from_array(np.fliplr(_img), meta=self.meta.copy() if keep_meta else {})

    def flipud(self, inplace: bool = False) -> "PIVImage":
        """flip image up-down"""
        _img = self.get()
        if inplace:
            self._img = np.flipud(_img)
            return self
        return self.from_array(np.flipud(_img), meta=self.meta.copy() if keep_meta else {})

    def rot180(self, inplace: bool = False) -> "PIVImage":
        """rotate image by 180 degrees"""
        _img = self.get()
        if inplace:
            self._img = np.rot90(_img, k=2)
            return self
        return self.from_array(np.rot90(_img, k=2), meta=self.meta.copy() if keep_meta else {})

    def apply_mask(self, mask_array: np.ndarray, fill_value: int, inplace: bool = False) -> "PIVImage":
        """Apply a mask to the image. Will set the masked values to `fill_value`."""
        masked_img = self.get().copy()
        masked_img[mask_array] = fill_value
        if inplace:
            self._img = masked_img
            return self
        return self.from_array(masked_img, meta=self.meta.copy() if keep_meta else {})

    def plot(self,
             figure_height: float = 3.,
             spacing: float = 0.0,
             ax_hist_ratio: float = 0.05,
             vmin: float = None,
             vmax: float = None,
             bins: int = 101,
             density: bool = False,
             axs: Tuple[plt.Axes, None] = None):
        """plot the image. If no `ax` is provided a histogram is automatically plotted below the image."""
        figure_height = figure_height
        spacing = spacing

        _shape = self.get().shape
        h, w = _shape[0], _shape[1]
        hist_height = ax_hist_ratio * figure_height

        left, width = 0.1, w / w
        bottom, height = 0.1, h / h

        hist_ax_pos = [left, bottom, width, hist_height]
        img_ax_pos = [left, hist_height + bottom + spacing, width, height]

        if axs is not None:
            if len(axs) == 1:
                return self._plot(ax=axs[0], cmap='gray', vmin=vmin, vmax=vmax, hide_colorbar=True)
            ax_img = axs[0]
            ax_hist = axs[1]
        else:
            # start with a square Figure
            fig = plt.figure(figsize=(w / h * figure_height, figure_height))

            ax_img = fig.add_axes(img_ax_pos)
            ax_img.axis('off')

            # if self._is_a is not None:
            #     if self._is_a:
            #         ax_img.set_title('A', size=12)
            #     else:
            #         ax_img.set_title('B', size=12)

            ax_hist = fig.add_axes(hist_ax_pos)

        self._plot(ax=ax_img, cmap='gray', vmin=vmin, vmax=vmax, hide_colorbar=True)
        self.hist(ax=ax_hist, bins=bins, color='k', density=density)

        if vmax:
            ax_hist.vlines(vmax, 0, ax_hist.get_ylim()[1], linestyles='--', color='gray')
        if vmin:
            ax_hist.vlines(vmin, 0, ax_hist.get_ylim()[1], linestyles='--', color='gray')

        return ax_img, ax_hist

    def _plot(self, ax=None, autoscale: bool = False, **kwargs):
        if ax is None:
            ax = plt.gca()
        _img = self.get()
        if autoscale:
            _vmin = kwargs.get('vmin', np.nanmin(_img))
            kwargs['vmin'] = _vmin
            _vmax = kwargs.get('vmax', np.nanmin(_img))
            kwargs['vmax'] = _vmax

        cmap = kwargs.pop('cmap', 'gray')
        hide_colorbar = kwargs.pop('hide_colorbar', False)
        im = ax.imshow(_img, cmap=cmap, **kwargs)
        if not hide_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        return ax

    def hist(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        color = kwargs.pop('color', 'k')
        ax.hist(self.get().ravel(), color=color, **kwargs)
        return ax

    def to_tiff(self, filename, write_meta: bool = False):
        filename = save_image(filename, self[:])
        if write_meta:
            yam_filename = filename.parent / f'{filename.stem}_meta.yaml'
            try:
                import yaml
            except ImportError:
                raise ImportError('Please install pyyaml to write the meta to a yaml file.')
            with open(yam_filename, 'w') as f:
                yaml.safe_dump(self.meta, f)
        return filename


class PIVBackgroundImage(_PIVImage):
    """PIV background image helper class"""

    @classmethod
    def from_array(cls, img: np.ndarray, meta: Union[Dict, Metadata] = None):
        """Create a PIVImage object from a numpy array"""
        obj = cls(filename=None, meta=meta)
        obj._img = img
        return obj


class PIVImage(_PIVImage):
    """PIV image helper class"""

    def __init__(self,
                 filename: Union[pathlib.Path, None],
                 is_first_image: bool = None,
                 pco: bool = False,
                 meta: Union[Dict, Metadata] = None,
                 **kwargs):
        super().__init__(filename=filename, meta=meta, **kwargs)

        self.pco = pco
        self._is_a = is_first_image
        if self._is_a is None and pco:
            raise ValueError('If pco is set, ou must specify whether this is the first or second image via '
                             'parameter "is_first_image".')
        self._img = None

    def __sub__(self, other):
        """Subtract two PIV images. If the result is negative, set it to zero."""
        if not isinstance(other, (PIVImage, np.ndarray)):
            raise TypeError(f'Cannot subtract {type(other)} from {type(self)}')
        if isinstance(other, PIVImage):
            diffimg = self.get() - other.get()
        else:
            diffimg = self.get() - other
        diffimg[diffimg < 0] = 0
        return self.__class__(filename=None, is_first_image=None).from_array(diffimg)

    def get(self) -> np.ndarray:
        """Return the image. If not yet loaded (self._img is None), load it."""
        if self._img is None:
            if self._filename is None:
                raise ValueError('No filename set!')
            img = loadimg(self._filename)
            if not self.pco:
                self._img = img
            else:
                ny, nx = img.shape
                if self._is_a:
                    print('return second part of image')
                    self._img = img[:ny // 2, ...]
                else:
                    print('return first part of image')
                    self._img = img[ny // 2:, ...]
        return self._img

    @classmethod
    def from_array(cls,
                   arr,
                   is_first_image: bool = None,
                   pco: bool = False,
                   meta: Union[Dict, Metadata] = None,
                   **kwargs) -> "PIVImage":
        """Init the class from a numpy array."""
        pivimg = cls(filename=None, is_first_image=is_first_image, pco=pco, meta=meta, **kwargs)
        ny, nx = arr.shape[0], arr.shape[1]
        if pco and is_first_image:
            pivimg._img = arr[:ny // 2, ...]
        elif pco and not is_first_image:
            pivimg._img = arr[ny // 2:, ...]
        else:
            pivimg._img = arr
        return pivimg


class PIVImages:
    """Collection of PIV images. The images are loaded on demand. The images have no
    relation to each other. They are just a collection of images. If you intend to work
    with image pairs, use the PIVImagePair class instead."""

    def __init__(self, filenames: List):
        self.filenames = list(filenames)
        self._images = {}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item) -> PIVImage:
        """Return image array. Loaded on demand, which means, that if the image was
        loaded before, it will be returned from memory, otherwise it will be loaded from
        disk."""
        if isinstance(item, slice):
            return PIVImages.from_array_list([self[i] for i in range(*item.indices(len(self)))])
        if item not in self._images:
            self._images[item] = PIVImage(self.filenames[item])
        return self._images[item]

    def compute_background(self, func) -> PIVImage:
        """Compute the background image using the given function. The function must
        accept a numpy array and axis=0 as input and return a numpy array as output.
        E.g.: np.min, np.mean, np.median, etc.

        Parameters
        ----------
        func : Callable
            Function to compute the background image. Must accept a numpy array and
            axis=0 as input and return a numpy array as output.

        Returns
        -------
        PIVImage
            Background image.
        """
        return PIVBackgroundImage.from_array(func(np.stack([self[i][:] for i in range(len(self))]), axis=0))

    @classmethod
    def from_array_list(cls, list_of_arrays: List[np.ndarray]):
        """Create a PIVImages object from a list of numpy arrays."""
        obj = cls([None for _ in range(len(list_of_arrays))])
        obj._images = {i: PIVImage(list_of_arrays[i]) for i in range(len(list_of_arrays))}
        return obj


class PIVImagePair:
    """Helper class to work with a pair of PIV images

    If a PIV recording is only stored in one file, than pass None for filename_B.
    """

    def __init__(self,
                 filename_A: Union[str, pathlib.Path, PIVImage,],
                 filename_B: Union[str, pathlib.Path, PIVImage, None]):
        if filename_A is None:
            raise ValueError('filename_A cannot be None!')

        if isinstance(filename_A, PIVImage):
            filename_A = filename_A.filename
        else:
            filename_A = Path(filename_A)

        if filename_B is None:
            # both images are assumed to be stored in the first image:
            self._A = PIVImage(filename_A, is_first_image=True, pco=True)
            self._B = PIVImage(filename_A, is_first_image=False, pco=True)
        else:
            if isinstance(filename_B, PIVImage):
                filename_B = filename_B.filename
            else:
                filename_B = Path(filename_B)

            if not filename_B.exists():
                raise FileNotFoundError(filename_A)
            if not filename_B.exists():
                raise FileNotFoundError(filename_B)
            self._A = PIVImage(filename_A, is_first_image=True)
            self._B = PIVImage(filename_B, is_first_image=False)

    def plot(self,
             figure_height: float = 3.,
             spacing: float = 0.0,
             ax_hist_ratio: float = 0.05,
             vmin: float = None,
             vmax: float = None,
             bins: int = 101,
             density: bool = False):
        """Plot both images next to each other"""
        figure_height = figure_height
        spacing = spacing

        _shape = self.A.get().shape
        h, w = _shape[0], _shape[1]
        hist_height = ax_hist_ratio * figure_height

        left, width = 0.1, w / w
        bottom, height = 0.1, h / h

        # start with a square Figure
        fig = plt.figure(figsize=(w / h * figure_height, figure_height))

        hist_ax_pos_A = [left, bottom, width, hist_height]
        img_ax_pos_A = [left, hist_height + bottom + spacing, width, height]
        ax_imgA = fig.add_axes(img_ax_pos_A)
        ax_imgA.axis('off')
        ax_histA = fig.add_axes(hist_ax_pos_A)

        hist_ax_pos_B = [left + hist_ax_pos_A[2], bottom, width, hist_height]
        img_ax_pos_B = [left + img_ax_pos_A[2], hist_height + bottom + spacing, width, height]
        ax_imgB = fig.add_axes(img_ax_pos_B)
        ax_imgB.axis('off')
        ax_histB = fig.add_axes(hist_ax_pos_B)

        self.A._plot(ax=ax_imgA, cmap='gray', vmin=vmin, vmax=vmax, hide_colorbar=True)
        self.A.hist(ax=ax_histA, bins=bins, color='k', density=density)

        self.B._plot(ax=ax_imgB, cmap='gray', vmin=vmin, vmax=vmax, hide_colorbar=True)
        self.B.hist(ax=ax_histB, bins=bins, color='k', density=density)

        if vmax:
            ax_histA.vlines(vmax, 0, ax_histA.get_ylim()[1], linestyles='--', color='gray')
        if vmin:
            ax_histA.vlines(vmin, 0, ax_histA.get_ylim()[1], linestyles='--', color='gray')

        if vmax:
            ax_histB.vlines(vmax, 0, ax_histB.get_ylim()[1], linestyles='--', color='gray')
        if vmin:
            ax_histB.vlines(vmin, 0, ax_histB.get_ylim()[1], linestyles='--', color='gray')

        ax_hist_A_ymax = ax_histA.get_ylim()[1]
        ax_hist_B_ymax = ax_histB.get_ylim()[1]
        common_hist_ymax = max(ax_hist_A_ymax, ax_hist_B_ymax)
        ax_histA.set_ylim([0, common_hist_ymax])
        ax_histB.set_ylim([0, common_hist_ymax])

        # disable ax_histB yticks
        ax_histB.set_yticks([])

        return ax_imgA, ax_histA, ax_imgB, ax_histB

    def plot_overlay(self, channel_A=0, channel_B=2, **kwargs):
        """plot both images in different channels. A is plotted in red, B in blue"""
        if channel_A not in [0, 1, 2]:
            raise ValueError('channel_A must be 0, 1 or 2')
        if channel_B not in [0, 1, 2]:
            raise ValueError('channel_B must be 0, 1 or 2')
        if channel_A == channel_B:
            raise ValueError('channel_A and channel_B must be different')
        ax = kwargs.pop('ax', plt.gca())
        arrA = self.A.get()
        arrRGB = np.zeros((*arrA.shape, 3))
        arrRGB[:, :, channel_A] = arrA
        arrRGB[:, :, channel_B] = self.B.get()
        ax.imshow(arrRGB)
        return ax

    def subtract_background(self, background_filename: Union[Union[str, pathlib.Path],
                                                             List[Union[str, pathlib.Path]]]) -> "PIVImagePair":
        """subtract the background from both images. A new PIVImagePair is returned"""
        if not isinstance(background_filename, (tuple, list)):
            background_filename = [background_filename, ]
        if len(background_filename) == 0:
            raise ValueError('No background file provided!')
        elif len(background_filename) > 2:
            raise ValueError('Too many background files provided!')
        bga = loadimg(background_filename[0])
        if len(background_filename) == 2:
            bgb = loadimg(background_filename[1])
        else:
            bgb = bga
        _piv_img_pari = copy.deepcopy(self)
        _piv_img_pari._A._img = _piv_img_pari._A._img - bga
        _piv_img_pari._B._img = _piv_img_pari._B._img - bgb
        return _piv_img_pari

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B


class PIVImagePairs:
    """Collection of PIV images paris"""

    def __init__(self, filenames_A: list, filenames_B: list):
        self.filenames_A = filenames_A
        self.filenames_B = filenames_B
        self._images_A = {}
        self._images_B = {}

    def __getitem__(self, item):
        """Return image array"""
        if item not in self._images_A:
            self._images_A[item] = PIVImage(self.filenames_A[item])
        if item not in self._images_B:
            self._images_B[item] = PIVImage(self.filenames_B[item])
        return PIVImagePair(self._images_A[item], self._images_B[item])


def load_piv_image(filename: Path, is_first: bool = None):
    """Initialize a PIVImage from a filename"""
    return PIVImage(filename, is_first)


def loadimg(img_filepath: Path):
    """
    loads b16 or other file format
    """
    img_filepath = Path(img_filepath)
    if img_filepath.suffix in ('b16', '.b16'):
        im_ = pco_reader.load(str(img_filepath))
    else:
        im_ = cv2_imread(str(img_filepath), -1)
    return im_


def save_image(filename, arr, img_type='tiff'):
    """write data to a tiff file"""
    if img_type in ('tiff', '.tiff'):
        cv2.imwrite(str(filename), arr)
        return pathlib.Path(filename).absolute()
    raise NotImplementedError('Image types other than tiff are not implemented yet.')
