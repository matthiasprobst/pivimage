import copy
import logging
import pathlib
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread as cv2_imread
from pco_tools import pco_reader as pco

logger = logging.getLogger(__package__)


class PIVImage:
    """PIV image helper class"""

    def __init__(self, filename: Union[pathlib.Path, None],
                 is_first_image: bool = None):
        if filename is not None:
            self._filename = pathlib.Path(filename)
        else:
            self._filename = None
        self._is_a = is_first_image
        self._img = None

    def __sub__(self, other):
        if not isinstance(other, PIVImage):
            raise TypeError(f'Cannot subtract {type(other)} from {type(self)}')
        diffimg = self.get() - other.get()
        return self.__class__(filename=None, is_first_image=None).from_array(diffimg)

    @property
    def ndim(self):
        """Return filename"""
        return self.get().ndim

    @property
    def filename(self):
        """Return filename"""
        return self._filename

    def __getitem__(self, item) -> "PIVImage":
        return self.get().__getitem__(item)

    def __array__(self):
        """returns numpy array if np.asarray() or e.g. np.roll() is called on this object"""
        return self._img

    @staticmethod
    def from_array(arr) -> "PIVImage":
        pivimg = PIVImage(filename=None, is_first_image=None)
        pivimg._img = arr
        return pivimg

    def grayscale(self) -> "PIVImage":
        """make image grayscale if ndim==3"""
        if self.ndim == 2:
            return self
        new_piv_image = copy.deepcopy(self)
        new_piv_image._img = cv2.cvtColor(self.get(), cv2.COLOR_BGR2GRAY)
        return new_piv_image

    def smooth(self, kernel_size) -> "PIVImage":
        """smooth the image"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
        smoothed_image = cv2.filter2D(self.get(), -1, kernel)
        return PIVImage.from_array(smoothed_image)

    def clear(self):
        """free the (RAM), aka unset _img"""
        self._img = None

    def get(self):
        if self._img is None:
            return loadimg(self._filename)
        return self._img

    def normalize(self) -> "PIVImage":
        _img = self.get()
        _min = np.nanmin(_img)
        _max = np.nanmax(_img)
        _img = (_img - _min) / (_max - _min)
        pivimg = PIVImage(self._filename, self._is_a)
        pivimg._img = _img
        return pivimg

    def rot90(self) -> "PIVImage":
        _img = self.get()
        pivimg = PIVImage(self._filename, self._is_a)
        pivimg._img = np.rot90(_img, k=1)
        return pivimg

    def rot180(self) -> "PIVImage":
        _img = self.get()
        pivimg = PIVImage(self._filename, self._is_a)
        pivimg._img = np.rot90(_img, k=2)
        return pivimg

    def plot(self,
             figure_height: float = 3.,
             spacing: float = 0.0,
             ax_hist_ratio: float = 0.05,
             vmin: float = None,
             vmax: float = None,
             bins: int = 101,
             density: bool = False,
             ax=None):
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

        if ax is not None:
            # dont plot histogram!
            return self._plot(ax=ax, cmap='gray', vmin=vmin, vmax=vmax, hide_colorbar=True)

        # start with a square Figure
        fig = plt.figure(figsize=(w / h * figure_height, figure_height))

        ax_img = fig.add_axes(img_ax_pos)
        ax_img.axis('off')
        if self._is_a is not None:
            if self._is_a:
                ax_img.set_title('A', size=12)
            else:
                ax_img.set_title('B', size=12)

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

    def to_tiff(self, filename):
        cv2.imwrite(filename, self[:])


class PIVImages:
    """Collection of PIV images"""

    def __init__(self, filenames: list):
        self.filenames = filenames
        self._images = {}

    def __getitem__(self, item):
        """Return image array"""
        if item not in self._images:
            self._images[item] = PIVImage(self.filenames[item])
        return self._images[item]


class PIVImagePair:
    """Helper class to work with a pair of PIV images"""

    def __init__(self, filename_A, filename_B):
        if isinstance(filename_A, PIVImage):
            filename_A = filename_A.filename
        else:
            filename_A = Path(filename_A)
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
        im_ = pco.load(str(img_filepath))
    else:
        im_ = cv2_imread(str(img_filepath), -1)
    return im_
