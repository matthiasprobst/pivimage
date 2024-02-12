import pathlib


class FileWrapper:
    """A (very basic) class associated with a file"""

    def __init__(self, filename, missing_ok=True):
        if filename is None:
            if not missing_ok:
                raise ValueError('filename cannot be None')
            self._filename = None
        else:
            self._filename = pathlib.Path(filename)
            if not missing_ok and not self._filename.exists():
                raise FileNotFoundError(f'File "{self._filename}" does not exist.')

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._filename}")'

    @property
    def filename(self):
        """Return filename"""
        return self._filename
