import abc
import pathlib
from typing import Dict, Union

from .file_wrapper import FileWrapper


def _readonly(self, *args, **kwargs):
    raise RuntimeError("Cannot modify ReadOnlyDict")


class ReadOnlyDict(dict):
    """from https://stackoverflow.com/questions/19022868/how-to-make-dictionary-read-only"""
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly


class MetaFileInterface(abc.ABC):
    """Abstract base class for metadata interfacing"""

    def __init__(self, hidden: bool = False):
        self.hidden = hidden

    @classmethod
    @abc.abstractmethod
    def load(cls, filename: Union[str, pathlib.Path]) -> Dict:
        """load a metadata file"""

    @classmethod
    @abc.abstractmethod
    def save(cls, data, filename: Union[str, pathlib.Path]) -> pathlib.Path:
        """save a metadata file"""


class Metadata:

    def __init__(self,
                 parent: FileWrapper):
        self._parent_filename = parent.filename
        self._data = {}
        self._filename = None

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.filename}")'

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._data[k] = v

    def add(self, key, value):
        self.update({key: value})

    def load(self,
             interface: MetaFileInterface = None,
             readonly=True) -> ReadOnlyDict:
        """Load metadata from file even if already loaded."""
        if interface is None:
            interface = metafile_interfaces[self.filename.suffix[1:]]
        data = interface.load(self.filename)
        self._data = ReadOnlyDict(data) if readonly else data

        return self._data

    def save(self,
             filename: Union[str, pathlib.Path] = None,
             interface: MetaFileInterface = None):
        """Write metadata to file. The metadata interface class is determined
        based on the extension, e.g. '.json' will use `JSONMeta`"""
        if filename is None:
            filename = self.filename
        filename = pathlib.Path(filename)

        if interface is None:
            try:
                interface = metafile_interfaces[filename.suffix[1:]]
            except KeyError as e:
                av_interfaces = metafile_interfaces
                av_interfaces.pop('dict')
                raise KeyError(f'A Metadata interface class for {filename.suffix} does not exist. '
                               f'Please choose one of {", ".join(av_interfaces.keys())}')
        return interface.save(self._data, filename)

    def unlink(self, missing_ok: bool = True) -> None:
        """Deletes the meta file. cannot be recovered."""
        if self.filename is not None:
            self.filename.unlink(missing_ok=missing_ok)

    @property
    def filename(self) -> str:
        """build filename for metadata file"""
        if self._filename is None:
            # the default meta file ext is json
            if self._parent_filename is None:
                raise ValueError('The parent class is not associated to a file!')
            self._filename = self._parent_filename.with_suffix('.json')
        return self._filename

    @property
    def data(self) -> ReadOnlyDict:
        """Return metadata dict. loads if not already loaded."""
        if self._data is None:
            return self.load()
        return ReadOnlyDict(self._data)


class DictMeta(MetaFileInterface):
    """Meta data manager for data stored in python dictionary"""

    SUFFIX = None

    def load(self, filename):
        raise NotImplementedError(f'{self.__class__.__name__} has no load method implemented')

    def save(self, data=None, filename=None):
        raise NotImplementedError(f'{self.__class__.__name__} has no generic save method.')

    @classmethod
    def json_save(cls, data, filename):
        """save as json"""
        import json
        with open(filename) as f:
            return json.dump(data, f)

    @classmethod
    def json_yaml(cls, data, filename):
        """save as yaml"""
        import yaml
        with open(filename) as f:
            return yaml.safe_dump(data, f)


class JSONMeta(MetaFileInterface):
    """Meta data manager for data stored in JSON files"""

    SUFFIX = '.json'

    @classmethod
    def load(cls, filename):
        """load metadata"""
        import json
        with open(filename) as f:
            return json.load(f)

    @classmethod
    def save(cls, data, filename) -> pathlib.Path:
        """save metadata"""
        import json
        with open(filename, 'w') as f:
            json.dump(data, f)
        return pathlib.Path(filename)


class YAMLMeta(MetaFileInterface):
    """Meta data manager for data stored in YAML files"""

    SUFFIX = '.yaml'

    @classmethod
    def load(self, filename):
        """load metadata"""
        import yaml
        with open(filename) as f:
            return yaml.safe_load(f)

    @classmethod
    def save(self, data, filename):
        """save metadata"""
        import yaml
        with open(filename) as f:
            return yaml.safe_dump(data, f)


metafile_interfaces = {
    'json': JSONMeta,
    'yaml': YAMLMeta,
    'dict': DictMeta}


def metafile(reader: str, hidden: bool = True) -> MetaFileInterface:
    """automatically selects meta file reader"""
    return metafile_interfaces[reader](hidden=hidden)

# class Metadata:
#     def __init__(self, hidden: bool, reader: Union[str, MetaFileManager]):
#         self.hidden = hidden  # whether filenames are hidden, thus starting with a "."
#         self.reader = reader if isinstance(reader) else metadata_managers.get(reader, None)
#         if self.reader is None:
#             raise KeyError(f'Invalid reader: {reader}')
#         self._parent = None
#         self.filename = None
#
#     def assign(self, parent: FileWrapper):
#         pf = parent.filename
#         if self.hidden:
#             self.filename = pf.with_name(f'.{pf.stem}')
#         self.filename = pf.with_suffix(self.reader.SUFFIX)
#
#     def load(self):
#         if self.filename is None:
#             raise RuntimeError('Please first assign the Metadata object to a PIVImage using ".assign(<img>)"')
#         self._data = self.reader.load(self.filename)
