from api.config import PACKAGE_ROOT

with open(PACKAGE_ROOT / 'VERSION') as _version:
    __version__ = _version.read().strip()