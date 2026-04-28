"""
Metadata engines for reading and writing EXIF/IPTC/XMP data
"""

from .metadata_reader import MetadataReader
from .metadata_writer import MetadataWriter
from .iptc_engine import IPTCEngine, IPTCTemplate

__all__ = ["MetadataReader", "MetadataWriter", "IPTCEngine", "IPTCTemplate"]
