"""ParkCast Vision — parking lot occupancy detection from CCTV images."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("parkcast")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["data", "eda", "train", "evaluate", "inference", "visualize", "utils"]
