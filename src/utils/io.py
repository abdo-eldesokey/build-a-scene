from pathlib import Path
from typing import Union, Optional, Any


def create_dir(path: Union[str, Path], logger: Optional[Any] = None) -> None:
    """Check if a directory exists, if not, it creates it

    Args:
        path (Union[str, Path]): Path to directory
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        if logger is not None:
            logger(f"Directory {path} was created successfully!")


def create_parent(file_path: Union[str, Path]):
    path = Path(file_path)
    if path.is_file:
        create_dir(path.parents[0])
