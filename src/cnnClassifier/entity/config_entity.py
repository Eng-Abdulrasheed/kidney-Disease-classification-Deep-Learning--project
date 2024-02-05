from dataclasses import dataclass
from pathlib import Path

# using @dataclass so the class will be access as variable from other file
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path