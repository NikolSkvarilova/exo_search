import datetime
from filelock import SoftFileLock
import sys
from pathlib import Path
from enum import Enum


class Category(Enum):
    """Log message categories."""

    INFO = "INFO"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    START = "START"
    END = "END"


class Logger:
    """Class for logging messages into a file."""

    def __init__(self, script_name: str, path: Path = None):
        """Constructor.

        Args:
            script_name (str): name of the script initializing the Logger object.
            path (Path, optional): directory for the logs. Defaults to logs/.
        """

        self.script_name = script_name
        self.filename = Path(f"logs/{get_timestamp()}.log")
        if path is not None:
            self.filename = path / f"{get_timestamp()}.log"
        self.lock_path = f"{self.filename}.lock"
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def log(self, category: Category, context: str, position: str, msg: str) -> None:
        """Log a mesage.

        Args:
            category (Category): type of message.
            context (str): identificator for the data that caused the log.
            position (str): place in the code that caused the log.
            msg (str): log message.
        """
        lock = SoftFileLock(self.lock_path, thread_local=False)
        with lock:
            try:
                with open(self.filename, "a") as f:
                    f.write(
                        f"{get_timestamp()} -- {self.script_name} -- {context} -- {position} -- {category.value}: {msg}\n"
                    )
            except OSError as e:
                print(f"Logging failed: {e}", file=sys.stderr)


def get_timestamp() -> str:
    """Return the current timestamp.

    Returns:
        str: timestamp.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-1]
