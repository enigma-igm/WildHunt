import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime as dt
from functools import lru_cache
from pathlib import Path

# =========================================================================== #
# ================================ Log config =============================== #
# =========================================================================== #

MIN_LEVEL = logging.DEBUG  # min level of warning to be printed
FLT_LEVEL = logging.WARNING  # min level that goes to stderr
CUR_LEVEL = logging.DEBUG  # level of the first child logger

# =========================================================================== #


class LogFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


# =========================================================================== #


# add this for year as well: `%Y-%m-%d - `
LOGGING_FORMATTER_STR = "%(name)s - %(levelname)s: %(message)s"

# =========================================================================== #
# ====================== Call this in scripts if needed ===================== #
# =========================================================================== #


def configure_logging(
    *,
    min_level: int = MIN_LEVEL,
    flt_level: int = FLT_LEVEL,
    logger_level: int = CUR_LEVEL,
    log_to_file: bool | None = None,
    local_path: Path | None = None,
    capture_warnings: bool = False,
    silence: list[str] | None = None,
    silence_all_external: bool = True,
    silcence_whitelist: list[str] = [],
) -> logging.Logger:
    """
    Configure root logging for the package and return the main package logger.

    This function sets up logging **exactly once** and is safe to call multiple times.
    It attaches handlers to the root logger, formats log messages, optionally logs
    to a file, redirects Python warnings, and can silence noisy external libraries.

    Parameters
    ----------
    min_level : int, optional
        Minimum logging level for the root logger and all handlers (default: DEBUG).
    flt_level : int, optional
        Threshold below which messages go to stdout. Messages at or above this
        level go to stderr (default: WARNING).
    logger_level : int, optional
        Logging level for the returned package logger (default: DEBUG).
    log_to_file : bool | None, optional
        Whether to write logs to a file. If None, the environment variable
        `EQS_LOG_TO_FILE` is checked (default: None).
    local_path : Path | None, optional
        Base path for log files if logging to a file. Defaults to `~/.EQS_LOCALPATH`.
    capture_warnings : bool, optional
        Whether to redirect Python warnings to the logging system (default: True).
    silence : list[str] | None, optional
        List of external library logger names to silence (set to WARNING level)
        to reduce verbose output (default: None).

    Returns
    -------
    logging.Logger
        A named logger for the package (`"euclid_qso_selection"`), configured
        with the requested level and propagating messages to the root logger.

    Notes
    -----
    - Messages below `flt_level` are printed to stdout, messages at or above
      `flt_level` are printed to stderr.
    - If `log_to_file` is True, a log file `euclid_qso_selection.log` will be
      created in `local_path/logs`, with old logs backed up automatically.
    - This function is idempotent: repeated calls do not add duplicate handlers.
    - Named loggers (e.g., `logging.getLogger("euclid_qso_selection.module")`)
      will automatically propagate messages to the root and use this formatting.
    - Use the `silence` parameter to suppress verbose logs from third-party libraries.
    """

    root = logging.getLogger()

    if getattr(root, "_wh_configured", False):
        return logging.getLogger("wildhunt")

    root.setLevel(min_level)

    formatter = logging.Formatter(fmt=LOGGING_FORMATTER_STR)

    # messages lower than WARNING go to stdout
    # messages >= WARNING (and >= STDOUT_LOG_LEVEL) go to stderr
    # stdout handler (< WARNING)
    i_handler = logging.StreamHandler(sys.stdout)
    i_handler.addFilter(LogFilter(flt_level))
    i_handler.setFormatter(formatter)

    # stderr handler (>= WARNING)
    e_handler = logging.StreamHandler(sys.stderr)
    e_handler.setLevel(max(min_level, flt_level))
    e_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(i_handler)
    root.addHandler(e_handler)

    # Logger to redirect warning to log, as I am essentially treating them as Info
    #  at this point
    if capture_warnings:
        # this logs to py.warnings, which is created ad hoc
        # act on this logger if you need to anything on this side
        logging.captureWarnings(True)

    # ======================================================================= #
    # =============== Add log to file if the user requests it =============== #
    # ======================================================================= #

    # check the env variable if not specified
    if log_to_file is None:
        log_to_file = os.getenv("WILDHUNT_LOG_TO_FILE", False) in {"1", "true", "True"}

    if log_to_file:
        if local_path is None:
            local_path = Path.home() / ".WILDHUNT_LOCALPATH"

        log_dir = local_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "wildhunt.log"
        if log_file.exists():
            ts = dt.fromtimestamp(log_file.stat().st_mtime).strftime(
                "%Y-%m-%dT%H-%M-%S"
            )
            log_file.rename(log_dir / f"wildhunt_{ts}.log.bak")

        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(min_level)
        f_handler.setFormatter(formatter)
        root.addHandler(f_handler)

    pkg_logger = logging.getLogger("wildhunt")
    pkg_logger.setLevel(logger_level)

    root._wh_configured = True

    # only silence packages if requested
    if silence:
        for mod in silence:
            logging.getLogger(mod).setLevel(logging.WARNING)

    # silences everything but wildhunt (defaults to True for everything below WARNING)
    if silence_all_external:
        for name, logger_obj in logging.root.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger) and name.split(".")[0] not in (
                silcence_whitelist
            ):
                if not name.startswith("wildhunt"):
                    logger_obj.setLevel(logging.WARNING)

    return pkg_logger


# =========================================================================== #


# In case I need to suppress things temporarily
# thanks to: https://gist.github.com/simon-weber/7853144
# If no logging this is the first offender to have a look at!
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


# =========================================================================== #


# Keep track of 10 different messages and then warn again
# 10 became None to only warn once
@lru_cache(None)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #
