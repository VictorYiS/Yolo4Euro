import logging
import logging.handlers
import os

# Ensure the 'logs' directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "running.log")

# Configure logging
logger = logging.getLogger("optimized_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # key to prevent duplicate logs

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  
console_formatter = logging.Formatter("%(message)s")  # only message in console
console_handler.setFormatter(console_formatter)

file_handler = logging.handlers.RotatingFileHandler(
    log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)  # debug level for file logging
file_formatter = logging.Formatter(
    "[%(asctime)s.%(msecs)03d][%(levelname).1s] %(message)s",
    datefmt="%H:%M:%S",
)
file_handler.setFormatter(file_formatter)

# make sure handlers are not added multiple times
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ANSI color codes
class LogColors:
    RESET = "\033[0m"
    GREEN = "\033[92m"  # Green
    RED = "\033[91m"  # Red


class CustomLogger:
    def __init__(self, logger):
        self.logger = logger

    def debug(self, message, *args):
        formatted_message = self._format_message(message, *args)
        self.logger.debug(formatted_message)

    def info(self, message, *args):
        formatted_message = self._format_message(message, *args)
        console_message = f"{LogColors.GREEN}{formatted_message}{LogColors.RESET}"
        self.logger.info(console_message)

    def error(self, message, *args):
        formatted_message = self._format_message(message, *args)
        console_message = f"{LogColors.RED}{formatted_message}{LogColors.RESET}"
        self.logger.error(console_message)

    def _format_message(self, message, *args):
        if args:
            try:
                return message % args if "%" in message else message.format(*args)
            except Exception:
                return message
        return message


# Instantiate the custom logger
log = CustomLogger(logger)
