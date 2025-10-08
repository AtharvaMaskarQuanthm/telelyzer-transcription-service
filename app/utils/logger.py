# import os
# import logging
# from datetime import datetime

# def get_logger(name: str = "telelyzer-transcription-service-logger") -> logging.Logger:
#     # Create logs directory if it doesn't exist
#     logs_dir = "/logs"
#     os.makedirs(logs_dir, exist_ok=True)

#     # Get today's date for filename
#     date_str = datetime.now().strftime("%Y-%m-%d")
#     log_filename = os.path.join(logs_dir, f"{date_str}.log")

#     # Set up logger
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)

#     # Prevent duplicate handlers if already added
#     if not logger.handlers:
#         # File handler
#         file_handler = logging.FileHandler(log_filename)
#         file_handler.setLevel(logging.DEBUG)

#         # Console handler (optional)
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)

#         # Formatter
#         formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#         )
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)

#         # Add handlers
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)

#     return logger


import logging

def get_logger(name: str = "telelyzer-transcription-service-logger") -> logging.Logger:
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.handlers:
        # Console handler only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler
        logger.addHandler(console_handler)

    return logger
