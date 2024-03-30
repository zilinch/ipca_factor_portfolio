import logging
import os

class ErrorLogger:
    def __init__(self, log_filename='error_log.txt'):
        self.log_filename = log_filename
        self.log_format = '[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.setup_logger()

    def setup_logger(self):
        """Set up the logger with basic configuration."""
        log_directory = os.path.dirname(self.log_filename)
        # Only attempt to create the directory if it is not empty
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)
            
        logging.basicConfig(filename=self.log_filename,
                            filemode='a',  # Append mode
                            format=self.log_format,
                            level=logging.ERROR,  # Capture only ERROR and CRITICAL by default
                            datefmt=self.date_format)

    def log_error(self, module_name, error_message, detailed_error=None, action=None):
        """
        Log an error message with optional details and action.

        Args:
            module_name (str): The name of the module or source of the error.
            error_message (str): A concise description of the error.
            detailed_error (str, optional): Additional details about the error.
            action (str, optional): The action taken or recommended.
        """
        logger = logging.getLogger(module_name)
        logger.error(f'Message: {error_message}')  # Log the basic error message
        if detailed_error:
            logger.error(f'Details: {detailed_error}')  # Optionally log detailed error
        if action:
            logger.error(f'Action: {action}')  # Optionally log the action taken or recommended



# if __name__ == "__main__":
#     logger = ErrorLogger()  # Instantiates the ErrorLogger

#     # Log another error with only the message
#     logger.log_error('user_authentication', 'User login failed due to invalid credentials.')