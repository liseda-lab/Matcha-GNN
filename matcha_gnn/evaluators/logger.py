import os

class Logger:
    """
    Logger class for logging training results.

    Parameters:
    metrics: list of metrics to log
    log_dir: directory to save log file
    log_file_name: name of log file
    log_file_extension: extension of log file
    """
    def __init__(self, metrics, log_dir = "", log_file_name = "results", log_file_extension = ".txt"):
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.log_file_extension = log_file_extension
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name + self.log_file_extension)
        self.metrics = metrics
        self.log_file = open(self.log_file_path, 'w')

    def start_log(self):
        """
        Start logging by writing the header of the log file. Only call this function once.
        """
        self.log_file.write('run\tepoch\t')
        for metric in self.metrics: self.log_file.write(metric + '\t')
        self.log_file.write('\n')

    def log(self, msg):
        """
        Log the message to the log file.
        Message should be in the format of 'run\tepoch\tmetric1\tmetric2\t...'
        """
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def close(self):
        """
        Call to close file when done logging for the current run.
        """
        self.log_file.close()

    def __del__(self):
        self.close()