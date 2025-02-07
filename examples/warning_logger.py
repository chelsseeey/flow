import json
import sys
from datetime import datetime

class WarningLogger:
    def __init__(self):
        self.logs = {}
        self.current_iteration = 1
        self.current_logs = []
    
    def collect_log(self, log_message):
        self.current_logs.append(log_message)
    
    def new_iteration(self):
        if self.current_logs:
            self.logs[f"iteration_{self.current_iteration}"] = {
                "logs": self.current_logs,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.current_logs = []
            self.current_iteration += 1
    
    def save_logs(self):
        with open('/home/mcnl/Desktop/chaeyoung/flow/examples/training_results/warning_logs.json', 'w') as f:
            json.dump(self.logs, f, indent=2)

# Initialize logger
logger = WarningLogger()

# Setup stdout/stderr redirection
original_stdout = sys.stdout
original_stderr = sys.stderr

class LoggerWriter:
    def write(self, message):
        if "Warning" in message or "Collision" in message:
            logger.collect_log(message.strip())
        original_stderr.write(message)
    
    def flush(self):
        original_stderr.flush()

# Redirect stderr to catch warnings
sys.stderr = LoggerWriter()

# Functions to be called from main training script
def on_iteration_complete():
    logger.new_iteration()

def on_training_end():
    logger.save_logs()

# Export logger instance
__all__ = ['logger', 'on_iteration_complete', 'on_training_end']