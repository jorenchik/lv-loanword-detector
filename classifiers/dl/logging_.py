
import structlog
from pathlib import Path
import sys
import csv

log = structlog.get_logger()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.KeyValueRenderer(key_order=['level', 'timestamp', 'event']),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

class TeeOutput:
    """Write to both console and file."""
    
    def __init__(self, file_path, mode='w'):
        self.terminal = sys.stdout
        self.log = open(file_path, mode, buffering=1)  # line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

class TeeError:
    """Write stderr to both console and file."""
    
    def __init__(self, file_path, mode='w'):
        self.terminal = sys.stderr
        self.log = open(file_path, mode, buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def setup_output_file_log(output_dir: Path):
    """Configure logging to both console and file."""
    log_file = output_dir / "logs" / "training.log"
    err_file = output_dir / "logs" / "training_err.log"
    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeError(err_file)


class EpochLogger:
    """Log epoch metrics to CSV."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.fieldnames = ['epoch', 'train_loss', 'dev_loss', 'learning_rate']
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_epoch(self, epoch: int, train_loss: float, dev_loss: float, lr: float):
        """Append epoch metrics."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': f'{train_loss:.6f}',
                'dev_loss': f'{dev_loss:.6f}',
                'learning_rate': f'{lr:.6e}'
            })

