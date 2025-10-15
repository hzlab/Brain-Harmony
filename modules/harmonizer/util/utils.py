import numpy as np
import torch
from absl import logging


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def set_logger(log_level="info", fname=None):
    import logging as _logging

    handler = logging.get_absl_handler()
    formatter = _logging.Formatter("%(asctime)s - %(filename)s - %(message)s")
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)
