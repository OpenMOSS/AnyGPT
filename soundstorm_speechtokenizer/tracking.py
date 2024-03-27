from accelerate.tracking import TensorBoardTracker, on_main_process
from typing import Any, Dict, List, Optional, Union
import os

class MyTensorBoardTracker(TensorBoardTracker):
    
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super.__init__(run_name, logging_dir, **kwargs)
        
    @on_main_process
    def log_audios(self, values: dict, step: Optional[int], sample_rate=16000, **kwargs):
        
        for k, v in values.items():
            self.writer.add_audio(k, v, global_step=step, sample_rate=sample_rate, **kwargs)