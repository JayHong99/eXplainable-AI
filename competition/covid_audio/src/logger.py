from pathlib import Path
from datetime import datetime


class logger : 
    def __init__(self, log_path : Path) : 
        log_path.mkdir(exist_ok=True)
        log_name = self.current_time() 
        self.log_save_path = log_path.joinpath(log_name).with_suffix('.txt')

    def __call__(self, log) : 
        print(f'[{self.current_time()}] {log}')
        with open(self.log_save_path, 'a') as f  :
            f.write(log + '\n')

    def current_time(self) : 
        return datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')