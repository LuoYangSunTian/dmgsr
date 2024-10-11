import os
from data import multiscalesrdata


# 加载DF2k数据
class DGSR(multiscalesrdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        super(DGSR, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr = super(DGSR, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]

        return names_hr

    def _set_filesystem(self, dir_data):
        super(DGSR, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')