import os
from data import multiscalesrdata


# 加载DF2k数据
class SRG(multiscalesrdata.SRData):
    def __init__(self, args, name='SRG', train=True, benchmark=False):
        super(SRG, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr = super(SRG, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]

        return names_hr

    def _set_filesystem(self, dir_data):
        super(SRG, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')