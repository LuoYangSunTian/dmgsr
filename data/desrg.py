import os
from data import multiscalesrdata


# 加载DF2k数据
class DESRG(multiscalesrdata.SRData):
    def __init__(self, args, name='SRG', train=True, benchmark=False):
        super(DESRG, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr = super(DESRG, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]

        return names_hr

    def _set_filesystem(self, dir_data):
        super(DESRG, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')