from tensorboardX import SummaryWriter

import pdb

"""
Ref: https://github.com/lanpa/tensorboard-pytorch
"""
class LogWriter(SummaryWriter):
    def __init__(self, log_dir):
        #pdb.set_trace()
        super(LogWriter, self).__init__(str(log_dir))
        self.log_dir = self.file_writer.get_logdir()

    def update_loss(self, loss, step_i, name = 'loss'):
        self.add_scalar(name, loss, step_i)
