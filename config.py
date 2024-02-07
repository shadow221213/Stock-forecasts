# ----------------参数配置----------------- #
class Arg:
    def __init__(self):
        self.data_path = './data'
        self.codes = ['sh518880', 'sz159937']
        
        self.rnn_unit = 256
        self.input_size = 6
        self.output_size = 1
        self.layer_num = 3
        
        self.lr = 0.001
        self.dropout = 0.1
        
        self.epoch = 200
        self.batch_size = 64
        self.time_step = 30
        
        self.ratio = 0.8
        self.cutoff = 0.68
