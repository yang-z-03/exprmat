
import os


class Configuration:

    def __init__(self):

        self.config = {}

        # font family settings

        self.config['plotting.family'] = 'Arial'
        self.config['plotting.font'] = None
        self.config['plotting.font.b'] = None
        self.config['plotting.font.i'] = None
        self.config['plotting.font.bi'] = None

        # target taxa

        self.config['taxa.reference'] = {
            'mm10': 'mmu',
            'hg19': 'hsa',
            'GRCm39': 'mmu',
            'GRCh38': 'hsa'
        }


    def __getitem__(self, index):
        return self.config[index]
    

    def update(self, conf):
        for key in conf:
            if key in self.config:
                self.config[key] = conf[key]
    

    def update_config(self, conf_name, value):
        self.config[conf_name] = value


default = Configuration()
