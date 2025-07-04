
import os


class Configuration:

    def __init__(self):

        self.config = {}

        # font family settings

        self.config['plotting.font'] = [
            'Helvetica Neue LT Std', 
            'Helvetica', 
            'Arial', 
            'Ubuntu', 
            'Verdana'
        ]
        
        self.config['backend'] = 'TkAgg'
        self.config['data'] = os.path.join(os.path.dirname(__file__), 'data')

        # target taxa

        self.config['taxa.reference'] = {
            'mm10': 'mmu', # alias of grcm38
            'grcm39': 'mmu',
            'grcm38': 'mmu',

            'hg19': 'hsa', # alias of grch37
            'grch37': 'hsa',
            'grch38': 'hsa',
        }

        self.config['default.assembly'] = {
            'mmu': 'grcm39',
            'hsa': 'grch38'
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
