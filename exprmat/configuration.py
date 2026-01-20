
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

        self.config['plotting.flavor'] = 'light'
        self.config['plotting.backend'] = 'TkAgg'
        self.config['data'] = os.path.join(os.path.dirname(__file__), 'data')

        # target taxa

        self.config['taxa.reference'] = {
            'mm10': 'mmu',   # the slightly modified version from ucsc based on grcm38
            'grcm39': 'mmu',
            'grcm38': 'mmu',

            'hg19': 'hsa',   # the slightly modified version from ucsc based on grch37
            'grch37': 'hsa',
            'grch38': 'hsa',

            'dm6': 'dme'
        }

        self.config['default.assembly'] = {
            'mmu': 'grcm39',
            'hsa': 'grch38',
            'dme': 'dm6'
        }

        self.config['max.image'] = int(10000 * 50000)


    def __getitem__(self, index):
        return self.config[index]
    

    def update(self, conf):
        for key in conf:
            if key in self.config:
                self.config[key] = conf[key]
    

    def update_config(self, conf_name, value):
        self.config[conf_name] = value
