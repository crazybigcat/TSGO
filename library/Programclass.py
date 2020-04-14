import torch
import time
from library import BasicFunctions_szz


class Program:

    def __init__(self, device='cuda', dtype=torch.float64, debug_mode=False):
        self.debug_mode = debug_mode
        self.program_info = dict()
        self.device_type = device
        self.device = BasicFunctions_szz.get_best_gpu(device=device)
        self.dtype = dtype
        self.calculate_program_info_time(mode='start')

    def print_program_info(self, mode='start'):
        if mode == 'start':
            print('This program starts at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(BasicFunctions_szz.sort_dict(self.para))
        elif mode == 'end':
            print('This program ends at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(
                'This program consumes ' +
                str(self.program_info['end_time']['time'] - self.program_info['start_time']['time']) +
                ' seconds of wall time.')

    def calculate_program_info_time(self, mode='start'):
        if mode == 'start':
            self.program_info['start_time'] = dict()
            self.program_info['start_time']['time'] = time.time()
            self.program_info['start_time']['clock'] = time.clock()
            self.program_info['start_time']['local'] = time.clock()
        elif mode == 'end':
            self.program_info['end_time'] = dict()
            self.program_info['end_time']['time'] = time.time()
            self.program_info['end_time']['clock'] = time.clock()
            self.program_info['end_time']['local'] = time.clock()

    def name_md5_generate(self):
        if not self.debug_mode:
            self.program_info['save_name'] = BasicFunctions_szz.name_generator_md5(
                self.program_info['path_save'], self.program_info['program_name'], self.para)

    def integrate_codebook(self):
        BasicFunctions_szz.integrate_codebook(
            self.program_info['path_save'], self.program_info['program_name'])