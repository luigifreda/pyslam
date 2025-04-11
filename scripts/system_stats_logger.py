#!/usr/bin/env python3
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

# This script allows to monitor the CPU load (global and on each core) and GPU load of the system.  
# The script can be used to:
# - first, log CPU and GPU activities in a CSV file
# - then, plot the logged data.

# How to use this script?
# First, run 
# $ ./system_stats_logger.py 
# in order to create a new CSV log file in /tmp.
# Then, run
# $ ./system_stats_logger.py --input-file /tmp/system_stats_log_< date and time of your last generated output>.csv
# in order to plot and analyze the data in the created CSV file.

try:
    import sys
    import argparse
    import os
    import io 
    import signal 
    import datetime
    import csv
    import numpy as np 
    import matplotlib.pyplot as plt
    import time
    import socket                  
    import platform
except ImportError as e:
    print("[ERROR] in system_stats_logger.py " + str(e))
    sys.exit(1)
    
try: 
    import psutil             # you can run: $ pip install psutil
except ImportError as e:
    print("[ERROR] in system_stats_logger.py " + str(e))
    print("please, run: $ pip install psutil")
    sys.exit(1)    
try: 
    import GPUtil as gputil   # you can run: $ pip install gputil   (check https://github.com/anderskm/gputil)
except ImportError as e:
    print("[ERROR] in system_stats_logger.py " + str(e))
    print("please, run: $ pip install psutil")
    sys.exit(1)    
            
if "tegra" in platform.platform().lower(): 
    try: 
        print(f'tegra platform detected: {platform.platform()}')
        from jtop import jtop, JtopException # you can run $ sudo -H pip install -U jetson-stats  (check https://github.com/rbonghi/jetson_stats)
    except ImportError as e:
        print("[ERROR] in system_stats_logger.py " + str(e))
        print("please, install jtop with: sudo -H pip install -U jetson-stats")
        sys.exit(1)    



kVerbose = False  # set this to True for debugging

kHostName = socket.gethostname()
kDateTimeStr = time.strftime("%Y%m%d-%H%M%S")
kOutputPath = f'/tmp/system_stats_log_{kHostName}_{kDateTimeStr}.csv'

kSuptitleFontsize = 18
kSubfigTitleFontsize = 14
kSubfigXlabelFontsize = 14
kFigSizeInches = (30, 24)


def utc_now():
    now = (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
    return now

def is_sys_linux(): 
    return 'linux' in sys.platform

class SystemStats:
    def __init__(self, args=None):
        self.args = args 
        self.cpu_total = None 
        self.num_cpu_cores = None 
        self.cpu_load_per_core = None 
        self.ram_memory_perc= None 
        self.ram_memory_tot= None
        self.swap_memory_perc = None   
        self.swap_memory_tot = None
        self.timestamp = None    
        self.sensors_temperatures = None   
        self.sensors_battery = None        
        self.gpus = None   
        self.jetson_info = None 
        self.number_observations = 0 
        self.process_name = args.process_name 
        self.is_process_running = False 
        self.process_cpu_percent = 0
        self.process_memory_percent = 0
        
        
    def get(self):        
        self.get_system()        
        if self.process_name: 
            self.get_process()

    def get_process(self):
        process = None 
        # place holders 
        self.is_process_running = False 
        self.process_cpu_percent = 0 
        self.process_memory_percent = 0 
                
        # get the process data from psutil
        #candidates = [proc for proc in psutil.process_iter() if self.process_name in proc.name()]
        candidates = [proc for proc in psutil.process_iter() if self.process_name == proc.name()]
        if candidates:
            self.is_process_running = True 
            process = candidates[0] # get the first one     
                       
        if not process:
            #print(f'\nno process with name {self.process_name} is running\n') 
            if self.args.wait_process: 
                return 
            else: 
                sys.exit(1)
                
        try: 
            self.process_cpu_percent = process.cpu_percent(interval=self.args.time_interval)/self.num_cpu_cores
            self.process_memory_percent = process.memory_percent()   
        except psutil.Error:
            print("Error: ", psutil.Error)         
                    
        
    def get_system(self):
        # check https://psutil.readthedocs.io/en/latest/
        self.number_observations += 1
        self.timestamp = utc_now()
        try:         
            self.cpu_total = psutil.cpu_percent(interval=self.args.time_interval)  # current system-wide CPU utilization
            self.cpu_load_per_core = psutil.cpu_percent(interval=self.args.time_interval, percpu=True) # a list of floats representing the utilization as a percentage for each CPU (First element of the list refers to first CPU, second element to second CPU and so on)
            #self.num_cpu_cores = psutil.cpu_count(logical=False)
            self.num_cpu_cores = len(self.cpu_load_per_core)
            self.ram_memory_perc= psutil.virtual_memory().percent  # RAM perc used
            self.ram_memory_tot= (psutil.virtual_memory().total / 1024 / 1024 / 1024)  # RAM total in GB            
            self.swap_memory_perc = psutil.swap_memory().percent  # Swap perc used
            self.swap_memory_tot= (psutil.swap_memory().total / 1024 / 1024 / 1024)  # Swap total in GB  
        except psutil.Error:
            print("Error: ", psutil.Error)  
                    
        self.get_gpu()
        self.get_sensors_info()
        
        
    def get_gpu(self):        
        # GPU management
        try:          
            self.gpus = gputil.getGPUs()
        except psutil.Error:
            print("Error: ", psutil.Error)  
                        
        if not self.gpus:
            # try tegra 
            with jtop() as jetson:
                # Read tegra stats with jtop 
                self.jetson_info = SystemJetsonStatus(jetson.stats, jetson.cpu, jetson.gpu, jetson.temperature, jetson.power, jetson.ram, jetson.emc)


    def get_sensors_info(self):
        # Get other information if available 
        try:
            self.sensors_temperatures = psutil.sensors_temperatures() # return hardware temperatures. Each entry is a named tuple representing a certain hardware temperature sensor
        except: 
            self.sensors_temperatures = None 
        try:    
            self.sensors_battery = psutil.sensors_battery() 
        except: 
            self.sensors_battery = None  
        return self 

    def dump(self): 
        if self.number_observations == 0:
            return         
        if self.number_observations == 1:
            print(f'logging into file: {self.args.output_file}')   
        print('----------------')                     
        if self.args.minimal_process_log and self.process_name:
            self.dump_process()
        else:
            self.dump_all()
        
    def dump_process(self):
        # dump info to std output
        if self.process_name: 
            is_running_string = "" if self.is_process_running else "[not running]"
            process_string = f'• Process: {self.process_name}{is_running_string} | CPU %: {self.process_cpu_percent} | RAM % {self.process_memory_percent:.2f}'
            print(process_string)  
                 
    def dump_all(self):
        # dump info to std output    
        self.dump_process()
        
        cpu_string=''          
        for cpu_load in self.cpu_load_per_core: 
            cpu_string += (str(cpu_load) + ' ')            
        print(f'• CPU Tot %: {self.cpu_total} | N Cores: {self.num_cpu_cores} | CPU % xCore: {cpu_string}') 
        print(f'• RAM %: {self.ram_memory_perc} | RAM tot GB: {self.ram_memory_tot:.2f} | Swap %: {self.swap_memory_perc} | Swap tot GB: {self.swap_memory_tot:.2f}')
        
        if self.sensors_temperatures:
            print(f'• Temperature: {str(self.sensors_temperatures)}')
        else: 
            print('• Temperature: no sensors temperature info available')
        if self.sensors_battery:
            print(f'• Battery: {str(self.sensors_battery)}')     
        else: 
            print('• Battery: no sensors battery info available')    
            
        if self.gpus:
            for gpu in self.gpus:
                gpu_load_perc = gpu.load*100
                gpu_mem_util_perc = gpu.memoryUtil*100
                # from https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py#L213 
                print(f'• GPU ID/name: {gpu.id:2d}/{gpu.name} | Load %: {gpu_load_perc:.2f} | Mem %: {gpu_mem_util_perc:.2f} | Temp. {gpu.temperature:.0f}C | Mem Tot: {gpu.memoryTotal:.0f}MB | Mem Used: {gpu.memoryUsed:.0f}MB | Mem Free: {gpu.memoryFree:.0f}MB')# || Diplay mode: {gpu.display_mode:s} | Display active: {gpu.display_active:s}')

        if self.jetson_info:
            print('• ' + self.jetson_info)        
            
    def log(self, file, separator=','): 
        if self.args.minimal_process_log and self.process_name:
            self.log_process(file, separator)
        else:
            self.log_all(file, separator)
        
    def log_process(self, file, separator=','): 
        if self.number_observations == 0:
            return          
        description_comment = f'# timestamp'
        if self.process_name: 
            description_comment += f',process_name, process_cpu_perc, process_mem_perc'        
        
        if self.number_observations == 1:
            # write description_comment on first line 
            file.write(description_comment + '\n')
            
        file.write(str(self.timestamp) + separator)            
        if self.process_name: 
            file.write(str(self.process_name) + separator)          
            file.write(str(self.process_cpu_percent) + separator)  
            file.write(str(self.process_memory_percent) )  # last one without a separator                                    
        file.write('\n')
                
    def log_all(self, file, separator=','): 
        if self.number_observations == 0:
            return         
        # create a first description comment line  
        description_comment = f'# timestamp'
        if self.process_name: 
            description_comment += f',process_name, process_cpu_perc, process_mem_perc'        
        cpu_cores_comment = ''
        for i in range(0,self.num_cpu_cores):
            cpu_cores_comment += f'cpu{i},' 
        description_comment += f', cpu total, num_cpu_cores, {cpu_cores_comment} ram_perc, ram_tot, swap_perc, swap_tot'  # no need of a comma after {cpu_cores_comment}
        if self.gpus: 
            for gpu in self.gpus:
                id = gpu.id
                description_comment += f', gpu{id}_load_perc, gpu{id}_mem_perc, gpu{id}_temperature'
        if self.jetson_info:
            # add 'jetson' in the keys so as to trigger a different set of plots 
            description_comment += f', gpu_jetson_load_perc, gpu_jetson_mem_perc, emc_jetson_perc, gpu_jetson_freq_MHz, gpu_jetson_temperature, power_jetson_usage_cur_mW, power_jetson_usage_avg_mW'

        if self.number_observations == 1:
            # write description_comment on first line 
            file.write(description_comment + '\n')
            
        file.write(str(self.timestamp) + separator)            
        if self.process_name: 
            file.write(str(self.process_name) + separator)          
            file.write(str(self.process_cpu_percent) + separator)  
            file.write(str(self.process_memory_percent) + separator)                                        
        file.write(str(self.cpu_total) + separator)
        file.write(str(self.num_cpu_cores) + separator)
        for cpu_load in self.cpu_load_per_core: 
            file.write(str(cpu_load) + separator)
        file.write(str(self.ram_memory_perc) + separator)
        file.write(str(self.ram_memory_tot) + separator)    
        file.write(str(self.swap_memory_perc) + separator )             
        file.write(str(self.swap_memory_tot))  # last one without a separator 
        
        if self.gpus: 
            for gpu in self.gpus:
                id = gpu.id
                gpu_load_perc = gpu.load*100
                gpu_mem_util_perc = gpu.memoryUtil*100
                file.write(separator) # recover missing separator 
                file.write(str(gpu_load_perc) + separator)
                file.write(str(gpu_mem_util_perc) + separator)
                file.write(str(gpu.temperature))  # last one without a separator       

        if self.jetson_info:
            gpu_load_perc = self.jetson_info.gpu['val']
            gpu_mem_util_perc = (float(self.jetson_info.ram['use'])/float(self.jetson_info.ram['tot'])) * 100.0
            #print(f'gpu_mem_util_perc: {gpu_mem_util_perc}')
            emc_perc = self.jetson_info.emc['val']
            gpu_freq_MHz = float(self.jetson_info.gpu['frq'])/1000
            gpu_temperature = self.jetson_info.temperature['GPU']
            power_usage_cur_mW = self.jetson_info.power[0]['cur']
            power_usage_avg_mW = self.jetson_info.power[0]['avg']
            file.write(separator)  # recover missing separator 
            file.write(str(gpu_load_perc) + separator)
            file.write(str(gpu_mem_util_perc) + separator)
            file.write(str(emc_perc) + separator)
            file.write(str(gpu_freq_MHz) + separator)
            file.write(str(gpu_temperature) + separator)  
            file.write(str(power_usage_cur_mW) + separator)
            file.write(str(power_usage_avg_mW))   # last one without a separator

        file.write('\n')
            
class SystemJetsonStatus:
    def __init__(self, stats=None, cpu=None, gpu=None, temperature=None, power=None, ram=None, emc=None):
        # https://github.com/rbonghi/jetson_stats/blob/master/examples/jtop_properties.py
        self.stats = stats
        self.cpu = cpu
        self.gpu = gpu
        self.temperature = temperature
        self.power = power
        self.ram = ram 
        self.emc = emc  # EMC is the external memory controller, through which all sysmem/carve-out/GART memory accesses go.
        
    def __str__(self): 
        out = '\nJetson\n'
        # CPU
        out += ' * Stats: '
        out += str(self.stats) + '\n'
        # CPU
        out += ' * CPUs: '
        out += str(self.cpu) + '\n'
        # GPU
        out += ' * GPU: '
        out += str(self.gpu) + '\n'
        # Temperature
        out += ' * temperature: '
        out += str(self.temperature) + '\n'
        # Power
        out += ' * power: '
        out += str(self.power) + '\n'
        # RAM
        out += ' * RAM: '
        out += str(self.ram) + '\n'   
        # EMC 
        out += ' * EMC: '
        out += str(self.emc) + '\n'       
        return out 

    
def parse_csv(file_path):
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        names = None
        data = {}
        data['filename'] = os.path.basename(file_path)
        for row in csv_reader:      
            if line_count == 0:
                names = row
                for i in range(0,len(names)):
                    if i == 0:
                        names[0] = names[0][1:].strip()
                    else:
                        names[i] = names[i].strip() 
                print(f'names: {names}')
            else: 
                assert(len(names) == len(row)) 
                for i in range(0,len(row)):
                    if line_count == 1:
                        data[names[i]] = []
                    data[names[i]].append(row[i])         
            line_count += 1
        data['keys'] = names 
        return data 
    return None

    
# Save figure with nice margins
def save_fig(figure, name, dpi = 300, bbox_inches = 0, pad_inches = 0.1):
    print(f'saving figure: {name}')
    figure.set_size_inches(24, 18)    
    figure.savefig(name, dpi = dpi, bbox_inches = bbox_inches, pad_inches = pad_inches, orientation='landscape')


def draw_data(data, args): 
    assert(data is not None) 
    memory_data_to_draw = ['ram_perc', 'swap_perc']
    filename = data['filename']
    names = data['keys']
    #print(f'names: {names}')
    timestamps = np.array(data['timestamp'], dtype=float)
    timestamps = np.subtract(timestamps, timestamps[0])
    process_name = None 

    has_tegra_stats = len([name for name in names if 'jetson' in name])>0
    
    has_process_data = len([name for name in names if 'process_' in name])>0

    # retrieve total ram memory and total swap memory (they are constants)
    ram_tot_string = ''
    swap_tot_string = ''
    try: 
        ram_tot = float(data['ram_tot'][0])
        ram_tot_string = f', RAM tot: {ram_tot:.2f} GB'
    except: 
        pass 
    try: 
        swap_tot = float(data['swap_tot'][0])
        swap_tot_string = f', SWAP tot: {swap_tot:.2f} GB'
    except: 
        pass     
    
    if not has_tegra_stats:
        fig, [ax_cpu, ax_mem, ax_gpu, ax_gpu_temp] = plt.subplots(4)
        fig2 = None 
        ax_gpu_freq = None 
        ax_power = None 
    else:
        fig, [ax_cpu, ax_mem] = plt.subplots(2)
        fig2, [ax_gpu, ax_gpu_temp, ax_gpu_freq, ax_power] = plt.subplots(4)
        
    fig3 = None       
    ax_proc_cpu = None   
    ax_proc_mem = None 
    if has_process_data: 
        process_name = data['process_name'][0]
        fig3, [ax_proc_cpu, ax_proc_mem] = plt.subplots(2)
    
    # set big figures (otherwise tight_layout() won't work below)

    fig.set_size_inches(kFigSizeInches)
    if fig2:
        fig2.set_size_inches(kFigSizeInches)  
    if fig3:
        fig3.set_size_inches(kFigSizeInches)          

    fig.suptitle('Data from: ' + filename, fontsize=kSuptitleFontsize)
    if fig2:
        fig2.suptitle('Data from: ' + filename, fontsize=kSuptitleFontsize) 
    if fig3:
        fig3.suptitle(f'Process: {process_name}, Data from: {filename}', fontsize=kSuptitleFontsize)         

    ax_cpu.set_title('CPU load %', fontsize=kSubfigTitleFontsize)
    ax_mem.set_title(f'Memory load % {ram_tot_string}{swap_tot_string}', fontsize=kSubfigTitleFontsize)
    ax_gpu.set_title('GPU load %', fontsize=kSubfigTitleFontsize)
    ax_gpu_temp.set_title('GPU temperature', fontsize=kSubfigTitleFontsize)
    if ax_gpu_freq:
        ax_gpu_freq.set_title('GPU frequency (MHz)', fontsize=kSubfigTitleFontsize)
    if ax_power:
        ax_power.set_title('Power (mW)', fontsize=kSubfigTitleFontsize)

    if fig3: 
        ax_proc_cpu.set_title('CPU load %', fontsize=kSubfigTitleFontsize)
        ax_proc_mem.set_title('Memory load %', fontsize=kSubfigTitleFontsize)        
        
        for i,name in enumerate(names): 
            if 'process' in name:   
                if 'cpu' in name:                       
                    ax_proc_cpu.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5) 
                elif 'mem' in name: 
                    ax_proc_mem.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)

    for i,name in enumerate(names): 
        if 'process' in name: 
            continue 
        if 'cpu' in name and name != 'num_cpu_cores':
            #print('drawing', name)
            ax_cpu.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)
        elif name in memory_data_to_draw:  
            #print('drawing', name)
            ax_mem.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)     
        elif 'emc' in name:
            ax_mem.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)         
        elif 'gpu' in name and 'perc' in name:
            ax_gpu.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)                  
        elif 'gpu' in name and 'temperature' in name:
            ax_gpu_temp.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)  
        elif 'gpu' in name and 'freq' in name and ax_gpu_freq:
            ax_gpu_freq.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)  
        elif 'power' in name and ax_power:
            ax_power.plot(timestamps, np.array(data[name],dtype=float), label=name, alpha=0.5)  

    if ax_proc_cpu:
        ax_proc_cpu.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
        ax_proc_cpu.set_ylabel('CPU %')            
        ax_proc_cpu.legend(loc='upper right', frameon=False)
        ax_proc_cpu.grid(True)
    
    if ax_proc_mem:
        ax_proc_mem.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
        ax_proc_mem.set_ylabel('Memory %')            
        ax_proc_mem.legend(loc='upper right', frameon=False)
        ax_proc_mem.grid(True)
    
    ax_cpu.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
    ax_cpu.set_ylabel('CPU %')            
    ax_cpu.legend(loc='upper right', frameon=False)
    ax_cpu.grid(True)
    
    ax_mem.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
    ax_mem.set_ylabel('Memory %')            
    ax_mem.legend(loc='upper right', frameon=False)
    ax_mem.grid(True)
    
    ax_gpu.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
    ax_gpu.set_ylabel('GPU %')            
    ax_gpu.legend(loc='upper right', frameon=False)
    ax_gpu.grid(True)

    ax_gpu_temp.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
    ax_gpu_temp.set_ylabel('C°')            
    ax_gpu_temp.legend(loc='upper right', frameon=False)
    ax_gpu_temp.grid(True)

    if ax_gpu_freq:
        ax_gpu_freq.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
        ax_gpu_freq.set_ylabel('MHz')            
        ax_gpu_freq.legend(loc='upper right', frameon=False)
        ax_gpu_freq.grid(True)

    if ax_power:
        ax_power.set_xlabel('time', fontsize=kSubfigXlabelFontsize)
        ax_power.set_ylabel('mW')            
        ax_power.legend(loc='upper right', frameon=False)
        ax_power.grid(True)

    # NOTE: concerning tight_layout(), you may provide an optional rect parameter, which specifies the bounding box 
    # that the subplots will be fit inside. The coordinates must be in normalized figure coordinates and 
    # the default is (0, 0, 1, 1). From https://matplotlib.org/2.0.2/users/tight_layout_guide.html 
    layout_rect_delta = 0.03
    layout_rect = [layout_rect_delta, layout_rect_delta, (1.0-layout_rect_delta), (1.0-layout_rect_delta)]
    layout_pad = 5
    #fig.tight_layout()  # avoid text overlapping
    fig.tight_layout(rect=layout_rect, pad=layout_pad)   # avoid text overlapping 
    
    if args.save_plots: 
        save_fig(fig, f'{filename}_time_plot1.png')
    if fig2: 
        fig2.tight_layout(rect=layout_rect, pad=layout_pad)  # avoid text overlapping 
        if args.save_plots:
            save_fig(fig2, f'{filename}_time_plot2.png') 
    plt.show()


def signal_handler(sig, frame):
    print('\n{} intercepted Ctrl+C!'.format(os.path.basename(__file__)))
    file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Create a log for cpu and memory load')
    argparser.add_argument("-o", "--output-file", type=str, default=kOutputPath, help="Path of the log file that will be generated.")
    argparser.add_argument("-t", "--time-interval", type=int, default=1, help="Interval of each observation.")
    argparser.add_argument("-i", "--input-file", type=str, default=None, help="Path of the input file we want to draw.")
    argparser.add_argument("-p", "--process-name", type=str, default=None, help="Optional process to monitor. If nothing, all system is monitored")
    argparser.add_argument("-w", "--wait-process", default=False, action='store_true', help="Wait for the input process name (if provided in input). Otherwise, exit when process is not running.")
    argparser.add_argument("-m", "--minimal-process-log", default=False, action='store_true', help="Just log the minimal information about the input process name. Otherwise, log all")    
    argparser.add_argument("-s", "--save-plots", default=False, action='store_true', help="Save the plots in case you give an input file.")
    argparser.add_argument("--sleep-start", type=float, default=0, help="Starting sleep time in seconds.")

    args = argparser.parse_args()
    
    if args.sleep_start>0: 
        print(f'sleeping before start for {args.sleep_start} seconds')
        time.sleep(args.sleep_start)
    
    system_stats = SystemStats(args) 
    
    # let's check if we want just to draw things 
    if args.input_file: 
        print('parsing and drawing input file ' + args.input_file)
        data = parse_csv(args.input_file)
        #print('data: ' + str(data))
        draw_data(data, args)
        sys.exit(0)
    else: 
        file = io.open(args.output_file, "w", encoding='utf-8')

        while True: 
            system_stats.get()
            system_stats.dump()
            system_stats.log(file)

        f.close()
