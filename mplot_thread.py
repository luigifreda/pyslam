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

import time 
import sys 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import multiprocessing as mp 
from multiprocessing import Process, Queue, Lock, RLock, Value
import ctypes

kPlotSleep = 0.04
kVerbose = False 
kSetDaemon = True   # from https://docs.python.org/3/library/threading.html#threading.Thread.daemon
                    # The entire Python program exits when no alive non-daemon threads are left.
                    
kUseFigCanvasDrawIdle = True 

# global lock for drawing with matplotlib 
mp_lock = RLock()

if kUseFigCanvasDrawIdle:
    plt.ion()
    

# use mplotlib figure to draw in 2d dynamic data
class Mplot2d:
    def __init__(self, xlabel='', ylabel='', title=''):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title 

        self.data = None 
        self.got_data = False 

        self.axis_computed = False 
        self.xlim = [float("inf"),float("-inf")]
        self.ylim = [float("inf"),float("-inf")]    

        self.key = Value('i',0)
        self.is_running = Value('i',1)

        self.handle_map = {}        

        self.queue = Queue()
        self.vp = Process(target=self.drawer_thread, args=(self.queue,mp_lock,self.key,self.is_running,))
        self.vp.daemon = kSetDaemon
        self.vp.start()

    def quit(self):
        self.is_running.value = 0
        self.vp.join(timeout=5)

    def drawer_thread(self, queue, lock, key, is_running):  
        self.init(lock) 
        while is_running.value == 1:
            self.drawer_refresh(queue, lock)                                    
            if kUseFigCanvasDrawIdle:               
                time.sleep(kPlotSleep) 
        print(mp.current_process().name,'closing fig ', self.fig)  
        plt.close(self.fig)              

    def drawer_refresh(self, queue, lock):            
        while not queue.empty():      
            self.got_data = True           
            self.data = queue.get()          
            xy_signal, name, color, marker = self.data 
            #print(mp.current_process().name,"refreshing : signal ", name)            
            if name in self.handle_map:
                handle = self.handle_map[name]
                handle.set_xdata(np.append(handle.get_xdata(), xy_signal[0]))
                handle.set_ydata(np.append(handle.get_ydata(), xy_signal[1]))                
            else: 
                handle, = self.ax.plot(xy_signal[0], xy_signal[1], c=color, marker=marker, label=name)    
                self.handle_map[name] = handle  
        #print(mp.current_process().name,"got data: ", self.got_data) 
        if self.got_data is True:                   
            self.plot_refresh(lock)

    def on_key_press(self, event):
        #print(mp.current_process().name,"key event pressed...", self._key)     
        self.key.value = ord(event.key) # conver to int 
        
    def on_key_release(self, event):
        #print(mp.current_process().name,"key event released...", self._key)             
        self.key.value = 0  # reset to no key symbol
        
    def get_key(self):
        return chr(self.key.value)            

    def init(self, lock):    
        lock.acquire()      
        if kVerbose:
            print(mp.current_process().name,"initializing...") 
        self.fig = plt.figure()
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle() 
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)       
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)               
        #self.ax = self.fig.gca(projection='3d')
        #self.ax = self.fig.gca()
        self.ax = self.fig.add_subplot(111)   
        if self.title is not '':
            self.ax.set_title(self.title) 
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)	   
        self.ax.grid()		
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.refresh()     
        lock.release()

    def setAxis(self):		                     
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()        

    def draw(self, xy_signal, name, color='r', marker='.'):    
        if self.queue is None:
            return
        self.queue.put((xy_signal, name, color, marker))

    def updateMinMax(self, np_signal):
        xmax,ymax = np.amax(np_signal,axis=0)
        xmin,ymin = np.amin(np_signal,axis=0)        
        cx = 0.5*(xmax+xmin)
        cy = 0.5*(ymax+ymin) 
        if False: 
            # update maxs       
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax 
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax                   
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin   
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin        
        # make axis actually squared
        if True:
            smin = min(xmin,ymin)                                            
            smax = max(xmax,ymax)            
            delta = 0.5*(smax - smin)
            self.xlim = [cx-delta,cx+delta]
            self.ylim = [cy-delta,cy+delta]   
        self.axis_computed = True   

    def plot_refresh(self, lock):
        if kVerbose:        
            print(mp.current_process().name,"refreshing ", self.title)          
        lock.acquire()         
        self.setAxis()
        if not kUseFigCanvasDrawIdle:        
            plt.pause(kPlotSleep)
        lock.release()

    # fake 
    def refresh(self):
        pass 


# use mplotlib figure to draw in 3D trajectories 
class Mplot3d:
    def __init__(self, title=''):
        self.title = title 

        self.data = None  
        self.got_data = False 

        self.axis_computed = False 
        self.xlim = [float("inf"),float("-inf")]
        self.ylim = [float("inf"),float("-inf")]
        self.zlim = [float("inf"),float("-inf")]        

        self.handle_map = {}     
        
        self.key = Value('i',0)
        self.is_running = Value('i',1)         

        self.queue = Queue()
        self.vp = Process(target=self.drawer_thread, args=(self.queue,mp_lock, self.key, self.is_running,))
        self.vp.daemon = kSetDaemon
        self.vp.start()

    def quit(self):
        self.is_running.value = 0
        self.vp.join(timeout=5)     
        
    def drawer_thread(self, queue, lock, key, is_running):  
        self.init(lock) 
        while is_running.value == 1:
            self.drawer_refresh(queue, lock)   
            if kUseFigCanvasDrawIdle:               
                time.sleep(kPlotSleep)    
        print(mp.current_process().name,'closing fig ', self.fig)     
        plt.close(self.fig)                                 

    def drawer_refresh(self, queue, lock):            
        while not queue.empty():    
            self.got_data = True  
            self.data = queue.get()  
            traj, name, color, marker = self.data         
            np_traj = np.asarray(traj)        
            if name in self.handle_map:
                handle = self.handle_map[name]
                self.ax.collections.remove(handle)
            self.updateMinMax(np_traj)
            handle = self.ax.scatter3D(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2], c=color, marker=marker)
            handle.set_label(name)
            self.handle_map[name] = handle     
        if self.got_data is True:               
            self.plot_refresh(lock)          

    def on_key_press(self, event):
        #print(mp.current_process().name,"key event pressed...", self._key)     
        self.key.value = ord(event.key) # conver to int 
        
    def on_key_release(self, event):
        #print(mp.current_process().name,"key event released...", self._key)             
        self.key.value = 0  # reset to no key symbol
        
    def get_key(self):
        return chr(self.key.value) 
    
    def init(self, lock):
        lock.acquire()      
        if kVerbose:
            print(mp.current_process().name,"initializing...") 
        self.fig = plt.figure()
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle()         
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)       
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)             
        self.ax = self.fig.gca(projection='3d')
        if self.title is not '':
            self.ax.set_title(self.title)     
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')		   		

        self.setAxis()
        lock.release() 

    def setAxis(self):		
        #self.ax.axis('equal')   # this does not work with the new matplotlib 3    
        if self.axis_computed:	
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)  
            self.ax.set_zlim(self.zlim)                             
        self.ax.legend()
        #We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()            

    def drawTraj(self, traj, name, color='r', marker='.'):
        if self.queue is None:
            return
        self.queue.put((traj, name, color, marker))

    def updateMinMax(self, np_traj):
        xmax,ymax,zmax = np.amax(np_traj,axis=0)
        xmin,ymin,zmin = np.amin(np_traj,axis=0)        
        cx = 0.5*(xmax+xmin)
        cy = 0.5*(ymax+ymin)
        cz = 0.5*(zmax+zmin) 
        if False: 
            # update maxs       
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax 
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax 
            if zmax > self.zlim[1]:
                self.zlim[1] = zmax                         
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin   
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin        
            if zmin < self.zlim[0]:
                self.zlim[0] = zmin     
        # make axis actually squared
        if True:
            #smin = min(self.xlim[0],self.ylim[0],self.zlim[0])                                            
            #smax = max(self.xlim[1],self.ylim[1],self.zlim[1])
            smin = min(xmin,ymin,zmin)                                            
            smax = max(xmax,ymax,zmax)            
            delta = 0.5*(smax - smin)
            self.xlim = [cx-delta,cx+delta]
            self.ylim = [cy-delta,cy+delta]
            self.zlim = [cz-delta,cz+delta]      
        self.axis_computed = True   

    def plot_refresh(self, lock):
        if kVerbose:        
            print(mp.current_process().name,"refreshing ", self.title)          
        lock.acquire()          
        self.setAxis()
        if not kUseFigCanvasDrawIdle:        
            plt.pause(kPlotSleep)      
        lock.release()

    # fake 
    def refresh(self):
        pass         