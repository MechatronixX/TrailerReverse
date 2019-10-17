# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:29:07 2019

@author: root
"""
# Import dependencies
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import matplotlib as mpl
from matplotlib import animation
import matplotlib.patches as patches

class animateRectangle: 
    
    #rectAnim = animateRectangle(fig, B,L, Px, Py, heading)
    def __init__( self,fig, B,L, Px, Py, angle):
        """Specify width and length of rectangle and a handle to a plot axis object """
        self.B = B
        self.L = L
        self.fig = fig
        
        #Data of angle and position of the rectangle. 
        self.Px = Px
        self.Py = Py
        self.angle = angle 
        #self.ax = ax
        boxSize = L*2
        self.ax = plt.axes(xlim=(-boxSize, boxSize), ylim=(-boxSize, boxSize))
        
        self.rectangle = patches.Rectangle((0, 0), 0.01*L, L, fc='r', ec = 'r')
#plt.gca().add_patch(rectangle)

        self.line, = self.ax.plot([], [],'--', lw=1)
        self.origin, = self.ax.plot([], [],'ko')
        
    def animate(self, interval): 
        Nframes = len(self.Px)
        
        return  animation.FuncAnimation(self.fig, 
                                       self.__runAnimation__, 
                                       init_func=self.__initAnimation__, 
                                       frames=Nframes, 
                                       interval=interval,
                                       blit=True, 
                                       repeat = False)
        
     
    #Functions for the animation funcion in matplotlib    
    def __initAnimation__(self):
        self.ax.add_patch(self.rectangle)
        self.line.set_data([], [])
    
        return self.rectangle, self.line

    def __runAnimation__(self, i):
    
        #TODO: Rectangle is shown left of the track it travels, should be fixed. 
        self.origin.set_data(self.Px[i], self.Py[i])
        self.rectangle.set_xy((self.Px[i], self.Py[i]) )
        ##Rotate the rectangle 
        #t2 = mpl.transforms.Affine2D().rotate_deg( np.rad2deg(heading[i])  )  + ax.transData
        
        #Rotate the body around the origin. Info: 
        # https://stackoverflow.com/questions/15557608/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-a
        t2 = mpl.transforms.Affine2D().rotate_deg_around(self.Px[i], self.Py[i], np.rad2deg(self.angle[i])  ) + self.ax.transData
        self.rectangle.set_transform(t2)
        
        #rectangle._angle = heading[i]
        #Return all objects that were changed. 
        
        self.line.set_data(self.Px[0:i], self.Py[0:i])
        
        return self.rectangle, self.line, self.origin
        
    