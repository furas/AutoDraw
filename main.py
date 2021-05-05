#!/usr/bin/env python3

"""
AutoDraw - version with modification

Original (incomplete) code: 

    Austin Nguyen, Jun 1, 2020

    How I Used Machine Learning to Automatically Hand-Draw Any Picture
    Supervised and unsupervised learning made easy!

    https://towardsdatascience.com/how-i-used-machine-learning-to-automatically-hand-draw-any-picture-7d024d0de997

Code completion:

    Bartlomiej "furas" Burek (https://blog.furas.pl)

    date: 2021.05.04
    
Changes:    

    date: 2021.05.04

    - debug messages
    - added pynput.keyboard.Listener to stop drawing on press ESC (when PyAutoGUI moves mouse in wrong place)
    - add doption to set screen size manually (when there are two monitors)
    
Suggestions:

    - add colors in messages - to better recognize when it ask to press Enter`
    - display color value as RGB instead of BGR, and as hex code - to simper copy it to painting program
    
Tested:

    date: 2021.05.04

    - GIMP 2.10 (fullscreen, hidden toolbars, etc)
    - computer with two monitors
    - Linux Mint 20.1 (MATE)
    - Python 3.8.5

# pip install opencv-python
# pip install numpy
# pip install PyAutoGUI
# pip install sklearn
# pip install kdtree
"""

import cv2
import numpy as np
import pyautogui as pg
from sklearn.cluster import KMeans
from kdtree import create
from collections import defaultdict
import operator
import time

# --- colors for Linux terminal ---

# https://misc.flogisoft.com/bash/tip_colors_and_formatting

PRINT_COLORS = True

if PRINT_COLORS:
    C0 = '\033[1;30m'   # color gray/black
    CR = '\033[1;31m'   # color red
    CG = '\033[1;32m'   # color green
    CY = '\033[1;33m'   # color yellow
    CB = '\033[1;34m'   # color blue
    CM = '\033[1;35m'   # color magenta
    CC = '\033[1;36m'   # color cyan
    CW = '\033[1;37m'   # color white
    #BR = '\033[1;41m'   # background color red
    CX = '\033[m'       # reset colors
else:
    C0 = ''   # color gray/black
    CR = ''   # color red
    CG = ''   # color green
    CY = ''   # color yellow
    CB = ''   # color blue
    CM = ''   # color magenta
    CC = ''   # color cyan
    CW = ''   # color white
    CX = ''   # reset colors

print(f'{CY}[DEBUG]{CX} colors: {C0}C0{CR}CR{CG}CG{CB}CB{CY}CY{CM}CM{CC}CC{CW}CW{CX}')

# --- 


class AutoDraw(object):

    def __init__(self, name, blur=0, screen_size=None):
        print(f'{CY}[DEBUG]{CX} __init__')

        # Tunable parameters
        self.detail = 1
        self.scale = 7/12
        self.sketch_before = False
        self.with_color = True
        self.num_colors = 10
        self.outline_again = False

        # Load Image. Switch axes to match computer screen
        self.img = self.load_img(name)
        self.blur = blur
        self.img = np.swapaxes(self.img, 0, 1)
        self.img_shape = self.img.shape
        print(f'{CY}[DEBUG]{CX} img.shape:', self.img.shape)
        
        self.dim = pg.size()
        print(f'{CY}[DEBUG]{CX} dim = pg.size():', self.dim)
        if screen_size:
            self.dim = screen_size
            print(f'{CY}[DEBUG]{CX} dim = screen_size:', self.dim)
        
        # Scale to draw inside part of screen
        self.startX = int(((1 - self.scale) / 2)*self.dim[0])
        self.startY = int(((1 - self.scale) / 2)*self.dim[1])
        self.dim = (self.dim[0] * self.scale, self.dim[1] * self.scale)
        print(f'{CY}[DEBUG]{CX} startX, startY:', self.startX, self.startY)
        print(f'{CY}[DEBUG]{CX} dim (scale):', self.dim, self.scale)

        # fit the picture into this section of the screen
        if self.img_shape[1] > self.img_shape[0]:   # furas change `>`  into `<
            # if it's taller that it is wide, truncate the wide section
            self.dim = (int(self.dim[1] * (self.img_shape[0] / self.img_shape[1])), self.dim[1])
        else:
            # if it's wider than it is tall, truncate the tall section
            self.dim = (self.dim[0], int(self.dim[0] *(self.img_shape[1] / self.img_shape[0])))
        print(f'{CY}[DEBUG]{CX} dim:', self.dim)

        # Get dimension to translate picture. Dimension 1 and 0 are switched due to comp dimensions
        ratio = self.img.shape[0] / self.img.shape[1]
        pseudo_x = int(self.img.shape[1] * self.detail)
        self.pseudoDim = (pseudo_x, int(pseudo_x * ratio))
        print(f'{CY}[DEBUG]{CX} pseudoDim:', self.pseudoDim)

          # Initialize directions for momentum when creating path
        self.maps = {0: (1, 1), 1: (1, 0), 2: (1, -1), 3: (0, -1), 4: (0, 1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1)}
        self.momentum = 1
        self.curr_delta = self.maps[self.momentum]

        return
        # Create Outline
        self.drawing = self.process_img(self.img)
        self.show()

    def load_img(self, name):
        print(f'{CY}[DEBUG]{CX} load_img')

        image = cv2.imread(name)
        return image

    def show(self):
        print(f'{CY}[DEBUG]{CX} show')

        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        print('close window')
        cv2.destroyAllWindows()

    def rescale(self, img, dim):
        print(f'{CY}[DEBUG]{CX} rescale')

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def translate(self, coord):
        #print(f'{CY}[DEBUG]{CX} translate')

        ratio = (coord[0] / self.pseudoDim[1], coord[1] / self.pseudoDim[0]) # this is correct
        deltas = (int(ratio[0] * self.dim[0]), int(ratio[1] * self.dim[1]))

        #print(f'{CY}[DEBUG]{CX} coord:', coord)
        #print(f'{CY}[DEBUG]{CX} pseudoDim:', self.pseudoDim)
        #print(f'{CY}[DEBUG]{CX} ratio:', ratio)
        #print(f'{CY}[DEBUG]{CX} deltas:', deltas)
        #print(f'{CY}[DEBUG]{CX} startX, startY:', self.startX, self.startY)
        
        #print(f'{CY}[DEBUG]{CX} translate', coord, '->', self.startX + deltas[0], self.startY + deltas[1])
        
        return self.startX + deltas[0], self.startY + deltas[1]

    def process_img(self, img):
        print(f'{CY}[DEBUG]{CX} process_img')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.blur == 2:
            gray = cv2.GaussianBlur(gray, (9, 9), 0)
            canny = cv2.Canny(gray, 25, 45)
        elif self.blur == 1:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            canny = cv2.Canny(gray, 25, 45)
        else:  # no blur
            canny = cv2.Canny(gray, 50, 75)
        canny = self.rescale(canny, self.pseudoDim)
        r, res = cv2.threshold(canny, 50, 255, cv2.THRESH_BINARY_INV)

        return res

    def execute(self, commands):
        print(f'{CY}[DEBUG]{CX} execute')

        # furas: Listenter to stop drawing on press `ESC`
        from pynput import keyboard

        global running
    
        def on_release(key):
            global running

            if key == keyboard.Key.esc:
                # Stop 
                running = False
                # Stop listener
                return False
            
        running = True
        
        # furas: Listenter to stop drawing on press `ESC`
        with keyboard.Listener(on_release=on_release) as listener:
       
            press = 0  # flag indicating whether we are putting pressure on paper

            for (i, comm) in enumerate(commands):
                if not running:
                    break
                    
                if type(comm) == str:
                    if comm == 'UP':
                        press = 0
                    if comm == 'DOWN':
                        press = 1
                else:
                    if press == 0:
                        pg.moveTo(comm[0], comm[1], 0)
                    else:
                        pg.dragTo(comm[0], comm[1], 0)
                    
            listener.stop()
            listener.join()
        
        return

    def drawOutline(self):
        print(f'{CY}[DEBUG]{CX} drawOutline')

        indices = np.argwhere(self.drawing < 127).tolist()  # get the black colors
        index_tuples = map(tuple, indices)

        self.hashSet = set(index_tuples)
        self.KDTree = reate(indices)
        self.commands = []
        self.curr_pos = (0, 0)
        point = self.translate(self.curr_pos)
        self.commands.append(point)

        print(f'Change: pen to THIN (small), color to BLACK.')
        input(f"Press {CG}ENTER{CX} once ready")
        print('')

        # DRAW THE BLACK OUTLINE
        self.createPath()
        input(f"Ready! Press {CG}ENTER{CX} to draw")
        print(f'{CY}5 seconds until drawing beings{CX}')
        time.sleep(5)

        self.execute(self.commands)

    def createPath(self):
        print(f'{CY}[DEBUG]{CX} createPath')

        # check for closest point. Go there. Add click down. Change curr. Remove from set and tree. Then, begin
        new_pos = tuple(self.KDTree.search_nn(self.curr_pos)[0].data)
        self.commands.append(new_pos)
        self.commands.append("DOWN")
        self.curr_pos = new_pos
        self.KDTree = self.KDTree.remove(list(new_pos))
        self.hashSet.remove(new_pos)

        hashset_size = len(self.hashSet)
        
        while len(self.hashSet) > 0:
            prev_direction = self.momentum
            candidate = self.checkMomentum(self.curr_pos)
            if self.isValid(candidate):
                new = tuple(map(operator.add, self.curr_pos, candidate))
                new_pos = self.translate(new)
                if prev_direction == self.momentum and type(self.commands[-1]) != str:  # the direction didn't change
                    self.commands.pop()
                self.commands.append(new_pos)
            else:
                self.commands.append("UP")
                new = tuple(self.KDTree.search_nn(self.curr_pos)[0].data)
                new_pos = self.translate(new)
                self.commands.append(new_pos)
                self.commands.append("DOWN")
            self.curr_pos = new
            self.KDTree = self.KDTree.remove(list(new))
            self.hashSet.remove(new)
            print('Making path... number points left: ', len(self.hashSet), '/', hashset_size, '          ', end='\r')
        print()            
        return

    def isValid(self, delta):
        #print(f'{CY}[DEBUG]{CX} isValid')
        return len(delta) == 2

    def checkMomentum(self, point):
        #print(f'{CY}[DEBUG]{CX} checkMomentum')

        # Returns best next relative move w.r.t. momentum and if in hashset
        self.curr_delta = self.maps[self.momentum]
        moments = self.maps.values()
        deltas = [d for d in moments if (tuple(map(operator.add, point, d)) in self.hashSet)]
        deltas.sort(key=self.checkDirection, reverse=True)
        if len(deltas) > 0:
            best = deltas[0]
            self.momentum = [item[0] for item in self.maps.items() if item[1] == best][0]
            return best
        return [-1]

    def checkDirection(self, element):
        #print(f'{CY}[DEBUG]{CX} checkDirection')

        return self.dot(self.curr_delta, element)

    def dot(self, pt1, pt2):
        #print(f'{CY}[DEBUG]{CX} dot')

        pt1 = self.unit(pt1)
        pt2 = self.unit(pt2)
        return pt1[0] * pt2[0] + pt1[1] * pt2[1]

    def unit(self, point):
        #print(f'{CY}[DEBUG]{CX} unit')

        norm = (point[0] ** 2 + point[1] ** 2)
        norm = np.sqrt(norm)
        return point[0] / norm, point[1] / norm

    def run(self):
        print(f'{CY}[DEBUG]{CX} run')

        if self.with_color:
            print('Counting colors ...')

            color = self.rescale(self.img, self.pseudoDim)
            collapsed = np.sum(color, axis=2)/3
            fill = np.argwhere(collapsed < 230)  # color 2-d indices
            fill = np.swapaxes(fill, 0, 1)  # swap to index into color
            RGB = color[fill[0], fill[1], :]
            k_means = KMeans(n_clusters=self.num_colors).fit(RGB)
            colors = k_means.cluster_centers_
            labels = k_means.labels_
            fill = np.swapaxes(fill, 0, 1).tolist()  # swap back to make dictionary
            label_2_index = defaultdict(list)

            for i, j in zip(labels, fill):
                label_2_index[i].append(j)

            print('Number of colors:', len(colors))
            
            for (i, color) in enumerate(colors):
                B, G, R = map(int, color)
                print(f'Change: pen to THICK (big), color to RGB values: R: {CR}{R}{CX} G: {CG}{G}{CX}, B: {CB}{B}{CX} (hex: #{CR}{R:02X}{CG}{G:02X}{CB}{B:02X}{CX})')
                input(f"Press {CG}ENTER{CX} once ready")
                print('')

                points = label_2_index[i]
                index_tuples = map(tuple, points)
                self.hashSet = set(index_tuples)
                self.KDTree = create(points)
                self.commands = []
                self.curr_pos = (0, 0)
                point = self.translate(self.curr_pos)
                self.commands.append(point)
                self.commands.append("UP")
                self.createPath()

                input(f'{CR}Ready!{CX} Press {CG}ENTER{CX} to draw')
                print(f'{CY}5 seconds until drawing begins...{CX}')
                time.sleep(5)

                self.execute(self.commands)
        if self.outline_again:
            self.drawOutline()
        
if __name__ == '__main__':        
    image = '/home/furas/test/lenna.png'
    image = 'autodraw-image-1a.png'

    ad = AutoDraw(image, screen_size=(1920,1200))
    ad.run()
