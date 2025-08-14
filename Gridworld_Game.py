from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import cv2 
import random
import os
import sys
from IPython.display import clear_output
import matplotlib.pyplot as plt

def find_legal_location(items_locs:list,Grid_size:int,all_locations) -> list:
    unlegal_idxs = []
    for loc in items_locs:
        unlegal_idxs.append(all_locations.index(tuple(loc)))
    legal_idxs = [x for x in range(len(all_locations)) if x not in unlegal_idxs]
    legal_locs = [all_locations[x] for x in legal_idxs]
    legal_loc = random.choice(legal_locs)
    return list(legal_loc)


def state_init(mode:str,Grid_size:int,items:dict,all_locations):
    if mode == 'static':
        locations = np.array([[0,0,items['Agent']],[1,1,items['Hole']],[3,3,items['Wall']],[3,1,items['Goal']]]).T
        state = np.zeros((Grid_size,Grid_size,len(items)) ,dtype='int32')
        state[locations[0],locations[1],locations[2]] = 1
    else:
        Agent_loc = [0 ,0] + [items['Agent']]
        Hole_loc = find_legal_location([Agent_loc[:2]],Grid_size,all_locations) + [items['Hole']]
        Wall_loc = find_legal_location([Agent_loc[:2],Hole_loc[:2]],Grid_size,all_locations) + [items['Wall']]
        Goal_loc = find_legal_location([Agent_loc[:2],Hole_loc[:2],Wall_loc[:2]],Grid_size,all_locations) + [items['Goal']]
        locations = np.array([Agent_loc,Hole_loc,Wall_loc,Goal_loc]).T
        state = np.zeros((Grid_size,Grid_size,len(items)) ,dtype='int32')
        state[locations[0],locations[1],locations[2]] = 1
    return state ,locations.T

def create_canvas(frame_size:tuple,Grid_size:int):
    canvas = np.ones(frame_size).astype(np.uint8) * 255
    cell = int(canvas.shape[0] // Grid_size)
    for i in range(Grid_size):
        canvas[:,i * cell] = 0
        if i == 7:
            canvas[:,-1] = 0

    for i in range(Grid_size):
        canvas[i * cell,:] = 0
        if i == 7:
            canvas[-1,:] = 0
    return canvas, cell

def load_the_icons(icon_paths, cell_size):
    icons = []
    for icon_path in icon_paths:
        icon = cv2.imread(icon_path)
        icon_w = int(cell_size * 0.9)
        icon_h = int(cell_size * 0.9)
        icon = cv2.resize(icon, (icon_h, icon_w))
        #icon = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
        icons.append(icon)
    return icons

class Gridworld(Env):
    def __init__(self, Grid_size, icon_paths = {'Agent':'robot.png','Hole':'hole.png','Wall':'wall.jpg','Goal':'goal.png'} ,mode = 'static' ,frame_size = (800,800,3)):
        icon_paths = [icon_paths['Agent'],icon_paths['Hole'],icon_paths["Wall"],icon_paths["Goal"]]
        self.items = {'Agent':0,'Hole':1,'Wall':2,'Goal':3}
        self.actions  = ['Left','Right','Up','Down']
        self.mode = mode
        self.actions_space = Discrete(4) # ← ↑ → ↓
        self.Grid_size = Grid_size
        self.frame_size = frame_size
        self.icon_paths = icon_paths
        self.observation_space = Box(low = np.zeros((Grid_size,Grid_size,len(self.items))),\
                                     high = np.ones((Grid_size,Grid_size,len(self.items))),dtype = np.int32) 
        self.all_locations = list(product(np.arange(Grid_size), repeat = 2))
        self.state ,self.locations = state_init(mode,Grid_size ,self.items ,self.all_locations)
        self.canvas, self.cell_size = create_canvas(self.frame_size ,Grid_size)
        self.icons = load_the_icons(self.icon_paths, self.cell_size)
        for i ,loc in enumerate(self.locations):
            y_start = int(loc[0] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[0]))
            y_end = y_start + self.icons[i].shape[0]
            x_start = int(loc[1] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[1]))
            x_end =   x_start + self.icons[i].shape[0] 
            self.canvas[y_start : y_end,x_start:x_end,:] = self.icons[i]
            
            
    def step(self ,action):
        action = self.actions[action]
        x1_old ,y1_old ,z1_old = self.locations[self.items['Agent']]
        x2 ,y2 ,z2 = self.locations[self.items['Hole']]
        x3 ,y3 ,z3 = self.locations[self.items['Wall']]
        x4 ,y4 ,z4 = self.locations[self.items['Goal']]
        you_can_draw = "NO"        
        if action == 'Left':
                self.locations[self.items['Agent']] = [x1_old  ,y1_old -1 ,z1_old]

        elif action == 'Right':
                self.locations[self.items['Agent']] = [x1_old ,y1_old +1,z1_old]

        elif action == 'Up':
                self.locations[self.items['Agent']] = [x1_old -1 ,y1_old ,z1_old]

        elif action == 'Down':
                self.locations[self.items['Agent']] = [x1_old +1 ,y1_old ,z1_old]
        
        x1 ,y1 ,z1 = self.locations[self.items['Agent']]
        x2 ,y2 ,z2 = self.locations[self.items['Hole']]
        x3 ,y3 ,z3 = self.locations[self.items['Wall']]
        x4 ,y4 ,z4 = self.locations[self.items['Goal']]

        if [x1 ,y1] == [x2 ,y2]:
                    reward = -10
                    done = True
                    you_can_draw = 'OK'
                    self.locations[self.items['Agent']] = [x1 ,y1 ,z1]
                    self.state[x1_old ,y1_old ,z1_old] = 0
                    self.state[x1 ,y1 ,z1] = 1

        elif [x1 ,y1] == [x3 ,y3]:
                    reward = -1
                    done = False
                    self.locations[self.items['Agent']] =  [x1_old ,y1_old ,z1_old]
                    self.state[x1_old ,y1_old ,z1_old] = 1
                    self.state[x1 ,y1 ,z1] = 0

                    you_can_draw = 'OK'
   
        elif [x1 ,y1] == [x4 ,y4]:
                    reward = +10
                    done = True
                    you_can_draw = 'OK'
                    self.locations[self.items['Agent']] = [x1 ,y1 ,z1]
                    self.state[x1_old ,y1_old ,z1_old] = 0
                    self.state[x1 ,y1 ,z1] = 1

        elif [x1 ,y1][0] >= self.Grid_size  or [x1 ,y1][1] >= self.Grid_size :
                    reward = -1

                    done = False
                    you_can_draw = 'OK'
                    self.locations[self.items['Agent']] = [x1_old ,y1_old ,z1_old]
                    self.state[x1_old ,y1_old ,z1_old] = 1 
                    
        elif [x1 ,y1][0] < 0 or [x1 ,y1][1] < 0 :
                    reward = -1
                    done = False
                    you_can_draw = 'OK'
                    self.locations[self.items['Agent']] = [x1_old ,y1_old ,z1_old]
                    self.state[x1_old ,y1_old ,z1_old] = 1 
                    
        else:
                    reward = -1
                    self.state[x1 ,y1 ,z1 ] = 1
                    self.locations[self.items['Agent']] = [x1 ,y1 ,z1]
                    self.state[x1_old ,y1_old ,z1_old] = 0
                    you_can_draw = 'OK'
                    done = False
                    
        if you_can_draw == 'OK':
            self.canvas, self.cell_size = create_canvas(self.frame_size ,self.Grid_size) 
            if reward == +10:               
                for i ,loc in enumerate(self.locations):
                    if i != self.items['Goal']:
                        y_start = int(loc[0] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[0]))
                        y_end = y_start + self.icons[i].shape[0]
                        x_start = int(loc[1] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[1]))
                        x_end =   x_start + self.icons[i].shape[0] 
                        self.canvas[y_start : y_end,x_start:x_end,:] = self.icons[i]  
            elif reward == -10:
                for i ,loc in enumerate(self.locations):
                     if i != self.items['Hole']:
                        y_start = int(loc[0] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[0]))
                        y_end = y_start + self.icons[i].shape[0]
                        x_start = int(loc[1] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[1]))
                        x_end =   x_start + self.icons[i].shape[0] 
                        self.canvas[y_start : y_end,x_start:x_end,:] = self.icons[i]                 
            else:    
                for i ,loc in enumerate(self.locations):
                    y_start = int(loc[0] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[0]))
                    y_end = y_start + self.icons[i].shape[0]
                    x_start = int(loc[1] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[1]))
                    x_end =   x_start + self.icons[i].shape[0] 
                    self.canvas[y_start : y_end,x_start:x_end,:] = self.icons[i]
            
                    

        
        info = {'items':self.items,'Grid_size':self.Grid_size,'actions':self.actions}    
        return self.state, reward, done, info
    
    
    def render(self, mode="human"):
        

        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        if mode == "human":
            if "ipykernel" in sys.modules:  # Notebook / VS Code interactive
                clear_output(wait=True)
                plt.imshow(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
            else:  # Standalone Python script
                cv2.imshow("Gridworld", self.canvas)
                cv2.waitKey(1)

        elif mode == "rgb_array":
            return self.canvas

        
    
    def reset(self):
        self.state ,self.locations = state_init(self.mode,self.Grid_size,self.items,self.all_locations)
        self.canvas, self.cell_size = create_canvas(self.frame_size,self.Grid_size)
        for i ,loc in enumerate(self.locations):
            y_start = int(loc[0] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[0]))
            y_end = y_start + self.icons[i].shape[0]
            x_start = int(loc[1] * self.cell_size + (1/2) * (self.cell_size - self.icons[i].shape[1]))
            x_end =   x_start + self.icons[i].shape[0] 
            self.canvas[y_start : y_end,x_start:x_end,:] = self.icons[i]
        return self.state