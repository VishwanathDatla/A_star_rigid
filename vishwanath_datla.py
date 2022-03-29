
import time
import matplotlib.pyplot as plt
import cv2
import pygame
import numpy as np
import copy
import math
import heapq


#Getting the start time to measure the time taken for solving

start_time = time.time()

#function to round the point 5

def roundto(num):
    return (round(num * 2) / 2)


    
def dist(a,b):  
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist  

    
step_count=1 
size_x = 401 
size_y = 251 
radius = 1 
clearance = 1 


def ActionMove(curr_node,degree,step_size=1.0):
    x = curr_node[0]
    y = curr_node[1]
    x_new = (step_size)*np.cos(np.deg2rad(degree)) + x
    y_new = (step_size)*np.sin(np.deg2rad(degree)) + y
    new_node = (round(x_new,2),round(y_new,2))
    if new_node[0]>=0.00 and new_node[0]<=400.00 and new_node[1]>=0.00 and new_node[1]<=250.00:
        return(new_node,True)
    else:
        return(curr_node,False)
        
all_possible_points = []
for i in range(0,801): 
    for j in range(0,501): 
        all_possible_points.append((roundto(i/2),roundto(j/2))) #appending
        
#all possible points in the obstacle space
list_of_obstacle_points=[]

for pt in all_possible_points:
    
    x = pt[0]
    y = pt[1]
    
#circle shaped obstacle
    #for path traversal
    if((x-300)**2 + (y-185)**2 <= (40+radius+clearance)**2):
        list_of_obstacle_points.append((x,y))
   
    (x1, y1) = (185,36)
    (x2, y2) = (180,80)
    (x3, y3) = (105,100)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist3 = 1
    if(first >= radius+clearance and second >= radius+clearance and third >= radius+clearance):
                list_of_obstacle_points.append((x,y))
    (x1, y1) = (185,36)
    (x2, y2) = (210,115)
    (x3, y3) = (180,80)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist4 = 1
    if(first >= radius+clearance and second >= radius+clearance and third >= 0):
            list_of_obstacle_points.append((x,y))
    
    (x1,y1) = (59.6,200)
    (x2,y2) = (79.8,235)
    (x3,y3) = (120.2,235)
    (x4,y4) = (140.4,200)
    (x5,y5) = (120.2,165)
    (x6,y6) = (79.8,165)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))
    third = ((x - y6) * (x1 - x6)) - ((y1 - y6) * (y - x6))
    if(first<=radius+clearance and second<=0 and third<=radius+clearance):
        list_of_obstacle_points.append((x,y))
    first = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))  #Horizontal line 1 
    second = ((x - y6) * (x5 - x6)) - ((y5 - y6) * (y - x6))
    third = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5)) # horizontal line 2 
    fourth = ((x - y3) * (x2 - x3)) - ((y2 - y3) * (y - x3))
    if(first>=0 and second>=0 and third>=radius+clearance and fourth>=radius+clearance):
        list_of_obstacle_points.append((x,y))
    second = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5))
    first = ((x - y4) * (x3 - x4)) - ((y3 - y4) * (y - x4))
    third = ((x - y4) * (x5 - x4)) - ((y5 - y4) * (y - x4))
    if(first>=0 and second<=radius+clearance and third<=radius+clearance):
        list_of_obstacle_points.append((x,y))


        
all_possible_int_points = []


map_points = []

for i in range(0,401): 
    for j in range(0,251): 
        all_possible_int_points.append((i,j)) 

for pt in all_possible_int_points:
    x = pt[0]
    y = pt[1]

    if((x-300)**2 + (y-185)**2 <= (40)**2):
        map_points.append((x,y))

    (x1, y1) = (185,36)
    (x2, y2) = (180,80)
    (x3, y3) = (105,100)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist3 = 1
    if(first >= 5 and second >= 5 and third >= 5):
          map_points.append((x,y))
           
    (x1, y1) = (185,36)
    (x2, y2) = (210,115)
    (x3, y3) = (180,80)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist4 = 1
    if(first >= 5 and second >= 5 and third >= 0):
            map_points.append((x,y))

    (x1,y1) = (59.6,200)
    (x2,y2) = (79.8,235)
    (x3,y3) = (120.2,235)
    (x4,y4) = (140.4,200)
    (x5,y5) = (120.2,165)
    (x6,y6) = (79.8,165)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))
    third = ((x - y6) * (x1 - x6)) - ((y1 - y6) * (y - x6))
    if(first<=5 and second<=0 and third<=5):
        map_points.append((x,y))
    first = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))  
    second = ((x - y6) * (x5 - x6)) - ((y5 - y6) * (y - x6))
    third = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5)) 
    fourth = ((x - y3) * (x2 - x3)) - ((y2 - y3) * (y - x3))
    if(first>=0 and second>=0 and third>=5 and fourth>=5):
        map_points.append((x,y))
    second = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5))
    first = ((x - y4) * (x3 - x4)) - ((y3 - y4) * (y - x4))
    third = ((x - y4) * (x5 - x4)) - ((y5 - y4) * (y - x4))
    if(first>=0 and second<=5 and third<=5):
        map_points.append((x,y))
    

                


list_of_obstacle_points.sort()

def checkObstaclespace(point):
    
    test = []
    x = point[0]
    y = point[1]

    
    if((x-300)**2 + (y-185)**2 <= (40+radius+clearance)**2):
        return False

    
   
    (x1, y1) = (185,36)
    (x2, y2) = (180,80)
    (x3, y3) = (105,100)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist3 = 1
    if(first >= radius+clearance and second >= radius+clearance and third >= radius+clearance):
                return False
    (x1, y1) = (185,36)
    (x2, y2) = (210,115)
    (x3, y3) = (180,80)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x3 - x2)) - ((y3 - y2) * (y - x2))
    third = ((x - y3) * (x1 - x3)) - ((y1 - y3) * (y - x3))
    dist4 = 1
    if(first >= radius+clearance and second >= radius+clearance and third >= 0):
            return False
    
    (x1,y1) = (59.6,200)
    (x2,y2) = (79.8,235)
    (x3,y3) = (120.2,235)
    (x4,y4) = (140.4,200)
    (x5,y5) = (120.2,165)
    (x6,y6) = (79.8,165)
    first = ((x - y1) * (x2 - x1)) - ((y2 - y1) * (y - x1))
    second = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))
    third = ((x - y6) * (x1 - x6)) - ((y1 - y6) * (y - x6))
    if(first<=radius+clearance and second<=0 and third<=radius+clearance):
        return False
    first = ((x - y2) * (x6 - x2)) - ((y6 - y2) * (y - x2))  
    second = ((x - y6) * (x5 - x6)) - ((y5 - y6) * (y - x6))
    third = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5)) 
    fourth = ((x - y3) * (x2 - x3)) - ((y2 - y3) * (y - x3))
    if(first>=0 and second>=0 and third>=radius+clearance and fourth>=radius+clearance):
        return False
    second = ((x - y5) * (x3 - x5)) - ((y3 - y5) * (y - x5))
    first = ((x - y4) * (x3 - x4)) - ((y3 - y4) * (y - x4))
    third = ((x - y4) * (x5 - x4)) - ((y5 - y4) * (y - x4))
    if(first>=0 and second<=radius+clearance and third<=radius+clearance):
        return False
   
    else:
        
        return True    
    
x_start= int(input("Enter the x coordinate of the start:  "))
y_start= int(input("Enter the y coordinate of the start:  "))
orientation = int(input("Enter the Orientation at start (enter in multiples of 30 degreees and less that 360 degrees), :  "))
x_goal= int(input("Enter the x coordinate of the goal:  "))
y_goal= int(input("Enter the y coordinate of the goal:  "))
radius= int(input("Enter the radius of the robot:  "))
clearance= int(input("Enter the clearance of the robot: "))
start = (x_start,y_start)
goal = (x_goal,y_goal)


list_of_points_for_graph = []

def generateGraph(point,size_x,size_y): 
    
    global step_count
    global orientation
    global list_of_points_for_graph
    
    i = point[0] 
    j = point[1] 
    
    if i <=size_x and j<=size_y and i>=0 and j>=0:
        
        cost_values = {}
        
        pos0 = ActionMove(point,orientation+0)[0]
        if pos0[0]>=0 and pos0[1]>=0 and pos0[0]<=size_x and pos0[1]<=size_y:
            cost_values[pos0] = (step_count,orientation)
            
        pos30 = ActionMove(point,orientation+30)[0]
        if pos30[0]>=0 and pos30[1]>=0 and pos30[0]<=size_x and pos30[1]<=size_y: 
            cost_values[pos30] = (1.4,orientation+30)
            
        pos60 = ActionMove(point,orientation+60)[0]
        if pos60[0]>=0 and pos60[1]>=0 and pos60[0]<=size_x and pos60[1]<=size_y:
            cost_values[pos60] = (1.7,orientation+60)
            
        pos_minus60 = ActionMove(point,orientation-60)[0]
        if pos_minus60[0]>=0 and pos_minus60[1]>=0 and pos_minus60[0]<=size_x and pos_minus60[1]<=size_y:
            cost_values[pos_minus60] = (1.7,orientation-60)
            
        pos_minus30 = ActionMove(point,orientation-30)[0]
        if pos_minus30[0]>=0 and pos_minus30[1]>=0 and pos_minus30[0]<=size_x and pos_minus30[1]<=size_y:
            cost_values[pos_minus30] = (1.4,orientation-30)
            
        cost_values_copy = cost_values.copy()
        
        for k,v in cost_values_copy.items():
            if k==point:
                del cost_values[k]
        return(cost_values)
    
    else:
        
        pass

        
def BackTrack(backtrack_dict,goal,start):
    
    
    back_track_list = []
   
    back_track_list.append(start)
    
    while goal!=[]:
        
        for k,v in backtracking.items():
            
            for k2,v2 in v.items():
                
                if k==start:
                    
                    if v2 not in back_track_list:
                        back_track_list.append(start)
                   
                    start=v2
                    
                   
                    if v2==goal:
                        goal=[]
                        break      

    return(back_track_list)

all_distance = {}

open_list = {}

backtracking = {}
rows = 501
columns = 801
layers =12 
V=np.zeros((rows,columns,layers))
visited = []

check=0


def a_star_Algorithm(start,goal):
    
    global orientation
    global list_of_obstacle_points
    global backtracking
    global check
    global visited
    global all_distance
    
    if goal in list_of_obstacle_points or start in list_of_obstacle_points:
        
       
        all_distance=0
        backtracking=0
        rounded_neighbour=0
        
    else:
        
        all_distance[start]=0
    
        priority_queue = [(0,start,orientation)]
        
        
        while len(priority_queue)>0 and check!=[]:
            
            curr_dist,curr_vert,orientation = heapq.heappop(priority_queue)

           
            
            if checkObstaclespace(curr_vert)==True:
                graph = generateGraph(curr_vert,401,251)
                

                for vertex,edge in graph.items():
                    all_distance[vertex]=math.inf
                graph_list = []
                
                for key,cost_value in graph.items():
                    graph_list.append((key,cost_value))
                    
                if curr_dist>all_distance[curr_vert]:
                    continue
                
                for neighbour,cost in graph_list:
                    this_cost = graph[neighbour][0]
                    curr_dist = 0
                    distance = curr_dist + this_cost + dist(neighbour,goal)               
                    
                    if distance < all_distance[neighbour]:
                        
                        rounded_neighbour = (roundto(neighbour[0]),roundto(neighbour[1]))

                        try:
                            
                            orientation = ((orientation) % 360)
                                   
                            orientation_to_layer={0:0,30:1,60:2,90:3,120:4,150:5,180:6,210:7,240:8,270:9,300:10,330:11, 360:0}
                            #print('rounded neighbour > ',rounded_neighbour,'with orientation', orientation)
                            
                            if rounded_neighbour not in visited:
                                if V[int(2*rounded_neighbour[0])][int(2*rounded_neighbour[1])][orientation_to_layer[orientation]]==0:
                                    V[int(2*rounded_neighbour[0])][int(2*rounded_neighbour[1])][orientation_to_layer[orientation]]=1
                                
                                    visited.append(rounded_neighbour)
                                    backtracking[rounded_neighbour]={}
                                   
                                    backtracking[rounded_neighbour][distance]=curr_vert
                                    all_distance[rounded_neighbour]=distance
                                
                                    orientation = graph[neighbour][1]
                                    heapq.heappush(priority_queue, (distance, rounded_neighbour,orientation))
                                    
                                    if ((rounded_neighbour[0]-goal[0])**2 + (rounded_neighbour[1]-goal[1])**2 <= (1.5)**2):
                                        
                                        
                                        check=[]
                                        
                                        break
                                    else:

                                        pass
                        except:
                            pass
    return(all_distance,backtracking,rounded_neighbour)     


all_distances,backtracking,new_goal_rounded= a_star_Algorithm(start,goal)

backtracked_final = BackTrack(backtracking,start,new_goal_rounded)
print(backtracked_final)

def generateChilds(backtrack):
    
    parents2children = {}
    for parent in backtrack:
        child = generateGraph(parent,401,251)
        
        all_children_here = set()
        for key,value in child.items():
            all_children_here.add(key)
        parents2children[parent] = all_children_here
        
    return(parents2children)
branched_parents = generateChilds(backtracked_final)


for i in range(1,len(backtracked_final)-1):
    
    x = backtracked_final[i][0]
    y = backtracked_final[i][1]
    x2 = backtracked_final[i+1][0]
    y2 = backtracked_final[i+1][1]
    plt.plot([x,x2],[y,y2])


plt.savefig('Path traversed.png', bbox_inches='tight')

print("Total Time Taken : ",time.time() - start_time, "seconds")



new_canvas = np.zeros((251,401,3),np.uint8) 

for c in map_points: 
    x = c[1]
    y = c[0]
    new_canvas[(x,y)]=[0,255,255] #assigning a yellow coloured pixel
    
new_canvas = np.flipud(new_canvas)

new_canvas_copy_backtrack = new_canvas.copy()
new_canvas_copy_visited = new_canvas.copy()


for path in backtracked_final:
    
   
    x = int(path[0])
    y = int(path[1])
    new_canvas_copy_backtrack[(250-y,x)]=[255,0,0] #setting every backtracked pixel to white

new_backtracked = cv2.resize(new_canvas_copy_backtrack,(800,500))


cv2.imshow('backtracked',new_backtracked)
cv2.imwrite('backtracked_img.jpg',new_backtracked)
cv2.waitKey(0)
cv2.destroyAllWindows()



for path in visited:
    
    
    x = int(path[0])
    y = int(path[1])
    new_canvas_copy_visited[(250-y,x)]=[255,0,0] #setting every backtracked pixel to white
    

new_visited = cv2.resize(new_canvas_copy_visited,(800,500))
cv2.imshow('visited',new_visited)
cv2.imwrite('visited_img.jpg',new_visited)
cv2.waitKey(0)
cv2.destroyAllWindows()


pygame.init()

display_width = 400
display_height = 250

gameDisplay = pygame.display.set_mode((display_width,display_height),pygame.FULLSCREEN)
pygame.display.set_caption('Covered Nodes- Animation')

black = (0,0,0)
white = (0,255,255)
surf = pygame.surfarray.make_surface(new_canvas_copy_visited)

clock = pygame.time.Clock()

done = False
while not done:
    
    for event in pygame.event.get(): 
        
        if event.type == pygame.QUIT:  
            done = True   
 
    gameDisplay.fill(black)
    for path in visited:
        if path not in new_canvas_copy_visited:
            pygame.time.wait(7)
            x = path[0]
            y = abs(200-path[1])
            pygame.draw.rect(gameDisplay, white, [x,y,1,1])
            pygame.display.flip()
            
    for path in backtracked_final:
        
        pygame.time.wait(5)
        x = path[0]
        y = abs(200-path[1])
        pygame.draw.rect(gameDisplay, (0,0,255), [x,y,1,1])
        pygame.display.flip()

    done = True
pygame.quit()

