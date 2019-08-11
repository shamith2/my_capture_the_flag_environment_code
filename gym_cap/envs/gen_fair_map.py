import numpy as np
import random
from gym_cap.envs.const import *

class gen_board():
        '''
          generates fair boards
        '''

	def gen_fair_map(path, num_board):
	    
	    for j in range(num_board):
		# constanst
		dim=20
		in_seed=None
		map_obj=[NUM_BLUE, NUM_UAV, NUM_GRAY]
		
		# init the seed and set new_map to zeros
		if not in_seed == None:
		    np.random.seed(in_seed)
		
		# quads init
		new_map = np.zeros([dim, dim], dtype=int)
		new_map[:,:] = TEAM2_BACKGROUND
		selector = 5 #np.random.randint(1, 6)
		if selector == 1 or selector == 5:
		    new_map[0:dim//2,0:dim//2] = TEAM1_BACKGROUND
		    new_map[dim//2:dim,dim//2:dim] = TEAM1_BACKGROUND
		elif selector == 2:
		    new_map[0:dim//2,dim//2:dim] = TEAM1_BACKGROUND
		    new_map[dim//2:dim,0:dim//2] = TEAM1_BACKGROUND
		elif selector == 3:
		    new_map[0:dim//2,:] = TEAM1_BACKGROUND
		elif selector == 4:
		    new_map[:,0:dim//2] = TEAM1_BACKGROUND

		# obtain coord
		team1_pool = np.argwhere(new_map==TEAM1_BACKGROUND).tolist() # (N,2)
		team2_pool = np.argwhere(new_map==TEAM2_BACKGROUND).tolist()   
		random.shuffle(team1_pool)
		random.shuffle(team2_pool)
		
		# obstacles init
		num_obst = int(np.sqrt(dim))
		for i in range(num_obst):
		    if selector == 5:
		        team1_pool[i][0] = dim//2 + (dim//2-team1_pool[i][0])
		        team1_pool[i][1] = dim//2 + (dim//2-team1_pool[i][1])
		        team2_pool[i][0] = dim//2 + (dim//2-team1_pool[i][0])
		        team2_pool[i][1] = team1_pool[i][1]
		        team2_pool[i][0] = team1_pool[i][0]
		        team2_pool[i][1] = dim//2 + (dim//2-team1_pool[i][1])
		    elif selector == 2 or selector == 3:
		        team2_pool[i][0] = dim//2 + (dim//2-team1_pool[i][0])
		        team2_pool[i][1] = team1_pool[i][1]
		    elif selector == 1 or selector == 4:
		        team2_pool[i][0] = team1_pool[i][0]
		        team2_pool[i][1] = dim//2 + (dim//2-team1_pool[i][1])

		    lx, ly = team1_pool[i]
		    llx, lly = team2_pool[i]
		    sx, sy = np.random.randint(1,4,[2])
		    if (lx < (dim - 2) and ly < (dim - 2) and llx < (dim - 2) and lly < (dim - 2)):
		        new_map[lx-sx:lx+sx, ly-sy:ly+sy] = OBSTACLE
		        new_map[llx-sx:llx+sx, lly-sy:lly+sy] = OBSTACLE

		team1_pool = np.argwhere(np.logical_and(new_map==TEAM1_BACKGROUND, new_map != OBSTACLE)).tolist()
		team2_pool = np.argwhere(np.logical_and(new_map==TEAM2_BACKGROUND, new_map != OBSTACLE)).tolist()
		random.shuffle(team1_pool)
		random.shuffle(team2_pool)
		
		# del improper coord
		p,d = 3,2  # p is for flag and d is for agent
		i = 0
		while i < len(team1_pool):
		    x, y = team1_pool[i]
		    if ((x<=d or y<=d) or ((x>=(dim//2-d) and x<=(dim//2+d)) or (y>=(dim//2-d) and y<=(dim//2+d))) or ((x>=(dim-d) or y>=(dim-d)))):
		        team1_pool.remove([x,y])
		        i += 1
		        i -= 1
		    else:
		        i += 1
		while i < len(team2_pool):
		    x, y = team2_pool[i]
		    if ((x<=d or y<=d) or ((x>=(dim//2-d) and x<=(dim//2+d)) or (y>=(dim//2-d) and y<=(dim//2+d))) or ((x>=(dim-d) or y>=(dim-d)))):
		        team2_pool.remove([x,y])
		        i += 1
		        i -= 1
		    else:
		        i += 1
		for x, y in team1_pool:
		    if (((x>p and (x<(dim//2-p))) and (y>p and y<(dim//2-p))) or ((x>(dim//2+p) and x<(dim-p)) and (y>(dim//2+p) and y<(dim-p)))):
		        team1_pool[0:1][0][0],team1_pool[0:1][0][1] = x,y

		# define location of flag
		team2_pool[0:1][0][0] = team1_pool[0:1][0][0]
		team2_pool[0:1][0][1] = (dim//2-1) + (dim//2-team1_pool[0:1][0][1])
		if selector == 2 or selector == 3:
		    team2_pool[0:1][0][0] = (dim//2-1) + (dim//2-team1_pool[0:1][0][0])
		    team2_pool[0:1][0][1] = team1_pool[0:1][0][1]
		
		new_map[team1_pool[0:1][0][0], team1_pool[0:1][0][1]] = TEAM1_FLAG
		new_map[team2_pool[0:1][0][0], team2_pool[0:1][0][1]] = TEAM2_FLAG
		    
		team1_pool = team1_pool[1:]
		team2_pool = team2_pool[1:]
		
		# define location of agents
		for i in range(map_obj[0]):
		    team2_pool[0:map_obj[0]][i][0] = team1_pool[0:map_obj[0]][i][0]
		    team2_pool[0:map_obj[0]][i][1] = (dim//2-1) + (dim//2-team1_pool[0:map_obj[0]][i][1])
		    if selector == 2 or selector == 3:
		        team2_pool[0:map_obj[0]][i][0] = (dim//2-1) + (dim//2-team1_pool[0:map_obj[0]][i][0])
		        team2_pool[0:map_obj[0]][i][1] = team1_pool[0:map_obj[0]][i][1]
	    
		    new_map[team1_pool[0:map_obj[0]][i][0], team1_pool[0:map_obj[0]][i][1]] = TEAM1_UGV
		    new_map[team2_pool[0:map_obj[0]][i][0], team2_pool[0:map_obj[0]][i][1]] = TEAM2_UGV
		
		np.savetxt('{}/board_{}.txt'.format(path, (j+1)), new_map, delimiter=' ', fmt='%d')

class board_connect():
	'''
	checks connectivity within board
	'''

	def __init__(self, board): 
	self.nodes = np.argwhere(board != OBSTACLE)
	self.neighbours = [[] for i in range(len(self.nodes))]

	def add_neighbours(self, node):
	x,y = node
	for i, j in self.nodes:
	    if abs(x-i) + abs(y-j) == 1:
		self.neighbours[node].append((i,j))

	def node_connect(self, node, visited):
	visited[node] = True
	self.add_neighbours(node)
	n = self.neighbours[node][0] 
	while n != self.neighbours[node][-1]: 
	    if (not visited[n]): 
		self.node_connect(n, visited) 
	    n += 1

	def countNotReach(self, node): 
	visited = [False] * len(self.nodes)
	count = 0
	self.node_connect(node, visited)    
	for n in range(len(self.nodes)): 
	    if (visited[n] == False):  
		count += 1
	if count >= 1:
	    return False
	else:
	    return True
