import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from nodes import TileGraph
from nodes import NodeGraph
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
import numpy as np
import math

class GameController(object):
    def __init__(self, bot):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

        # ( Begin edit )
        # Add in the necessary instance variables so that pacbot has all relevant information

        self.bot = bot
        self.running = True
        self.power = False
        self.count = 0
        self.tile_graph = {}
        self.pel_dict = {}
        self.intersections = {}
        self.inter_graph = {}

        # ( End edit )

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)

        # ( Begin edit )
        # Create a graph consisting of all tiles on the map, and a graph consisting of all nodes on the map (minus home nodes/tiles)

        self.tile_graph = TileGraph(self.nodes)
        self.inter_graph = NodeGraph(self.tile_graph, self.nodes)

        # ( End edit )

        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart), self.bot)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")

        # ( Begin edit )
        # Create a dictionary of pellet locations and a dictionary of intersection (node) locations

        for pel in self.pellets.pelletList:
            self.pel_dict[self.getTile(pel.position.x, pel.position.y)] = True
        for pel in self.pellets.powerpellets:
            self.pel_dict[self.getTile(pel.position.x, pel.position.y)] = True

        for node in self.nodes.nodesLUT:
            self.intersections[self.getTile(node[0], node[1])] = True

        # ( End edit )

        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def startGame_old(self):      
        self.mazedata.loadMaze(self.level)#######
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))

        # ( Begin edit )
        # Create a graph consisting of all tiles on the map, and a graph consisting of all nodes on the map (minus home nodes/tiles)

        self.tile_graph = TileGraph(self.nodes)
        self.inter_graph = NodeGraph(self.tile_graph, self.nodes)

        # ( End edit )

        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26), self.bot)
        self.pellets = PelletGroup("maze1.txt")

        # ( Begin edit )
        # Create a dictionary of pellet locations and a dictionary of intersection (node) locations

        for pel in self.pellets.pelletList:
            self.pel_dict[self.getTile(pel.position.x, pel.position.y)] = True
        for pel in self.pellets.powerpellets:
            self.pel_dict[self.getTile(pel.position.x, pel.position.y)] = True

        for node in self.nodes.nodesLUT:
            self.intersections[self.getTile(node[0], node[1])] = True

        # ( End edit )

        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

        

    def update(self):
        dt = self.clock.tick(30) / 1000.0

        # ( Begin edit )
        # Set pacman's input variables

        if self.power:
            self.count -= dt
        if self.count <= 0:
            self.power = False

        self.getInputs()

        # ( End edit )

        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)

            # ( Begin edit )
            # Bookkeep for the pellet dictionary

            self.pel_dict[self.getTile(pellet.position.x, pellet.position.y)] = False

            # ( End edit )

            if pellet.name == POWERPELLET:

                # ( Begin edit )
                # Initialize the power pellet instance variables when a power pellet is eaten

                self.power = True
                self.count = 7

                # ( End edit )

                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.running = False
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                # print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()

    # ( Begin edit )
    # Function which converts pixel values into tile values
    def getTile(self, x, y):
        return (int(x / TILEWIDTH + 0.5), int(y / TILEHEIGHT + 0.5))

    # Function which sets Pacman's inputs for his Pacbot neural network to use
    def getInputs(self):
        inputs = [[], [], [], []]
        # The first input variable denotes how much progress Pacman has made in the level
        # by counting how many pellets are left uneaten for the given stage
        level_progress = (244 - len(self.pellets.pelletList)) / 244
        # Second input variable denotes how much time left Pacman has in the power-up phase
        # after he eats a powerpill
        powerpill = 0
        if self.power:
            powerpill = (self.count) / 7
        # The next three stages indicate the distance of the nearest pill for any given direction,
        # the distance of the nearest unscared ghost for any given direction, and the distance of
        # the nearest scared ghost for any given direction
        pills, ghosts_pos, scared_ghosts = [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]
        pac_pos = self.getTile(self.pacman.position.x, self.pacman.position.y)
        tiles = [1, 1, 1, 1]
        # Find the tiles that are above, below, to the left, and to the right of Pacman
        if self.tile_graph.vertices[pac_pos][0] != None:
            tiles[0] = self.tile_graph.vertices[pac_pos][0]
        if self.tile_graph.vertices[pac_pos][1] != None:
            tiles[1] = self.tile_graph.vertices[pac_pos][1]
        if self.tile_graph.vertices[pac_pos][2] != None:
            tiles[2] = self.tile_graph.vertices[pac_pos][2]
        if self.tile_graph.vertices[pac_pos][3] != None:
            tiles[3] = self.tile_graph.vertices[pac_pos][3]
        # Find all locations of ghosts and record whether they are currently scared or not
        ghost_tracking = {}
        for ghost in self.ghosts:
            if ghost.mode.current == FREIGHT:
                ghost_tracking[self.getTile(ghost.position.x, ghost.position.y)] = True
            else:
                ghost_tracking[self.getTile(ghost.position.x, ghost.position.y)] = False
        # Using breadth-first search, find the distances from each ghost to each tile in the map
        index = 0
        ghost_dists = {}
        # Initialize all distance values to an empty list
        for vertex in self.tile_graph.vertices:
            ghost_dists[vertex] = []
        # For each ghost in the list
        for ghost in ghost_tracking:
            # If the ghost is in freight mode or is in the home nodes, do not perform a search for that ghost
            if ghost_tracking[ghost] or self.tile_graph.vertices.get(ghost) == None:
                continue
            else:
                # Otherwise, BFS through each tile in the tile graph to find the distance from each ghost to each tile
                self.tile_graph.resetVisited()
                queue = [ghost]
                self.tile_graph.visited[ghost] = True
                ghost_dists[ghost].append(0)
                while len(queue) != 0:
                    v = queue.pop(0)
                    for j in range(4):
                        vj = self.tile_graph.vertices[v][j]
                        if vj != None and not self.tile_graph.visited[vj]:
                            queue.append(vj)
                            ghost_dists[vj].append(ghost_dists[v][index] + 1)
                            self.tile_graph.visited[vj] = True
                index += 1
        # Perform BFS from each tile surrounding Pacman (up, down, left, right) 
        for i in range(4):
            # If there is no tile in the current direction, do not perform a search from that tile
            if tiles[i] == 1:
                pills[i], ghosts_pos[i], scared_ghosts[i] = 0.0, 0.0, 0.0
            else:
                # Initialize all variables to perform BFS
                self.tile_graph.resetVisited()
                queue = [tiles[i]]
                self.tile_graph.visited[tiles[i]] = True
                dists = {}
                closest_pill, closest_ghost = np.inf, np.inf
                maximum_path = 0
                nearest_inter = ()
                inter_dist = np.inf
                for vertex in self.tile_graph.vertices:
                    dists[vertex] = np.inf
                dists[tiles[i]] = 0
                while len(queue) != 0:
                    v = queue.pop(0)
                    for j in range(4):
                        vj = self.tile_graph.vertices[v][j]
                        if vj != None and not self.tile_graph.visited[vj]:
                            queue.append(vj)
                            dists[vj] = dists[v] + 1
                            self.tile_graph.visited[vj] = True
                            # If Pacman is is power-up mode
                            if self.power:
                                # Check to see if the current tile has a scared ghost on it
                                if ghost_tracking.get(vj) != None and ghost_tracking[vj] and dists[vj] < closest_ghost:
                                    # If it does, mark that tile as the closes ghost tile
                                    closest_ghost = dists[vj]
                            # If the current tile is an intersection that is closer than the current closest intersection
                            if self.nodes.getNodeFromTiles(vj[0], vj[1]) != None and dists[vj] < inter_dist:
                                # Mark this intersection as the closest
                                inter_dist = dists[vj]
                                nearest_inter = vj
                            # If the current tile contains a pellet and that pellet is closer than the current closest pellet
                            if self.pel_dict.get(vj) != None and self.pel_dict[vj] and dists[vj] < closest_pill:
                                # Mark this tile as the closest pill
                                closest_pill = dists[vj]
                            # If the current tile is the farther than the previous maximum distance
                            if dists[vj] > maximum_path:
                                # Mark this tile as the maximum length path
                                maximum_path = dists[vj]
                if closest_ghost == np.inf or maximum_path == 0:
                    scared_ghosts[i] = 0.0
                # For each direction, set the scared ghosts input variable to the distance of the nearest scared ghost for that direction 
                else:
                    scared_ghosts[i] = (maximum_path - closest_ghost) / maximum_path
                if closest_pill == np.inf or maximum_path == 0:
                    pills[i] = 0.0
                # For each direction, set the pills input variable to the distance to the nearest pill for that direction
                else:
                    pills[i] = (maximum_path - closest_pill) / maximum_path
                if inter_dist == np.inf or maximum_path == 0:
                    ghosts_pos[i] = 0.0
                # For each direction, set the ghosts position input variable to the distance from the nearest intersection to the nearest ghost
                # from that intersection for that direction
                else:
                    shortest = np.inf
                    for j in range(len(ghost_dists[nearest_inter])):
                        if ghost_dists[nearest_inter][j] < shortest:
                            shortest = ghost_dists[nearest_inter][j]
                    ghosts_pos[i] = (maximum_path + inter_dist - (shortest - inter_dist)) / maximum_path
        # Iterate through the intersection graph to mark which intersections are closer to Pacman than any of the ghosts
        for inter in self.intersections:
            if ghost_dists.get(inter) != None:
                for j in range(len(ghost_dists[inter])):
                    if dists[inter] > ghost_dists[inter][j]:
                        self.intersections[inter] = False
        # Find the intersections that are closest in each direction Pacman can turn
        entrapment = [0.0, 0.0, 0.0, 0.0]
        if self.nodes.getNodeFromTiles(pac_pos[0], pac_pos[1]) != None:
            for i in range(4):
                if self.inter_graph.vertices[pac_pos][i] != None:
                    entrapment[i] = self.inter_graph.vertices[pac_pos][i]
        else:
            first_index, second_index, direc = -1, -1, -1
            if self.pacman.direction == UP:
                first_index, second_index, direc = 0, 1, DOWN
            elif self.pacman.direction == DOWN:
                first_index, second_index, direc = 1, 2, UP
            elif self.pacman.direction == LEFT:
                first_index, second_index, direc = 2, 3, RIGHT
            else:
                first_index, second_index, direc = 3, 2, LEFT
            entrapment[first_index] = self.getTile(self.pacman.target.position.x, self.pacman.target.position.y)
            entrapment[second_index] = self.getTile(self.pacman.target.neighbors[direc].position.x, self.pacman.target.neighbors[direc].position.y)
        # Perform BFS on the intersection graph, finding all routes in which Pacman can safely make it 3 intersections away from danger
        num_inter = 3
        total_safe = 0
        safe_dir = [0, 0, 0, 0]
        while(num_inter > 0):
            for i in range(4):
                # Only search in valid directions
                if entrapment[i] == 0.0:
                    entrapment[i] = 1.0
                else:
                    # Initialize all instance variables
                    self.inter_graph.resetVisited()
                    queue = [entrapment[i]]
                    self.inter_graph.visited[entrapment[i]] = True
                    inter_dists = {}
                    for vertex in self.inter_graph.vertices:
                        inter_dists[vertex] = np.inf
                    inter_dists[entrapment[i]] = 1
                    while len(queue) != 0:
                        v = queue.pop(0)
                        for j in range(5):
                            if self.inter_graph.vertices.get(vj) != None:
                                vj = self.inter_graph.vertices[v][j]
                                # Only find intersections in which Pacman can escape to
                                if vj != None and not self.inter_graph.visited[vj] and self.intersections[vj]:
                                    inter_dists[vj] = inter_dists[v] + 1
                                    if inter_dists[vj] == num_inter:
                                        total_safe += 1
                                        safe_dir[i] += 1
                                        continue
                                    queue.append(vj)
                                    self.inter_graph.visited[vj] = True
            # If there are no safe routes consisting of 3 intersections, lower the threshold to 2 and so on
            if total_safe == 0:
                num_inter -= 1
            else:
                break
        # If no safe routes are found, initialize all entrapment input variable values to zero
        if total_safe == 0:
            entrapment = [1.0, 1.0, 1.0, 1.0]
        # Otherwise, they are equal to the (number of total safe routes minus all safe routes in the given direction) divided by the number of total safe routes 
        else:
            for i in range(4):
                entrapment[i] = (total_safe - safe_dir[i]) / total_safe
        # Reset intersection graph
        for inter in self.intersections:
            self.intersections[inter] = True
        # The directions input variable is a boolean value denoting whether or not Pacman is already moving in a certain direction
        directions = []
        if self.pacman.direction == UP:
            directions = [1, 0, 0, 0]
        elif self.pacman.direction == DOWN:
            directions = [0, 1, 0, 0]
        elif self.pacman.direction == LEFT:
            directions = [0, 0, 1, 0]
        elif self.pacman.direction == RIGHT:
            directions = [0, 0, 0, 1]
        else:
            directions = [0, 0, 0, 0]
        for i in range(4):
            inputs[i] = [level_progress, powerpill, pills[i], ghosts_pos[i], scared_ghosts[i], entrapment[i], directions[i]]
        # Send input variables over to Pacman for the Pacbot to use
        self.pacman.setInputs(inputs)

    # Function which returns whether the game is running or not
    def isRunning(self):
        return self.running

    # ( End edit )


