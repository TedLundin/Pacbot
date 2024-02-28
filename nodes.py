import pygame
from vector import Vector2
from constants import *
import numpy as np

class Node(object):
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.neighbors = {UP:None, DOWN:None, LEFT:None, RIGHT:None, PORTAL:None}
        self.access = {UP:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       DOWN:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       LEFT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       RIGHT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT]}

    def denyAccess(self, direction, entity):
        if entity.name in self.access[direction]:
            self.access[direction].remove(entity.name)

    def allowAccess(self, direction, entity):
        if entity.name not in self.access[direction]:
            self.access[direction].append(entity.name)

    def render(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.position.asTuple()
                line_end = self.neighbors[n].position.asTuple()
                pygame.draw.line(screen, WHITE, line_start, line_end, 4)
                pygame.draw.circle(screen, RED, self.position.asInt(), 12)


class NodeGroup(object):
    def __init__(self, level):
        self.level = level
        self.nodesLUT = {}
        self.nodeSymbols = ['+', 'P', 'n']
        self.pathSymbols = ['.', '-', '|', 'p']
        data = self.readMazeFile(level)
        self.createNodeTable(data)
        self.connectHorizontally(data)
        self.connectVertically(data)
        self.homekey = None

    def readMazeFile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')

    def createNodeTable(self, data, xoffset=0, yoffset=0):
        for row in list(range(data.shape[0])):
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    x, y = self.constructKey(col+xoffset, row+yoffset)
                    self.nodesLUT[(x, y)] = Node(x, y)

    def constructKey(self, x, y):
        return x * TILEWIDTH, y * TILEHEIGHT


    def connectHorizontally(self, data, xoffset=0, yoffset=0):
        for row in list(range(data.shape[0])):
            key = None
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    if key is None:
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        self.nodesLUT[key].neighbors[RIGHT] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[LEFT] = self.nodesLUT[key]
                        key = otherkey
                elif data[row][col] not in self.pathSymbols:
                    key = None

    def connectVertically(self, data, xoffset=0, yoffset=0):
        dataT = data.transpose()
        for col in list(range(dataT.shape[0])):
            key = None
            for row in list(range(dataT.shape[1])):
                if dataT[col][row] in self.nodeSymbols:
                    if key is None:
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        self.nodesLUT[key].neighbors[DOWN] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[UP] = self.nodesLUT[key]
                        key = otherkey
                elif dataT[col][row] not in self.pathSymbols:
                    key = None


    def getStartTempNode(self):
        nodes = list(self.nodesLUT.values())
        return nodes[0]

    def setPortalPair(self, pair1, pair2):
        key1 = self.constructKey(*pair1)
        key2 = self.constructKey(*pair2)
        if key1 in self.nodesLUT.keys() and key2 in self.nodesLUT.keys():
            self.nodesLUT[key1].neighbors[PORTAL] = self.nodesLUT[key2]
            self.nodesLUT[key2].neighbors[PORTAL] = self.nodesLUT[key1]

    def createHomeNodes(self, xoffset, yoffset):
        homedata = np.array([['X','X','+','X','X'],
                             ['X','X','.','X','X'],
                             ['+','X','.','X','+'],
                             ['+','.','+','.','+'],
                             ['+','X','X','X','+']])

        self.createNodeTable(homedata, xoffset, yoffset)
        self.connectHorizontally(homedata, xoffset, yoffset)
        self.connectVertically(homedata, xoffset, yoffset)
        self.homekey = self.constructKey(xoffset+2, yoffset)
        return self.homekey

    def connectHomeNodes(self, homekey, otherkey, direction):     
        key = self.constructKey(*otherkey)
        self.nodesLUT[homekey].neighbors[direction] = self.nodesLUT[key]
        self.nodesLUT[key].neighbors[direction*-1] = self.nodesLUT[homekey]

    def getNodeFromPixels(self, xpixel, ypixel):
        if (xpixel, ypixel) in self.nodesLUT.keys():
            return self.nodesLUT[(xpixel, ypixel)]
        return None

    def getNodeFromTiles(self, col, row):
        x, y = self.constructKey(col, row)
        if (x, y) in self.nodesLUT.keys():
            return self.nodesLUT[(x, y)]
        return None

    def denyAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.denyAccess(direction, entity)

    def allowAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.allowAccess(direction, entity)

    def denyAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.denyAccess(col, row, direction, entity)

    def allowAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.allowAccess(col, row, direction, entity)

    def denyHomeAccess(self, entity):
        self.nodesLUT[self.homekey].denyAccess(DOWN, entity)

    def allowHomeAccess(self, entity):
        self.nodesLUT[self.homekey].allowAccess(DOWN, entity)

    def denyHomeAccessList(self, entities):
        for entity in entities:
            self.denyHomeAccess(entity)

    def allowHomeAccessList(self, entities):
        for entity in entities:
            self.allowHomeAccess(entity)

    def render(self, screen):
        for node in self.nodesLUT.values():
            node.render(screen)

# ( Begin edit )
# TileGraph class
# This is a class made for a breadth-first search that keeps a record of all tiles and their neighbors and which tiles have been visited
# The TileGraph class takes in a list of nodes in a level and creates a graph from the tiles that are between the nodes
class TileGraph():
    def __init__(self, nodes):
        self.vertices = {}
        self.visited = {}
        for node in nodes.nodesLUT:
            node_pos = self.getTile(node[0], node[1])
            self.vertices[node_pos] = [None, None, None, None]
            self.visited[node_pos] = False
            # Iterate through each node's neighbors
            for neighbor in nodes.nodesLUT[node].neighbors:
                # If the neighbor exists, find its position
                if nodes.nodesLUT[node].neighbors[neighbor] != None:
                    r, c = node_pos[0], node_pos[1]
                    neighbor_pos = self.getTile(nodes.nodesLUT[node].neighbors[neighbor].position.x, nodes.nodesLUT[node].neighbors[neighbor].position.y)
                    # If neighbor is up, add all tiles between the two nodes to the graph
                    if neighbor == UP:
                        c -= 1
                        self.vertices[node_pos][0] = (r, c)
                        if self.vertices.get((r, c)) == None:
                            while (r, c) != neighbor_pos:
                                self.vertices[(r, c)] = [(r, c - 1), (r, c + 1), None, None]
                                self.visited[(r, c)] = False
                                c -= 1
                    # If neighbor is down, add all tiles between the two nodes to the graph
                    elif neighbor == DOWN:
                        c += 1
                        self.vertices[node_pos][1] = (r, c)
                        if self.vertices.get((r, c)) == None:
                            while (r, c) != neighbor_pos:
                                self.vertices[(r, c)] = [(r, c - 1), (r, c + 1), None, None]
                                self.visited[(r, c)] = False
                                c += 1
                    # If neighbor is left, add all tiles between the two nodes to the graph
                    elif neighbor == LEFT:
                        r -= 1
                        self.vertices[node_pos][2] = (r, c)
                        if self.vertices.get((r, c)) == None:
                            while (r, c) != neighbor_pos:
                                self.vertices[(r, c)] = [None, None, (r - 1, c), (r + 1, c)]
                                self.visited[(r, c)] = False
                                r -= 1
                    # If neighbor is right, add all tiles between the two nodes to the graph
                    elif neighbor == RIGHT:
                        r += 1
                        self.vertices[node_pos][3] = (r, c)
                        if self.vertices.get((r, c)) == None:
                            while (r, c) != neighbor_pos:
                                self.vertices[(r, c)] = [None, None, (r - 1, c), (r + 1, c)]
                                self.visited[(r, c)] = False
                                r += 1
                    # If the neighbor is a portal pair, add those tiles as neighbors to the graph
                    else: 
                        if neighbor_pos[1] > node_pos[1]:
                            self.vertices[node_pos][0] = neighbor_pos
                        elif neighbor_pos[1] < node_pos[1]:
                            self.vertices[node_pos][1] = neighbor_pos
                        elif neighbor_pos[0] > node_pos[0]:
                            self.vertices[node_pos][2] = neighbor_pos
                        else:
                            self.vertices[node_pos][3] = neighbor_pos
                        self.visited[neighbor_pos] = False
    # Returns the graph instance variables
    def getGraph(self):
        return self.vertices, self.visited
    # Resets all visited booleans to false
    def resetVisited(self):
        for element in self.visited:
            self.visited[element] = False
    # Returns the tile a given pixel is in
    def getTile(self, x, y):
        return (int(x / TILEWIDTH + 0.5), int(y / TILEHEIGHT + 0.5))

# Class NodeGraph
# NodeGraph is a graph made up of all the intersections/nodes of a level
# NodeGraph takes in a TileGraph and a NodeGroup to create itself
class NodeGraph():
    def __init__(self, tile_graph, node_group):
        self.vertices = {}
        self.visited = {}
        for node in node_group.nodesLUT:
            node_pos = self.getTile(node[0], node[1])
            if tile_graph.vertices.get(node_pos) != None:
                self.vertices[node_pos] = []
                self.visited[node_pos] = False
                # Initialize all neighboring nodes for every node found in both node_group and tile_graph
                for neighbor in node_group.nodesLUT[node].neighbors:
                    if node_group.nodesLUT[node].neighbors[neighbor] != None:
                        neighbor_pos = self.getTile(node_group.nodesLUT[node].neighbors[neighbor].position.x, node_group.nodesLUT[node].neighbors[neighbor].position.y)
                        self.vertices[node_pos].append(neighbor_pos)
                    else:
                        self.vertices[node_pos].append(None)
    # Returns the elements of the NodeGraph
    def getGraph(self):
        return self.vertices, self.visited
    # Resets all the visited booleans to false
    def resetVisited(self):
        for element in self.visited:
            self.visited[element] = False
    # Returns the tile a given pixel is in
    def getTile(self, x, y):
        return (int(x / TILEWIDTH + 0.5), int(y / TILEHEIGHT + 0.5))

# ( End edit )