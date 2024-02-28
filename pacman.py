import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
import numpy as np

class Pacman(Entity):
    def __init__(self, node, bot):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)

        # ( Begin edit )
        # Alter the Pacman constructor to include a Pacbot instance variable and an inputs instance variable

        self.bot = bot
        self.up_inputs = [1, 1, 1, 1, 1, 1, 1]
        self.down_inputs = [1, 1, 1, 1, 1, 1, 1]
        self.left_inputs = [1, 1, 1, 1, 1, 1, 1]
        self.right_inputs = [1, 1, 1, 1, 1, 1, 1]

        # ( End edit )

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

        # ( Begin edit )
        # Edit Pacman update function so that its Pacbot neural network informs it on which way to turn

        inputs = [self.up_inputs, self.down_inputs, self.left_inputs, self.right_inputs]
        direction = self.bot.direction(inputs)

        # ( End edit )

        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP  

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False

    # ( Begin edit )
    # Function which sets the input variables for the Pacbot neural network

    def setInputs(self, i):
        self.up_inputs = i[0]
        self.down_inputs = i[1]
        self.left_inputs = i[2]
        self.right_inputs = i[3]

    # ( End edit )