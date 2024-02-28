# Pacbot
## Project
This is a program I created that implements the NEAT (Neuro-Evolution of Augmenting Topologies) algorithm in Python to generate the optimal structure of a neural network for playing the game of Pac-Man. Python has their own framework for the NEAT algorithm already documented, but I wanted to learn the ins and outs of this mode of artificial intelligence and I wished to challenge myself with a complex project and to learn something new. 

## Disclaimer
In the run.py, nodes.py, and pacman.py, I only wrote the sections of code within the ( Begin edit ) and ( End edit ) comments, the rest of the code is from the website: (https://pacmancode.com/). Many thank you's to the creator of that website. 

## Files
#### NEAT.py
This file contains all the various classes and code necessary to begin and continue evolution of neural networks in an attempt to find a structure ideal for gameplay.
#### run.py
I modified this file in order to extract the input values that are to be fed into the Pacbot neural network so that the correct direction to turn can be calculated.
#### nodes.py
Created two new classes in this file, one is a graph of all moveable tiles in a level, and the other is a graph of all moveable nodes in a level.
#### pacman.py
Edited this file so that Pacbot would inform Pacman's moves and so that Pacman would be receiving all input values.

## Resources Utilized
I would like to thank the authors of the websites listed for the information and code they provided to the public; all of these sites were of great help while I was attempting to put together this project. 
#### pacmancode.com
The website https://pacmancode.com/ is where I was able to find the Python implementation of the game of Pacman. I modified three of these files and added one of my own in order to execute my NEAT program.
#### Luuk Bom, Ruud Henken, and Marco Wiering
These are the authors of the paper "Reinforcement Learning to Train Ms. Pac-Man Using Higher-order Action-relative Inputs" (https://www.researchgate.net/profile/Marco-Wiering/publication/236645821_Reinforcement_Learning_to_Train_Ms_Pac-Man_Using_Higher-order_Action-relative_Inputs/links/0deec518a22042f5d7000000/Reinforcement-Learning-to-Train-Ms-Pac-Man-Using-Higher-order-Action-relative-Inputs.pdf) which I used to develop the necessary inputs that my Pacbot would take in to determine the direction to move in. 
#### Kenneth O. Stanley and Risto Miikkulainen
These are the authors of the orginal paper that gives the theory and mathematics behind the Neuro-Evolution of Augmenting Topologies process. The paper is called "Evolving Neural Networks through Augmenting Topologies" and can be found at this link: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

## Running and Use
In order for this program to work, one needs to visit the site https://pacmancode.com/ and download the Pacman_Complete zip file at the bottom of the page. Replace the run.py, nodes.py, and pacman.py files in the folder with the ones in this project and insert the NEAT.py file from this project into the folder. From there, enter into a Python3 and Pygame enabled terminal and run the NEAT.py file. This should begin the Neuro-Evolution process.

## Results
The program works very well at the development and improvement of the Pacbot neural network. In my trials the average fitness of the population steadily improved from generation to generation as more connections were made and more weights were fine tuned. The biggest drawback to this program is that it takes an extensive amount of time for it to work; each individual of a population must play their own game of Pacman in order to determine the fitness of the individual. The runtime could possibly be improved using threading to run multiple games at the same time, but my attempt at this proved to be ineffective as my computer couldn't handle the workload. 
