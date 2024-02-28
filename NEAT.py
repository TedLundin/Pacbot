from constants import *
from run import GameController
import numpy as np
import random
import time


# Node class
# A Node is a part of a Genome which represents a neuron in a neural network
# Each Node has an identifying number and a type: input, output, or hidden
class Node():
    def __init__(self, number, node_type):
        self.number = number
        self.node_type = node_type
    # Returns number instance variable
    def getNumber(self):
        return self.number
    # Returns node_type instance variable
    def getNodeType(self):
        return self.node_type

# Connection class
# A Connection is a part of a Genome which represents a link between two neurons in a neural network
# Each Connection has an in node, and out node, a weight, an enabled boolean, and an indentifying number
class Connection():
    def __init__(self, in_node, out_node, weight, is_enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number
    # Returns in_node instance variable
    def getInNode(self):
        return self.in_node
    # Returns out_node instance variable
    def getOutNode(self):
        return self.out_node
    # Returns weight instance variable
    def getWeight(self):
        return self.weight
    # Sets the value of the weight instance variable
    def setWeight(self, w):
        self.weight = w
    # Returns the is_enabled instance variable
    def getIsEnabled(self):
        return self.is_enabled
    # Sets the value of the is_enabled instance variable
    def setIsEnabled(self, i):
        self.is_enabled = i
    # Returns the innovation_number instance variable
    def getInnovationNumber(self):
        return self.innovation_number

# Genome class
# A Genome is a less intensive way of storing the information that makes up a neural network
# Each Genome is a collection of Nodes and Connections between nodes
class Genome():
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
        self.fitness = 0
    # Returns the nodes instance variable
    def getNodes(self):
        return self.nodes
    # Adds a Node to the Genome
    def addNode(self, n):
        self.nodes.append(n)
    # Returns the connections instance variable
    def getConnections(self):
        return self.connections
    # Adds a Connection to the Genome
    def addConnection(self, c):
        self.connections.append(c)
    # Returns the fitness instance variable
    def getFitness(self):
        return self.fitness
    # Sets the value of the fitness instance variable
    def setFitness(self, f):
        self.fitness = f
    # Prints out the Genome in a legible manner
    def printGenome(self):
        print("Nodes: " + str(len(self.nodes)) + ", Connections: " + str(len(self.connections)) + ", Fitness: " + str(self.getFitness()))
        print("In Nodes:\tWeights:\tOut Nodes:\tInnovation Number:")
        for c in self.connections:
            if c.getIsEnabled():
                print(str(c.getInNode()) + "\t\t" + str(int(c.getWeight() * 10000) / 10000.0) + "\t\t" + str(c.getOutNode()) + "\t\t" + str(c.getInnovationNumber()))

# Population class
# The Population class is a means of generating the optimal structure for playing a game of Pacman
# A Population is given a base network and a fitness goal to work towards and simulates ongoing generations until an ideal structure is reached
class Population():
    connection_count = 0
    def __init__(self, nodes, connections, size, goal, constants, num_spec):
        self.nodes = nodes
        self.connections = connections
        Population.connection_count = len(self.connections)
        self.size = size
        self.population = []
        self.wheel = []
        self.generation = 0
        self.goal = goal
        self.constants = constants
        self.num_spec = num_spec
        self.species = []
        ind = 0
        for i in range(self.size):
            for j in range(0, i + 1):
                self.wheel.append(self.size - 1 - i)
                ind += 1
    # Adds a Connection to the Population's Connection bank
    def addConnection(self, i, o):
        if self.connectionExists(i, o) == -1:
            self.connections.append(Connection(i, o, 0.0, True, Population.connection_count))
            Population.connection_count += 1
    # Checks to see if a Connection between two Nodes exists anywhere else in the Population
    def connectionExists(self, n1, n2):
        for i in range(len(self.connections)):
            if self.connections[i].getInNode() == n1 and self.connections[i].getOutNode() == n2:
                return self.connections[i].getInnovationNumber()
        return -1
    # Initializes the original Population by generating {self.size} number of networks and randomly creating a few Connections for each network to give a base to start from
    def initializePopulation(self):
        for i in range(self.size):
            # Create a new Genome from the base structure
            n = []
            for j in range(len(self.nodes)):
                n.append(Node(self.nodes[j].getNumber(), self.nodes[j].getNodeType()))
            c = []
            for j in range(len(self.connections)):
                cj = self.connections[j]
                w = random.uniform(-0.01, 0.01)
                while w == 0:
                    w = random.uniform(-0.01, 0.01)
                c.append(Connection(cj.getInNode(), cj.getOutNode(), w, cj.getIsEnabled(), cj.getInnovationNumber()))
            g = Genome(n, c)
            self.population.append(g)
        # Mutate each Genome's structure to create initial Connections to work off
        for i in range(self.size):
            for j in range(3):
                self.population[i] = self.mutateStructure(self.population[i])
    # Modifies the structure of a Genome by either adding a Connection between two Nodes or inserting a Node between two connected Nodes
    def mutateStructure(self, g):
        # Find two random Nodes in a Genome
        n1 = random.randint(0, len(g.getNodes()) - 1)
        n2 = random.randint(0, len(g.getNodes()) - 1)
        while n2 == n1 or g.getNodes()[n1].getNodeType() == g.getNodes()[n2].getNodeType():
            n2 = random.randint(0, len(g.getNodes()) - 1)
        n1t = g.getNodes()[n1].getNodeType()
        n2t = g.getNodes()[n2].getNodeType()
        if (n1t == "output" and (n2t == "hidden" or n2t == "input")) or (n1t == "hidden" and n2t == "input"):
            temp = n1
            n1 = n2
            n2 = temp
        # Create new Genome to be mutated
        n = g.getNodes()
        c = g.getConnections()
        mn = []
        mc = []
        for i in range(len(n)):
            mn.append(Node(n[i].getNumber(), n[i].getNodeType()))
        for i in range(len(c)):
            mc.append(Connection(c[i].getInNode(), c[i].getOutNode(), c[i].getWeight(), c[i].getIsEnabled(), c[i].getInnovationNumber()))
        mutated = Genome(mn, mc)
        # Check to see if a Connection already exists somewhere in the Population between the two randomly chosen Nodes
        cin = self.connectionExists(n1, n2)
        # If no such Connection exists, create a new one, add it to the mutated Genome, and add it to the Population's Connection bank
        if cin == -1:
            w = random.uniform(-0.1, 0.1)
            while w == 0:
                w = random.uniform(-0.1, 0.1)
            self.addConnection(n1, n2)
            mutated.addConnection(Connection(n1, n2, w, True, self.connectionExists(n1, n2)))
        # Otherwise, if the Connection is found in the Population's bank
        else:
            c = mutated.getConnections()
            found = False
            # Iterate through the Genome's Connections and check to see if it exists in there
            for i in range(len(c)):
                # If the Connection is found and it is disabled, re-enable it with a new weight
                if c[i].getInnovationNumber() == cin and not c[i].getIsEnabled():
                    c[i].setIsEnabled(True)
                    w = random.uniform(-0.1, 0.1)
                    while w == 0:
                        w = random.uniform(-0.1, 0.1)
                    c[i].setWeight(w)
                    found = True
                    break
                # If the Connection is found and it is enabled, disable it and add a new Node and connect the two previously connected Nodes to the new Node
                elif c[i].getInnovationNumber() == cin and c[i].getIsEnabled():
                    c[i].setIsEnabled(False)
                    nn = len(mutated.getNodes())
                    mutated.addNode(Node(nn, "hidden"))
                    in1 = self.connectionExists(n1, nn)
                    in2 = self.connectionExists(nn, n2)
                    if in1 == -1:
                        self.addConnection(n1, nn)
                        mutated.addConnection(Connection(n1, nn, 1.0, True, self.connectionExists(n1, nn)))
                    else:
                        mutated.addConnection(Connection(n1, nn, 1.0, True, in1))
                    if in2 == -1:
                        self.addConnection(nn, n2)
                        mutated.addConnection(Connection(nn, n2, c[i].getWeight(), True, self.connectionExists(nn, n2)))
                    else:
                        mutated.addConnection(Connection(nn, n2, c[i].getWeight(), True, in2))
                    found = True
                    break
            # If the Connection is not found in the Genome, add it in
            if not found:
                w = random.uniform(-0.1, 0.1)
                while w == 0:
                    w = random.uniform(-0.1, 0.1)
                mutated.addConnection(Connection(n1, n2, w, True, cin))
        return mutated
    # Modifies the weights of a Genome by 10 to 50%
    def mutateWeights(self, g):
        # Duplicate the given Genome so that the new Genome can be mutated
        n = g.getNodes()
        c = g.getConnections()
        mn = []
        mc = []
        for i in range(len(n)):
            mn.append(Node(n[i].getNumber(), n[i].getNodeType()))
        for i in range(len(c)):
            mc.append(Connection(c[i].getInNode(), c[i].getOutNode(), c[i].getWeight(), c[i].getIsEnabled(), c[i].getInnovationNumber()))
        mutated = Genome(mn, mc)
        # Iterate through each weight in the Genome, altering them by 10 to 50%
        for i in range(len(mutated.getConnections())):
            r = random.randint(0, 1)
            # If the weight equals zero, give it a new value with 50% probability
            if mutated.getConnections()[i].getWeight() == 0:
                if r == 1:
                    mutated.getConnections()[i].setWeight(random.uniform(-0.5, 0.5))
            else: 
                mult = 1
                if r == 0:
                    mult *= -1
                mod = random.uniform(0.1, 0.5) * mutated.getConnections()[i].getWeight()
                mutated.getConnections()[i].setWeight(mutated.getConnections()[i].getWeight() + mult * mod)
        return mutated
    # Generates a child Genome given two parent Genomes
    def crossover(self, m, f):
        dc = []
        dn = []
        sc = []
        # Select the more fit parent to be the dominant one
        if m.getFitness() > f.getFitness():
            dc = m.getConnections()
            dn = m.getNodes()
            sc = f.getConnections()
        else:
            dc = f.getConnections()
            dn = f.getNodes()
            sc = m.getConnections()
        # Copy the structure of the dominant parent to create the child
        cn = []
        for i in range(len(dn)):
            cn.append(Node(dn[i].getNumber(), dn[i].getNodeType()))
        cc = []
        for i in range(len(dc)):
            cc.append(Connection(dc[i].getInNode(), dc[i].getOutNode(), dc[i].getWeight(), dc[i].getIsEnabled(), dc[i].getInnovationNumber()))
        # For each Connection that exists in both Genomes, randomly pick a weight from one of them and give it to the child
        for i in range(len(cc)):
            for j in range(len(sc)):
                if cc[i].getInnovationNumber() == sc[j].getInnovationNumber():
                    r = random.randint(0, 1)
                    if r == 0:
                        cc[i].setWeight(sc[j].getWeight())
                    break
        return Genome(cn, cc)
    # For each individual in a Population, let them run a game of Pacman and score them on how well they do
    def updateFitnesses(self):
        # For each individual in the Population
        for i in range(self.size):
            # Initialize a game of Pacman with the Genome's corresponding Pacbot at the helm of Pacman
            start = time.time()
            game = GameController(Pacbot(self.population[i]))
            game.startGame()
            while game.isRunning():
                game.update()
            end = time.time()
            # Calculate the fitness by multiplying the ratio of the points scored to the maximum single level score by the total time in seconds survived
            elapsed = end - start
            score = game.score / 19600.0
            fitness = int(elapsed * score * 10000) / 10000.0
            # Print the individual's fitness and Genome structure to the terminal
            print("individual " + str(i))
            self.population[i].setFitness(fitness)
            self.population[i].printGenome()
            if i != self.size - 1:
                print()
            del(game)
    # Split each Individual in a Population into an assortment of different species 
    def speciate(self):
        threshold = 2.5
        mint = 0.1
        maxt = 4
        # Find an ideal distance threshold for dividing the Population into num_species +/- 1
        while True:
            self.species = []
            self.species.append([0])
            # For each individual in the Population, check to see if the distance between them and the species representative is less than the threshold
            # If it is, add that individual to that species, otherwise, if the individual doesn't fit into any species, create a new species and have the
            # individual be its representative
            for i in range(1, self.size):
                num = 0
                ins = False
                while num < len(self.species):
                    if self.distance(self.population[self.species[num][0]], self.population[i]) < threshold:
                        self.species[num].append(i)
                        ins = True
                        break
                    num += 1
                if not ins:
                    self.species.append([i])
            # If the number of species is off, try again with a new threshold
            if len(self.species) < self.num_spec - 1:
                maxt = threshold
            elif len(self.species) > self.num_spec + 1:
                mint = threshold
            # If the number of species is within one of num_species, return 1 for a successful speciation run
            else:
                return 1
            # If speciation is not possible, return 0
            if abs(maxt - mint) < 0.01:
                return 0
            # Calculate new threshold
            threshold = (mint + maxt) / 2
    # Generate the next generation of individuals from the previous generation (without using speciation)
    def nextGeneration(self):
        pop = []
        # Create self.size number of new individuals
        for i in range(self.size):
            child = None
            # Randomly pick two parents from the Population
            m = self.wheel[random.randint(0, len(self.wheel) - 1)]
            p = self.wheel[random.randint(0, len(self.wheel) - 1)]
            while p == m:
                p = self.wheel[random.randint(0, len(self.wheel) - 1)]
            mum = self.population[m]
            pap = self.population[p]
            # Call the crossover function to get a child from the randomly chosen parents
            child = self.crossover(mum, pap)
            ms = random.randint(1, 100)
            # Randomly mutate the child's structure with 10% probability
            if ms <= 10:
                child = self.mutateStructure(child)
            mw = random.randint(1, 100)
            # Randomly mutate the child's weights with 90% probability
            if mw <= 90:
                child = self.mutateWeights(child)
            pop.append(child)
        # Set the Population to consist of the newly generated children
        self.population = []
        for i in range(len(pop)):
            self.population.append(pop[i])
    # Generate the next generation of individuals from the previous generation (while using speciation)
    def nextGenerationWithSpecies(self):
        # Calculate the number of children each species should get
        nc = []
        pavf = 0.0
        for i in range(self.size):
            pavf += self.population[i].getFitness()
        pavf /= self.size
        for i in range(len(self.species)):
            savf = 0.0
            for j in range(len(self.species[i])):
                savf += self.population[self.species[i][j]].getFitness()
            savf /= len(self.species[i])
            nc.append(int(savf / pavf * len(self.species[i]) + 0.5))
        nct = sum(nc)
        for i in range(len(nc)):
            nc[i] = int((nc[i] / nct) * self.size + 0.5)
        pop = []
        # For each species
        for i in range(len(nc)):
            # Generate the amount of children each has decidedly gotten
            for j in range(nc[i]):
                child = None
                # If the species only has one member, asexually reproduce
                if len(self.species[i]) > 1:
                    m = random.randint(0, len(self.species[i]) - 1)
                    f = random.randint(0, len(self.species[i]) - 1)
                    while f == m:
                        f = random.randint(0, len(self.species[i]) - 1)
                    child = self.crossover(self.population[self.species[i][m]], self.population[self.species[i][f]])
                # Otherwise, perform crossover with two randomly selected individuals from the species
                else:
                    child = self.population[self.species[i][0]]
                ms = random.randint(1, 100)
                # Randomly mutate the child's structure with 10% probability
                if ms <= 10:
                    child = self.mutateStructure(child)
                mw = random.randint(1, 100)
                # Randomly mutate the child's weights with 90% probability
                if mw <= 90:
                    child = self.mutateWeights(child)
                pop.append(child)
        # If there are remaining slots in the next generation for children, select the fittest individuals to move on to the next generation
        leftovers = []
        for i in range(self.size - len(pop)):
            pop.append(self.population[i])
        # Set the Population to consist of the newly generated children
        self.population = []
        for i in range(len(pop)):
            self.population.append(pop[i])
    # The function numerically approximates the difference between two genomes
    def distance(self, g1, g2):
        g1c = g1.getConnections()
        g2c = g2.getConnections()
        # Sum of the difference in weights between the two Genomes
        W = 0.0
        # Number of disjoint Connections (number of Connections that only exist in one Genome)
        D = 0
        # Number of excess Connections (absolute value in the difference between the amount of Connections in each Genome)
        E = 0
        # Number of Connections the larger Genome has
        N = [0, 0]
        for i in range(len(g1c)):
            isin = False
            if g1c[i].getIsEnabled():
                E += 1
                N[0] += 1
            for j in range(len(g2c)):
                if g1c[i].getIsEnabled() and g2c[j].getIsEnabled() and g1c[i].getInnovationNumber() == g2c[j].getInnovationNumber():
                    isin = True
                    W += abs(g1c[i].getWeight() - g2c[j].getWeight())
            if not isin:
                D += 1
        for i in range(len(g2c)):
            isin = False
            if g2c[i].getIsEnabled():
                E -= 1
                N[1] += 1
            for j in range(len(g1c)):
                if g2c[i].getIsEnabled() and g1c[j].getIsEnabled() and g2c[i].getInnovationNumber() == g1c[j].getInnovationNumber():
                    isin = True
            if not isin:
                D += 1
        E = abs(E)
        if N[0] > N[1]:
            N = N[0]
        else:
            N = N[1]
        if N == 0:
            return -1
        # Calculate and return distance by adding E/N, D/N, and the sum of the weights all multiplied by some constants
        dist = self.constants[0] * E / N + self.constants[1] * D / N + self.constants[2] * W
        return dist
    # Sort the Population by fitness
    def sortPopulation(self):
        no = []
        # For each individual in the Population
        for i in range(len(self.population)):
            mf = self.population[0].getFitness()
            ind = 0
            # Find the fittest remaining individual in the unsorted Population and append it to the end of the sorted array
            for j in range(len(self.population)):
                if self.population[j].getFitness() > mf:
                    mf = self.population[j].getFitness()
                    ind = j
            no.append(self.population.pop(ind))
        # Reorder the Population with the newly sorted array of Genomes
        for i in range(len(no)):
            self.population.append(no[i])
    # Use the previous functions to simulate evolution and find an ideal structure for a Pacman playing neural network
    def simulate(self):
        # Create initial Population
        self.initializePopulation()
        # Continue to create new generations until one network is able to achieve the goal fitness
        while(True):
            self.generation += 1
            print("\nGeneration " + str(self.generation))
            print("=============")
            # Find fitnesses of the current generation
            self.updateFitnesses()
            # Sort the Population
            self.sortPopulation()
            # Calculate and print to terminal the average fitness of the generation
            avg_fit = 0.0
            for i in range(self.size):
                avg_fit += self.population[i].getFitness()
            avg_fit = int(avg_fit / self.size * 10000) / 10000.0
            print("=========================")
            print("Average Fitness: " + str(avg_fit))
            # Check to see if the generation consists of an individual with the goal fitness or greater
            if self.population[0].getFitness() >= self.goal:
                break
            # Try to speciate the generation and create a new generation of individuals
            if self.speciate() == 1:
                self.nextGenerationWithSpecies()
            else:
                self.nextGeneration()

# Neuron class
# A Neuron is a part of a neural network that takes on a value
# Each Neuron has an identifying number, a value, a set of input Neurons, a set of weights, and an is_set boolean
class Neuron():
    def __init__(self, number, value, input_neurons, input_weights, is_set):
        self.number = number
        self.value = value
        self.input_neurons = input_neurons
        self.input_weights = input_weights
        self.is_set = is_set
    # Returns the number instance variable
    def getNumber(self):
        return self.number
    # Returns the value instance variable
    def getValue(self):
        return self.value
    # Sets the value of the value instance variable
    def setValue(self, v):
        self.value = v
    # Returns the input_neurons instance variable
    def getInputNeurons(self):
        return self.input_neurons
    # Returns the input_weights instance variable
    def getInputWeights(self):
        return self.input_weights
    # Returns the is_set instance variable
    def getIsSet(self):
        return self.is_set
    # Sets the value of the is_set instance variable
    def setIsSet(self, i):
        self.is_set = i

# Pacbot class
# A Pacbot is a neural network that is tasked with finding the ideal direction to move in the game of Pacman given a set of relevant inputs
# A Pacbot is given a Genome, and from the information encoded in the Genome, creates a neural network
class Pacbot():
    def __init__(self, structure):
        # Get a set of Neurons from the Genome structure
        self.neurons = self.genomeToNetwork(structure)
        self.input_indices = []
        self.output_indices = []
        self.hidden_indices = []
        # Order all Neurons by type
        for i in range(len(structure.getNodes())):
            if structure.getNodes()[i].getNodeType() == "input":
                self.input_indices.append(i)
            elif structure.getNodes()[i].getNodeType() == "output":
                self.output_indices.append(i)
            else:
                self.hidden_indices.append(i)
    # Converts a Genome into a neural network
    def genomeToNetwork(self, g):
        n = g.getNodes()
        c = g.getConnections()
        network = []
        # Parse through the Nodes of the Genome and create a Neuron for each one
        for i in range(len(n)):
            network.append(Neuron(n[i].getNumber(), None, [], [], False))
        # Parse through the Connections of the Genome and find each Neuron's input Neurons and weights
        for i in range(len(network)):
            iw = {}
            for j in range(len(c)):
                if c[j].getIsEnabled() and c[j].getOutNode() == network[i].getNumber():
                    iw[c[j].getInNode()] = c[j].getWeight()
            for j in range(len(network)):
                if iw.get(network[j].getNumber()) != None:
                    network[i].getInputNeurons().append(network[j])
                    network[i].getInputWeights().append(iw.get(network[j].getNumber()))
        return network
    # Returns whether all outputs of a neural network are set or not
    def outputsSet(self):
        for i in range(len(self.output_indices)):
            if not self.neurons[self.output_indices[i]].getIsSet():
                return False
        return True
    # Returns whether all inputs of a Neuron are set or not
    def allInputsSet(self, n):
        c = self.neurons[n].getInputNeurons()
        for i in range(len(c)):
            if not c[i].getIsSet():
                return False
        return True
    # Finds the ideal direction to move given the inputs according to the Pacbot neural network
    def direction(self, inputs):
        outputs = []
        # Find a value for each direction by running the neural network using the corresponding input values for each of the four directions
        for i in range(4):
            # Set the values of the input Neurons
            for j in range(len(self.input_indices)):
                self.neurons[self.input_indices[j]].setValue(inputs[i][j])
                self.neurons[self.input_indices[j]].setIsSet(True)
            # Cascade through the following Neurons in the network until every Neuron's value is updated and set
            while not self.outputsSet():
                for j in range(len(self.neurons)):
                    if j not in self.input_indices and self.allInputsSet(j):
                        # Multiply all input Neuron values by the corresponding weights and sum them together
                        s = 0
                        inp = self.neurons[j].getInputNeurons()
                        wei = self.neurons[j].getInputWeights()
                        for k in range(len(inp)):
                            s += wei[k] * inp[k].getValue()
                        # Apply activation function (sigmoid function) and set the Neuron
                        self.neurons[j].setValue(1 / (1 + np.exp(-s)))
                        self.neurons[j].setIsSet(True)
            # Append the value for the direction to the outputs array
            outputs.append(self.neurons[self.output_indices[0]].getValue())
            # Reset the Neurons
            for j in range(len(self.neurons)):
                self.neurons[j].setIsSet(False)
        # Find the argument of the outputs array that is the largest; that is the direction that Pacman should move in
        ind = np.argmax(outputs)
        if ind == 0:
            return UP
        elif ind == 1:
            return DOWN
        elif ind == 2:
            return LEFT
        return RIGHT


# Find the optimal structure
if __name__ == "__main__":
    nodes = []
    for i in range(7):
        nodes.append(Node(i, "input"))
    for i in range(1):
        nodes.append(Node(i + 7, "output"))
    pop = Population(nodes, [], 50, 25, [1, 1, 0.5], 6)
    pop.simulate()
    pop.population[0].printGenome()