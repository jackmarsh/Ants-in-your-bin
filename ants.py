import numpy as np

class Ant:
    def __init__(self, num_items, num_bins):
        """Initialises the variables for the Ant object"""
        self.num_items = num_items
        self.num_bins = num_bins
        self.selected_bins = []
        self.bin_weights = [0 for i in range(num_bins)]
    
    def select_bin(self, bins, s):
        """Selects a bin via psuedorandom proportional rule"""
        r = np.random.uniform(0, s)
        for i in range(len(bins)):
            r -= bins[i]
            if r <= 0:
                return i
    
    def calculate_path(self, graph):
        """Calculates Ant's path through pheromone graph"""
        for i in range(self.num_items):
            s = sum(graph.pheromones[i])
            self.selected_bins.append(self.select_bin(graph.pheromones[i], s))
    
    def calculate_fitness(self, items):
        """Calculates the fitness of Ant's path"""
        for i in range(len(self.selected_bins)):
            # Weigh each bin
            self.bin_weights[self.selected_bins[i]] += items[i]
        # Set the fitness
        self.fitness = max(self.bin_weights) - min(self.bin_weights)


class Graph:
    def __init__(self, num_items, num_bins, mx=0):
        """Initialises the variables for the Graph object"""
        self.num_items = num_items
        self.num_bins = num_bins
        self.mx = mx
        # Initialise pheromones according to ACO implementation
        if self.mx != 0:
            # to self.
            self.pheromones = np.random.uniform(self.mx, self.mx, (self.num_items, self.num_bins))
        else:
            # Between 0 and 1
            self.pheromones = np.random.uniform(0, 1, (self.num_items, self.num_bins))
    
    def evaporate(self, evaporation_rate):
        """Evaporate the pheromone graph"""
        self.pheromones *= evaporation_rate
    
    def update(self, ant):
        """Update the pheromone graph based on selected bins"""
        for i in range(self.num_items):
            self.pheromones[i][ant.selected_bins[i]] += 100/ant.fitness
    
    def maxmin(self, mx, mn):
        """Limit the pheromone graph between mx and mn limits"""
        self.pheromones[self.pheromones > mx] = mx
        self.pheromones[self.pheromones < mn] = mn
    
    def reinitialise(self, stag):
        """Reinitialise the pheromone graph if stagnation occurs"""
        self.pheromones *= stag

class ACO:
    def __init__(self, evap, n_bins, n_items, n_paths, evals, maxmin=False):
        """Initialises the variables for the ACO object"""
        self.evaporation_rate = evap
        self.num_bins = n_bins
        self.num_items = n_items
        self.num_paths = n_paths
        self.evaluations = evals
        # If True, implements Max-Min
        self.maxmin = maxmin
        # Initialise variables according to ACO implementation
        if self.maxmin:
            self.mx = 1/(1-self.evaporation_rate)
            self.p = 1/self.num_bins
    
    def init_items(self):
        """Initialise items based on the BPP"""
        if self.num_bins == 10:
            return np.random.permutation(range(1, self.num_items+1))
        elif self.num_bins == 50:
            r = range(1, self.num_items+1)
            return np.random.permutation(r)*r/2
    
    def optimise(self):
        """Runs the ACO Optimisation"""

        # Create the item list
        items = self.init_items()
        
        # Create graph depending on ACO implementation
        if self.maxmin:
            # This is for Max-Min
            g = Graph(self.num_items, self.num_bins, self.mx)
        else:
            # This is for standard
            g = Graph(self.num_items, self.num_bins)

        # Stagnation variables
        last_fitness = 0
        stagnation = 0
        global_best = 1e10

        for i in range(self.evaluations):
            # Generate p paths
            ants = [Ant(self.num_items, self.num_bins) for ant in range(self.num_paths)]
            
            # For each ant calculate path and fitness
            for ant in ants:
                ant.calculate_path(g)
                ant.calculate_fitness(items)
            
            # Evaporate the pheromone graph
            g.evaporate(self.evaporation_rate)
            
            # Find the ant with best fitness
            best_ant = min(ants, key=lambda ant : ant.fitness)
            
            if self.maxmin:
                # Update global best if condition met
                if best_ant.fitness < global_best:
                    global_best = best_ant.fitness

                # Calculate max and min values
                self.mx = (1/(1-self.evaporation_rate))*(100/global_best)
                self.mn = (self.mx * (1-(self.p)))/((self.num_items/2)-1)*(self.p)
                # Ensure min is not greater than max
                if self.mn > self.mx:
                    self.mn = self.mx
                
                # Update with best ant only
                g.update(best_ant)
                # Limit the pheromone graph between max and min
                g.maxmin(self.mx, self.mn)

                # Increment if stagnated, reset if not
                if last_fitness == best_ant.fitness:
                    stagnation += 1
                else:
                    stagnation = 0

                # Reinitialise if stagnated for 200 evaluations
                if stagnation >= 200:
                    g.reinitialise(1/(global_best*self.mn))
                    g.maxmin(self.mx, self.mn)
                    stagnation = 0

                last_fitness = best_ant.fitness
            else:
                # Update with all ants
                for ant in ants:
                    g.update(ant)
          
            if i%100 == 0:
                print("Evaluation:",i)
                print("Global fitness:",best_ant.fitness)
                print("")
                
        self.global_fitness = min(ants, key=lambda ant : ant.fitness).fitness

def BPP1(maxmin=False):
    """Bin-Packing Problem 1 with best parameters"""
    evap_rate = 0.9
    bins = 10
    num_items = 200
    paths = 100
    evaluations = 10000

    print("Bin Packing Problem 1")
    print("Evap: "+str(evap_rate)+" Paths:"+str(paths))
    aco = ACO(evap_rate, bins, num_items, paths, evaluations, maxmin)
    aco.optimise()

    return aco.global_fitness

def BPP2(maxmin=False):
    """Bin-Packing Problem 2 with best parameters"""
    evap_rate = 0.9
    bins = 50
    num_items = 200
    paths = 100
    evaluations = 10000

    print("Bin Packing Problem 2")
    print("Evap: "+str(evap_rate)+" Paths:"+str(paths))
    aco = ACO(evap_rate, bins, num_items, paths, evaluations, maxmin)
    aco.optimise()

    return aco.global_fitness

def main():
    """Main method. Uncomment any test to perform it"""
    bpp1 = BPP1()
    print("Final fitness:",bpp1)
    #bpp2 = BPP2()
    #print("Final fitness:",bpp2)
    #maxmin1 = BPP1(True)
    #print("Final fitness:",maxmin1)
    #maxmin2 = BPP2(True)
    #print("Final fitness:",maxmin1)

main()