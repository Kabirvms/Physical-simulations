import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Person:
    def __init__(self, position, contagious, step_size):
        """Initializes a person with a position, infection status, and step size"""
        self.position = position
        self.contagious = contagious
        self.step_size = step_size
        self.direction = np.random.choice([0, np.pi])
        self.is_infected = contagious
        self.position_history = [[], []]
        self.position_history[0].append(position[0])
        self.position_history[1].append(position[1])

    def step(self, bounds, step_size):
        """Performs a single random walk step and impose boundary conditions on a rectangular room"""
        angle = np.random.normal(0, 2 * np.pi)
        print(angle)
        new_x = self.position[0] + step_size * np.cos(angle)
        new_y = self.position[1] + step_size * np.sin(angle)
        
        # Apply boundary conditions
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_y = np.clip(new_y, bounds[0], bounds[1])
            
        self.position = np.array([new_x, new_y])
        return self.position

class Map:
    def __init__(self, bounds, people_density, step_size, probably_infected):
        """initialise the environment with the people and defines infected, contagious and healthy people. Note for simplicity assume standard units
        Distance is in meters
        Time is in seconds

        Args:
            bounds (array): the boundary of the corridor
            people_density (float): people per square meter
            step_size (float): how big the steps are
            initially_infected (int): the number of people who are infected at the start    
        """
        self.bounds = bounds
        self.num_people = int(people_density * abs(bounds[0] - bounds[1])**2)
        self.no_infected = 0
        self.step_size = step_size
        self.all_people = []
        self.personal_space = 1
        self.contagious_tracking = [[], []]
        self.infection_range = 2

        # Generate people and assign initial infection status
        for all in range(self.num_people):
            pos = np.random.uniform(bounds[0], bounds[1], 2)
            step_size = np.random.uniform(5,1.5,10)
            if np.random.random(1) < probably_infected:
                # Create a person and add to the infected and contagious list
                person = Person(pos,True, step_size)
            else:
                # Creates a healthy person 
                person = Person(pos,False, step_size)
            self.all_people.append(person)
      
        
    def move(self, index):
        #initialise move validation
        move = False
        
        #Creates a vaild step for person at index
        proposed_updated_people = self.all_people.copy()
        while move == False:
            proposed_step = proposed_updated_people[index].step(self.bounds, self.step_size)
            move ,proposed_updated_people = self.position_check(proposed_updated_people, self.personal_space,self.infection_range,index,proposed_step)

        #Updates the person's position and the list of people
        self.all_people[index].person.position = proposed_step
        self.all_people[index].position_history[0].append(proposed_step[0])
        self.all_people[index].position_history[1].append(proposed_step[1])
        self.all_people = proposed_updated_people
        return self.all_people

        
    def position_check(self,proposed_updated_people, exclusion_zone, infection_range,proposed_step,index):
        """Check if a proposed step is valid and update the list of people
           focusing on the social distancing and infection range"""
        
        # Check if the proposed step is within the bounds
        #also point to add in social distancing

        for i in range(len(proposed_updated_people)):
            #checks if comparing to self
            if i == index:
                pass

            # Checks social distancing
            elif np.linalg.norm(proposed_updated_people[i] - proposed_step) < exclusion_zone:
                return False
            
            # Check if the proposed step infects another person
            # Then gives a temp asign incase its not a vailed position with socal distancing
            elif np.linalg.norm(proposed_updated_people[i] - proposed_step) < infection_range:
                proposed_updated_people[i].is_infected = True
        return True, proposed_updated_people
    

class Simulation:
    def __init__(self, bounds, people_density, step_size=1, probably_infected=0.01,time=1000):
        """Initializes the map and the simulation"""
        self.map = Map(bounds, people_density, step_size, probably_infected)
        self.time = 500

        
        # Create figure and visualization elements
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[0], bounds[1])
        # Create visualization elements
        self.scatter_all = self.ax.scatter(map.all_people[0],map.all_people[1], c='green', alpha=0.5, s=10, label='Healthy')
        self.scatter_infected = self.ax.scatter([], [], c='blue', alpha=0.8, s=15, label='Infected')
        self.scatter_contagious = self.ax.scatter([], [], c='red', alpha=1, s=25, label='Contagious')
        self.lines_path_of_contagious, = self.ax.plot([], [], c='orange', alpha=0.1, label='Path of Contagious')
     
        
        # Add labels and grid
        plt.xlabel('X Primary axis of travel')
        plt.ylabel('Y Secondary axis of travel')
        plt.title('Simulation of Covid-19 in a corridor')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Set up the animation
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=time, init_func=self.init_animation, blit=True, interval=50, repeat=False)
            
    def init_animation(self):
        """Initialize the animation"""
        # Initialize with empty positions
        all_positions = np.array([[person.position[0], person.position[1]] for person in self.map.all_people])
        self.scatter_all.set_offsets(all_positions)
        self.scatter_infected.set_offsets(np.empty((0, 2)))
        self.scatter_contagious.set_offsets(np.empty((0, 2)))
        self.lines_path_of_contagious.set_data([], [])
        return self.scatter_all, self.scatter_infected, self.scatter_contagious, self.lines_path_of_contagious
    
    def update(self, frame):
        """Update the simulation for each animation frame
        
        Args:
            frame: Frame number (required by FuncAnimation but not used in this function)
        """
        # Using frame parameter (commenting to avoid linter warnings)
        # frame is required by FuncAnimation but not used here
        # Move all people and check for new infections
        for i in range(len(self.map.all_people)):
            # Call move once per person instead of twice
            self.map.move(i)
            
        # Update the scatter plots
        all_positions = np.array([[person.position[0], person.position[1]] for person in self.map.all_people])
        infected_positions = np.array([[person.position[0], person.position[1]] for person in self.map.all_people if person.is_infected])
        contagious_positions = np.array([[person.position[0], person.position[1]] for person in self.map.all_people if person.contagious])
        
        self.scatter_all.set_offsets(all_positions)
        self.scatter_infected.set_offsets(infected_positions)
        self.scatter_contagious.set_offsets(contagious_positions)
        
        return self.scatter_all, self.scatter_infected, self.scatter_contagious, self.lines_path_of_contagious
          
        
    def run(self):
        """Run the simulation"""
        plt.show()

# Create and run the simulation
if __name__ == "__main__":
    sim = Simulation((-50, 50),0.1,1,1000)
    sim.run()
