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
        std = 0.5 # The effective woobe of people (random walk would be 2 * np.pi) 

        angle = np.random.normal(0,std) + self.direction
        # Remove print statement to avoid console flooding
        new_x = self.position[0] + step_size * np.cos(angle)
        new_y = self.position[1] + step_size * np.sin(angle)
        
        # Apply boundary conditions

        new_x = np.clip(new_x, 0, bounds[0])
        new_y = np.clip(new_y, 0, bounds[1])
            
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
        self.num_people = 10
        self.no_infected = 0
        self.step_size = step_size
        self.all_people = []
        self.personal_space = 2
        self.contagious_tracking = [[], []]
        self.infection_range = 10

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
        self.all_people[index].position = proposed_step
        self.all_people[index].position_history[0].append(proposed_step[0])
        self.all_people[index].position_history[1].append(proposed_step[1])
        self.all_people = proposed_updated_people
        return self.all_people

        
    def position_check(self, proposed_updated_people, exclusion_zone, infection_range, index, proposed_step):
        """Check if a proposed step is valid and update the list of people
           focusing on the social distancing and infection range"""
        
        # Check if the proposed step is within the bounds
        #also point to add in social distancing

        for i in range(len(proposed_updated_people)):
            #checks if comparing to self
            if i == index:
                pass

            # Checks social distancing
            elif np.linalg.norm(proposed_updated_people[i].position - proposed_step) <= exclusion_zone:
                return False, proposed_updated_people
            
            # Check if the proposed step infects another person
            # Then gives a temp asign incase its not a vailed position with socal distancing
            elif np.linalg.norm(proposed_updated_people[i].position - proposed_step) <= infection_range:
                proposed_updated_people[i].is_infected = True
        return True, proposed_updated_people
    

class Simulation:
    def __init__(self, bounds, people_density, step_size=1, probably_infected=0.01, time=1000):
        self.map = Map(bounds, people_density, step_size, probably_infected)
        self.time = time
        self.ani = None  # Store animation reference
        
        # Create figure and visualization elements
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, bounds[0])
        self.ax.set_ylim(0, bounds[1])
        
        # Initialize empty plots
        self.scatter_healthy = self.ax.scatter([], [], c='green', alpha=0.5, s=10, label='Healthy')
        self.scatter_infected = self.ax.scatter([], [], c='blue', alpha=0.8, s=15, label='Infected')
        self.scatter_contagious = self.ax.scatter([], [], c='red', alpha=1, s=25, label='Contagious')
        self.lines_path_of_contagious, = self.ax.plot([], [], c='orange', alpha=0.3, linewidth=1, label='Path of Contagious')
        
        # Add legend
        self.ax.legend(loc='upper right')
        self.ax.set_title('COVID-19 Spread Simulation')
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Position (m)')

    def animate(self, frame):
        """Animation function called for each frame"""
        # Move all people for this frame
        all_people = []

        #for i in range(len(self.map.all_people)):
        self.map.move(frame)
        
        # Separate people by their status
        healthy_x, healthy_y = [], []
        infected_x, infected_y = [], []
        contagious_x, contagious_y = [], []
        
        # Track paths of contagious people
        path_x, path_y = [], []
        
        for person in self.map.all_people:
            if person.contagious:
                contagious_x.append(person.position[0])
                contagious_y.append(person.position[1])
                # Add path history for contagious people
                path_x.extend(person.position_history[0])
                path_y.extend(person.position_history[1])
                path_x.append(None)  # Add None to separate different people's paths
                path_y.append(None)
            elif person.is_infected:
                infected_x.append(person.position[0])
                infected_y.append(person.position[1])
            else:
                healthy_x.append(person.position[0])
                healthy_y.append(person.position[1])
        
        # Update scatter plots
        self.scatter_healthy.set_offsets(np.column_stack((healthy_x, healthy_y)) if healthy_x else np.empty((0, 2)))
        self.scatter_infected.set_offsets(np.column_stack((infected_x, infected_y)) if infected_x else np.empty((0, 2)))
        self.scatter_contagious.set_offsets(np.column_stack((contagious_x, contagious_y)) if contagious_x else np.empty((0, 2)))
        
        # Update path lines
        self.lines_path_of_contagious.set_data(path_x, path_y)
        
        # Add info text with current statistics
        info_text = f"Frame: {frame}\nHealthy: {len(healthy_x)}\nInfected: {len(infected_x)}\nContagious: {len(contagious_x)}"
        if hasattr(self, 'text_info'):
            self.text_info.set_text(info_text)
        else:
            self.text_info = self.ax.text(0.02, 0.95, info_text, transform=self.ax.transAxes, 
                                         bbox=dict(facecolor='white', alpha=0.7))
        
        return [self.scatter_healthy, self.scatter_infected, self.scatter_contagious, 
                self.lines_path_of_contagious, self.text_info]
    def run(self):
        """Run the simulation animation"""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.time,
            interval=500, blit=False, repeat=False
        )
        plt.show()
        return self.ani



if __name__ == "__main__":
    # Run simulation with a 50x50 meter area
    bounds = [100, 5]
    sim = Simulation(bounds, number_of_people= 10, step_size=1.2, probably_infected=0.)
    ani = sim.run()
