import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Map:
    def __init__(self, bounds, initial_people, false_positive_rate):
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
        self.total_number_of_people = 0
        self.false_positive_rate = false_positive_rate
        self.total_infected = 0
        self.total_contagious = 0
        self.all_people = np.empty((initial_people), dtype=object)
        self.personal_space = 2
        self.contagious_tracking = [[], []]
        self.infection_range = 5
        self.infection_probability = 0.3
        for i in range(initial_people):
            person = self.person_gen(false_positive_rate)
            print(f"Person {i}: Position: {person.position}, Contagious: {person.contagious}")
            self.all_people[i] = person

    def person_gen(self, false_positive_rate):
        """Generates people with random positions and step sizes"""
        if np.random.random() < 0.5:
            x = 0
            theta = 0
        else:
            x = self.bounds[0]
            theta = np.pi
    
        y = np.random.uniform(0,self.bounds[1])
        pos = np.array([x, y])

        if np.random.random() < false_positive_rate:
            # Create a person and add to the infected and contagious list
            person = Person(pos,True)
            self.total_infected += 1
            self.total_contagious += 1
            self.total_number_of_people += 1
        else:
            # Creates a healthy person 
            self.total_number_of_people += 1
            person = Person(pos,False)
        print("person direction", person.direction)
        person.direction = theta
        return person
        
      
    def move(self, index):
        # Initialize move validation
        move = False
        apc = self.all_people.copy()
        # Try to move the person
        proposed_step = apc[index].step(self.bounds, apc[index].step_size)
        if proposed_step is None:
            apc[index] = self.person_gen(self.false_positive_rate)
            self.all_people = apc
            return self.all_people
        else:
            if apc[index].contagious == False:
                move = True
            move, apc = self.position_check(apc, self.personal_space, self.infection_range, index, proposed_step)
            # Create a new person (non-contagious by default to balane
            while move == False:
                proposed_step = apc[index].step(self.bounds, apc[index].step_size)
                move, apc = self.position_check(apc, self.personal_space, self.infection_range, index, proposed_step)
        apc[index].position = proposed_step
        self.all_people = apc
        return self.all_people

        
    def position_check(self, people_list, exclusion_zone, infection_range, index, proposed_step):
        """Check if a proposed step is valid and update the list of people
           focusing on the social distancing and infection range"""
        
        # Check if the proposed step is within the bounds
        #also point to add in social distancing

        for i in range(len(people_list)):
            #checks if comparing to self
            if i == index:
                continue
            else:
                # Then gives a temp asign incase its not a vailed position with socal 
                distance = np.linalg.norm(people_list[i].position - proposed_step)
                if distance <= infection_range:
                    if self.people_list[index].contagious == True:
                        # Check if the distance is less than the exclusion zone
                        if distance <= exclusion_zone:
                            return False, people_list
                        else:
                            if np.random.random() <self.infection_probability:
                                people_list[i].is_infected = True
                                self.total_infected += 1
                                # Update the total number of people

                else:
                    pass

        return True, people_list
    
class Person:
    def __init__(self, position, contagious):
        """Initializes a person with a position, infection status, and step size"""
        step_size = np.random.uniform(1.0, 1.5)
        self.position = position
        self.contagious = contagious
        self.step_size = step_size
        self.direction = 0
        self.is_infected = contagious
        self.position_history = [[], []]

    def step(self, bounds, step_size):
        """Performs a single random walk step and impose boundary conditions on a rectangular room"""
        std = 0.1 # The effective woobe of people (random walk would be 2 * np.pi) 
        angle = np.random.normal(0,std) + self.direction
        new_x = self.position[0] + step_size * np.cos(angle)
        new_y = self.position[1] + step_size * np.sin(angle)
        
        # Check if the person would go out of bounds
        if new_x < 0 or new_x > bounds[0] or new_y < 0 or new_y > bounds[1]:
            # Person is leaving the area - signal this with None
            position = None
        else:
            # Person stays in bounds
            position = np.array([new_x, new_y])
        return position
    


class Simulation:
    def __init__(self, bounds, number_of_people, probably_infected=0.01, time=10):
        self.map = Map(bounds, number_of_people, probably_infected)
        self.time = time
        self.count_of_people  = number_of_people
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

        for i in range(len(self.map.all_people)):
            self.map.move(i)
        
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
            interval=800, blit=False, repeat=False
        )
        plt.show()
        return self.ani



if __name__ == "__main__":
    # Run simulation with a 50x50 meter area
    bounds = [150, 5]
    sim = Simulation(bounds, number_of_people= 10, probably_infected=0.3)
    ani = sim.run()
