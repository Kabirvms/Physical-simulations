import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Map:
    def __init__(self, bounds, initial_people, false_positive_rate):
        """Initialises the map with a defined area, number of people, and false positive rate"""
        self.bounds = bounds #sets boundaries of the map note its from [0,0] to [bounds[0], bounds[1]]
        self.total_number_of_people = 0 # counts the total number of people simulated
        self.false_positive_rate = false_positive_rate #rate of intially infected within the population
        self.total_infected = 0 #counts the total number of infected people
        self.total_contagious = 0 #counts the total number of contagious people (to be contagious) you must be infected prior to arrival at the train station
        self.all_people = np.empty((initial_people), dtype=object) #stores the people objects currently in the simulation
        self.personal_space = 2 #sets the personal space (social distancing) of the people
        self.contagious_tracking = [[], []] #tracks the position of contagious people
        self.infection_range = 5 #sets the range the infection can spread over
        self.infection_probability = 0.3 #given that you are in the range computes the probability of infection

        ## Generate initial people
        for i in range(initial_people):
            person = self.person_gen(false_positive_rate)
            self.all_people[i] = person

    def person_gen(self, false_positive_rate):
        """Generates people with random entry positions and step sizes"""
        # randomly chooses the side and point of entry
        if np.random.random() < 0.5:
            x = 0
            theta = 0
        else:
            x = self.bounds[0]
            theta = np.pi
    
        y = np.random.uniform(0,self.bounds[1])
        pos = np.array([x, y])

        # Randomly assigns contagious status based on the false positive rate
        if np.random.random() < false_positive_rate:
            person = Person(pos,True)
            self.total_infected += 1
            self.total_contagious += 1
            self.total_number_of_people += 1
        else:
            person = Person(pos,False)
        person.direction = theta

        # Returns a peron object
        return person

    def move(self, index):
        """Moves a person in the simulation and checks for infection spread"""

        #Initialise the move variable to false
        move = False
        apc = self.all_people.copy() #creates a copy of the people in the simulation
        
        #creates a proposed step for the person at index i in current people (all_people)
        proposed_step = apc[index].step(self.bounds, apc[index].step_size)
        #checkes if the proposed step made the person exit the bounds of the map.
        if proposed_step is None:
            #If required generates a new person
            apc[index] = self.person_gen(self.false_positive_rate)
            self.all_people = apc
            return self.all_people
        else:
            #Checks if they are contagious (to determine if more computations are required)
            if apc[index].contagious == False:
                #If not contagious then we can just move them
                move = True
                move, apc = self.position_check(apc, self.personal_space, self.infection_range, index, proposed_step)
            #if they are contagious then we need to check if they are going to infect anyone and now inforces social distancing
            while move == False:
                proposed_step = apc[index].step(self.bounds, apc[index].step_size)
                move, apc = self.position_check(apc, self.personal_space, self.infection_range, index, proposed_step)
        #returns the new position of the person with updated infection status if required
        apc[index].position = proposed_step
        self.all_people = apc
        return self.all_people

    def position_check(self, people_list, exclusion_zone, infection_range, index, proposed_step):
        """Check if a proposed step is valid and update the list of people
           focusing on the social distancing and infection range"""
        # Check if self comparison
        for i in range(len(people_list)):
            if i == index:
                continue
            else:
                # computes the distance to all other people in the simulation
                distance = np.linalg.norm(people_list[i].position - proposed_step)
                # checks if the distance is less than the infection range
                if distance <= infection_range:
                    # checks if the person is contagious and if they are within the exclusion zone (social distancing)
                    if people_list[index].contagious == True:
                        # If they are within social distancing the move is not valid assumption strickt social distancing
                        if distance <= exclusion_zone:
                            return False, people_list
                        else:
                            # If they are not within social distancing then we can infect them
                            if np.random.random() <self.infection_probability:
                                people_list[i].is_infected = True
                                self.total_infected += 1

                else:
                    pass

        return True, people_list
    
class Person:
    def __init__(self, position, contagious):
        """Initializes a person with a position, infection status, and step size"""
        step_size = np.random.uniform(1.0, 1.5) #generates random step size
        self.position = position
        self.contagious = contagious
        self.step_size = step_size
        self.direction = 0 #this property is updated by map 
        self.is_infected = contagious
        self.position_history = [[], []]

    def step(self, bounds, step_size):
        """Performs a single random walk step and impose boundary conditions on a rectangular room"""
        std = 0.1 # determinses the "peudo random walk" of the person
        angle = np.random.normal(0,std) + self.direction
        new_x = self.position[0] + step_size * np.cos(angle)
        new_y = self.position[1] + step_size * np.sin(angle)

        #checks if the move is valid
        if new_x < 0 or new_x > bounds[0] or new_y < 0 or new_y > bounds[1]:
            
            position = None
        else:
            
            position = np.array([new_x, new_y])
        return position

class Simulation:
    def __init__(self, bounds, number_of_people, probably_infected=0.01, time=100):
        """Sets up the simulation with the map, number of people, and time"""
        self.map = Map(bounds, number_of_people, probably_infected)
        self.time = time

        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, bounds[0])
        self.ax.set_ylim(0, bounds[1])

        # Create scatter plots for different states
        self.scatter_healthy = self.ax.scatter([], [], c='green', alpha=0.5, s=10, label='Healthy')
        self.scatter_infected = self.ax.scatter([], [], c='blue', alpha=0.8, s=15, label='Infected')
        self.scatter_contagious = self.ax.scatter([], [], c='red', alpha=1, s=25, label='Contagious')
        self.lines_path_of_contagious, = self.ax.plot([], [], c='orange', alpha=0.3, linewidth=1, label='Path of Contagious')
        
        # Add text for statistics
        self.text_info = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        self.ax.legend(loc='upper right')
        self.ax.set_title('COVID-19 Spread Simulation')
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Position (m)')
        
        self.path_x_con, self.path_y_con = [], []

    def animate(self, frame):
        """Animation function called for each frame"""
        # Move all people
        for i in range(len(self.map.all_people)):
            self.map.move(i)
    
        # Separate people by health status
        healthy_x, healthy_y = [], []
        infected_x, infected_y = [], []
        contagious_x, contagious_y = [], []
        path_x_con, path_y_con = [], []
    
        # Collect positions by health status
        for person in self.map.all_people:
            if not person.is_infected:
                healthy_x.append(person.position[0])
                healthy_y.append(person.position[1])
            else:
                infected_x.append(person.position[0])
                infected_y.append(person.position[1])
                if person.contagious:
                    contagious_x.append(person.position[0])
                    contagious_y.append(person.position[1])
                    path_x_con.extend(person.position_history[0])
                    path_y_con.extend(person.position_history[1])
    
        # Update scatter plots
        self.scatter_healthy.set_offsets(np.c_[healthy_x, healthy_y])
        self.scatter_infected.set_offsets(np.c_[infected_x, infected_y])
        self.scatter_contagious.set_offsets(np.c_[contagious_x, contagious_y])
    
        # Update contagious path
        self.lines_path_of_contagious.set_data(self.path_x_con, self.path_y_con)
    
        # Update statistics
        total_people = len(self.map.all_people)
        percent_infected = (self.map.total_infected / self.map.total_number_of_people) * 100 if self.map.total_number_of_people > 0 else 0
    
        stats_text = (f"Total people: {total_people}\n"
                      f"Total infected: {self.map.total_infected}\n"
                      f"Contagious: {self.map.total_contagious}\n"
                      f"Infection rate: {percent_infected:.1f}%")
        self.text_info.set_text(stats_text)
    
        return [self.scatter_healthy, self.scatter_infected, self.scatter_contagious, 
            self.lines_path_of_contagious, self.text_info]

                
    def run(self):
        """Run the simulation animation"""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.time,
            interval=50, blit=True, repeat=False
        )
        plt.show()
        return self.ani

if __name__ == "__main__":
    bounds = [150, 100]  # Increased y-bound for better visualization
    sim = Simulation(bounds, number_of_people=20, probably_infected=0.3, time=200)
    ani = sim.run()
