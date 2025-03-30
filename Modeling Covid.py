import numpy as np
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 10000000
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.setrecursionlimit(1000000)  # Set the recursion limit to 10,000
import pandas as pd



class Map:
    def __init__(self, bounds, initial_people, false_positive_rate):
        """Initialises the map with a defined area, number of people, and false positive rate"""
        self.bounds = bounds #sets boundaries of the map note its from [0,0] to [bounds[0], bounds[1]]
        self.false_positive_rate = false_positive_rate #rate of intially infected within the population

        self.total_number_of_people = 0 # counts the total number of people simulated
        self.total_infected = 0 #counts the total number of infected people
        self.total_contagious = 0 #counts the total number of contagious people (to be contagious) you must be infected prior to arrival at the train station
        
        self.all_people = np.empty((initial_people), dtype=object) #stores the people objects currently in the simulation
        self.personal_space = 0 #sets the personal space (social distancing) of the people
        self.contagious_tracking = [[], []] #tracks the position of contagious people
        self.infection_probability_plus_2 = 0.013 #sets the range the infection can spread over
        self.infection_probability_sub_2 = 0.13 #given that you are in the range computes the probability of infection
        self.flows_stop = 0 #The number of times the flow of contagious people has to be stopped to maintain social distancing

        self.vac_num_mask_num_combined_num = None
        self.vacination_infection_para = None # this is the chance of getting infected if you are vaccinated [sub two meters,post two meters]
        self.mask_infection_para = None # this is the chance of getting infected if you are wearing a mask [sub two meters,post two meters]
        self.combined_infection_para = None # this is the chance of getting infected if you are vaccinated and wearing a mask [sub two meters,post two meters]
        
        
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
            self.total_number_of_people += 1
            self.total_number_of_people += 1
        person.direction = theta

        #new simple mask and vacine handling 
        if self.vac_num_mask_num_combined_num != None:
            if np.random.random() < self.vac_num_mask_num_combined_num:
                person.infection_probability_plus_2 = self.vacination_infection_para[1]
                person.infection_probability_sub_2 = self.vacination_infection_para[0]
            elif np.random.random() < self.vac_num_mask_num_combined_num:
                person.infection_probability_plus_2 = self.mask_infection_para[1]
                person.infection_probability_sub_2 = self.mask_infection_para[0]
            elif np.random.random() < self.vac_num_mask_num_combined_num:
                person.infection_probability_plus_2 = self.combined_infection_para[1]
                person.infection_probability_sub_2 = self.combined_infection_para[0]

        # Returns a peron object
        return person

    def move(self, index):
        """Moves a person in the simulation and checks for infection spread"""
        #Initialise the move variable to false
        move = False
        apc = self.all_people.copy() #creates a copy of the people in the simulation
        emergency_stop = 0
        emergency_stop = 0
        
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
            #If not contagious then we can just move them
            move = True
            move, apc = self.position_check(apc, self.personal_space, index, proposed_step)
            apc[index].position = proposed_step
           #if they are contagious then we need to check if they are going to infect anyone and now inforces social distancing
            #If not contagious then we can just move them
            move = True
            move, apc = self.position_check(apc, self.personal_space, index, proposed_step)
            apc[index].position = proposed_step
           #if they are contagious then we need to check if they are going to infect anyone and now inforces social distancing
            while move == False:
                proposed_step = apc[index].step(self.bounds, apc[index].step_size)
                if proposed_step is None:
                    #If required generates a new person
                    apc[index] = self.person_gen(self.false_positive_rate)
                else:
                    move, apc = self.position_check(apc, self.personal_space, index, proposed_step)
                    apc[index].position = proposed_step
                    if emergency_stop > 3:
                        print("waited", index)
                        self.flows_stop += 1
                        apc = self.all_people.copy() #resets the apc temporarily
                        break
                    emergency_stop += 1
        #returns the new position of the person with updated infection status if required
        self.all_people = apc
        return self.all_people

    def position_check(self, people_list, exclusion_zone, index, proposed_step):
        """Check if a proposed step is valid and update the list of people
           focusing on the social distancing and infection range"""
        # Check if self comparison
        for i in range(len(people_list)):
            if i == index:
                continue
            else:
                if people_list[index].contagious or people_list[i].contagious:
                    # computes the distance to all other people in the simulation
                    distance = np.linalg.norm(people_list[i].position - proposed_step)    
                    # checks if the distance is less than the infection range
                    if distance <= 4:
                        # checks if the person is contagious and if they are within the exclusion zone (social distancing)
                        if distance <= exclusion_zone:
                            # If they are contagious and within the exclusion zone then we cannot move
                            # If they are contagious and within the exclusion zone then we cannot move
                            return False, people_list
                        else: 
                            contagious = people_list[index]
                            healthy = people_list[i]
                            if contagious.direction == 0 and healthy.position[0]>contagious.position[0] or contagious.direction == np.pi and healthy.position[0]<contagious.position[0]:
                                if distance > 2:
                                    #checks if the distance is greater than 2
                                    if np.random.random() < healthy.infection_probability_plus_2:
                                        if people_list[i].is_infected == False:
                                            self.total_infected += 1
                                        people_list[i].is_infected = True
                                    #checks if the distance is less than 2
                                    elif np.random.random() < healthy.infection_probability_sub_2:
                                        if people_list[i].is_infected == False:
                                            self.total_infected += 1
                                        people_list[i].is_infected = True                
        return True, people_list
    
class Person:
    def __init__(self, position, contagious):
        """Initializes a person with a position, infection status, and step size"""
        self.position = position
        self.contagious = contagious
        self.step_size = np.random.uniform(1.0, 1.5) 
        self.direction = 0 #this property is updated by map 
        self.is_infected = contagious

        self.infection_probability_plus_2 = 0.013 #sets the sub 2 meters infection probability
        self.infection_probability_sub_2 = 0.13 #sets the post 2 meters infection probability

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
    def __init__(self, number_of_people=30):
        """Sets up the simulation with the map, number of people, and time"""
        self.time_per_simulation = 1000 #sets the time of each simulation
        self.contagious_rate = 0.3 #sets the rate of contagious people
        self.bounds = [5,5] #sets the bounds of the simulation where bounds are [x_max,y_max] in meters

        self.map = Map(self.bounds, number_of_people, self.contagious_rate)

        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(30, 8))
        self.ax.set_xlim(0, self.bounds[0])
        self.ax.set_ylim(0, self.bounds[1])

        # Create scatter plots for different states
        self.scatter_healthy = self.ax.scatter([], [], c='green', alpha=0.5, s=10, label='Healthy')
        self.scatter_infected = self.ax.scatter([], [], c='blue', alpha=0.8, s=15, label='Infected')
        self.scatter_contagious = self.ax.scatter([], [], c='red', alpha=1, s=25, label='Contagious')
        
        # Add text for statistics
        self.text_info = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        self.ax.legend(loc='upper right')
        self.ax.set_title('COVID-19 Spread Simulation')
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Position (m)')
        

    def animate(self, frame):
        print("frame", frame)
        print("frame", frame)
        """Animation function called for each frame"""
        # Move all people
        for i in range(len(self.map.all_people)):
            self.map.move(i)
    
        # Separate people by health status
        healthy_x, healthy_y = [], []
        infected_x, infected_y = [], []
        contagious_x, contagious_y = [], []
    
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
    
        # Update scatter plots
        self.scatter_healthy.set_offsets(np.c_[healthy_x, healthy_y])
        self.scatter_infected.set_offsets(np.c_[infected_x, infected_y])
        self.scatter_contagious.set_offsets(np.c_[contagious_x, contagious_y])
    
        # Update statistics
        total_people = self.map.total_number_of_people
        percent_infected = round((self.map.total_infected / self.map.total_number_of_people) * 100,2)
        total_people = self.map.total_number_of_people
        percent_infected = round((self.map.total_infected / self.map.total_number_of_people) * 100,2)
    
        stats_text = (f"Total people: {total_people}\n"
                      f"Total healthy: {total_people - self.map.total_infected}\n"
                      f"Total healthy: {total_people - self.map.total_infected}\n"
                      f"Total infected: {self.map.total_infected}\n"
                      f"Contagious: {self.map.total_contagious}\n"
                      f"Infection rate: {percent_infected}%")
        self.text_info.set_text(stats_text)
        return [self.scatter_healthy, self.scatter_infected, self.scatter_contagious, self.text_info]

                
    def run(self):
        """Run the simulation animation"""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.time_per_simulation,
            interval=1, blit=True, repeat=False
        )
        plt.show()
        return self.ani
    
    def save_animation(self, filename="covid_simulation.mp4", fps=20, dpi=100):
        """Save the animation as a video file
        
        Parameters:
        - filename: Output filename (default: covid_simulation.mp4)
        - fps: Frames per second (default: 20)
        - dpi: Resolution (dots per inch) (default: 100)
        """        
        # Create writer
        writer = animation.FFMpegWriter(fps=fps)
        
        print(f"Saving animation to {filename}...")
        self.ani.save(filename, writer=writer, dpi=dpi)
        print(f"Animation saved to {filename}")

class Comparison:
    def __init__(self, upper_bound_people = 30, upper_bound_social_distancing=4):
        """
        Initialize a comparison simulation with varying population sizes and social distancing values.
        
        Parameters:
        - bounds: Boundaries of the map
        - upper_bound_people: Maximum number of people to simulate
        - probably_infected: Initial infection rate
        - time_per_simulation: Time steps for each simulation
        - max_social_distancing: Maximum social distancing value to test
        """
        self.upper_bound_people = upper_bound_people #sets the upper bound of number of people 
        self.max_social_distancing = upper_bound_social_distancing #sets the upper bound for social distancing

        self.time_per_simulation = 100 #sets the time of each simulation
        self.contagious_rate = 0.3 #sets the rate of contagious people
        self.bounds = [5,5] #sets the bounds of the simulation where bounds are [x_max,y_max]

        self.results = []
        
    def run_comparison(self):
        """Run multiple simulations with varying numbers of people and social distancing values."""
        # Loop through different social distancing values
        for social_dist in np.arange(0.1, self.max_social_distancing + 0.1, 0.1):
            social_dist = round(social_dist, 1)  
            print(f"\nTesting social distancing: {social_dist}")
            # Loop through different numbers of people in the station
            for num_people in range(10, self.upper_bound_people + 1, 10):
                print(f"Running simulation with {num_people} people, social distancing: {social_dist}...")

                map = Map(self.bounds, num_people, self.contagious_rate)
                map.personal_space = social_dist  
                #the board move sequance
                for _ in range(self.time_per_simulation):
                    for i in range(len(map.all_people)):
                        map.move(i)

                # Calculate infection rate
                infection_rate = (map.total_infected / map.total_number_of_people) * 100

                #update studnent household manger to use this 'storage technique
                result = {
                    'social_distancing': social_dist,
                    'num_people': num_people,
                    'total_people': map.total_number_of_people,
                    'total_infected': map.total_infected,
                    'total_contagious': map.total_contagious,
                    'infection_rate': infection_rate,
                    'flows_stopped': map.flows_stop
                }

                #stores the results 
                self.results.append(result)
            
        return self.results
    

    def save_results_to_excel(self, filename="simulation_results.xlsx"):
        """Save the results to an Excel file with initial conditions on a separate sheet."""
        # Check if results are available
        if not self.results:
            print("No results to save. Run the comparison first.")
            return
        # Create a DataFrame with intial and input conditions
        results_df = pd.DataFrame(self.results)

        initial_conditions = {
            "Bounds": [self.bounds],
            "Upper Bound People": [self.upper_bound_people],
            "Contagious Rate": [self.contagious_rate],
            "Time Per Simulation": [self.time_per_simulation],
            "Max Social Distancing": [self.max_social_distancing]
        }
        initial_conditions_df = pd.DataFrame(initial_conditions)

        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f"simulation_results_{time}.xlsx"
        # Save to Excel file with two sheets
        with pd.ExcelWriter(filename) as writer:
            initial_conditions_df.to_excel(writer, sheet_name=name, index=False)
            results_df.to_excel(writer, sheet_name="Results", index=False)
        
        print(f"Results saved to {filename}")
    
    def plot_infection_rates(self):
        """Plot infection rates against number of people with trendlines for each social distancing value."""
        # Check if results are available
        if not self.results:
            print("No results to plot. Run the comparison first.")
            return

        results_df = pd.DataFrame(self.results)

        plt.figure(figsize=(12, 8))

        social_dist_values = sorted(results_df['social_distancing'].unique())
        # Create a color map for different social distancing values
        colors = plt.cm.viridis(np.linspace(0, 1, len(social_dist_values)))
        # Plot infection rates for each social distancing value
        for i, sd in enumerate(social_dist_values):
            
            sd_data = results_df[results_df['social_distancing'] == sd]

            plt.plot(sd_data['num_people'], sd_data['infection_rate'], 
                   color=colors[i], alpha=0.7, label=f'SD = {sd}m')

        plt.xlabel('Number of People')
        plt.ylabel('Infection Rate (%)')
        plt.title('Effect of Social Distancing and population size for a prevalence of: {contagious_rate}')
        plt.legend(title='Social Distancing', loc='best')
        plt.grid(True, alpha=0.3)

        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f"simulation_graph{time}.xlsx"
        plt.savefig(name, dpi=300, bbox_inches='tight')
        print("Plot saved as 'infection_rate_analysis.png'")

        plt.show()
    
    def run_full_comparison(self):
        """Run the full comparison, save results to an Excel file, and generate plots."""
        self.run_comparison()
        self.save_results_to_excel()
        self.plot_infection_rates()

if __name__ == "__main__":
    #Example usage
    run = Simulation()
    run.time_per_simulation = 1000 #sets the time of each simulation
    run.bounds = [240, 5] #sets the bounds of the simulation where bounds are [x_max,y_max] in meters
    run.contagious_rate = 0.01 #sets the rate of contagious people in the simulation

   
    run.map.vacination_infection_para = None# this is the chance of getting infected if you are vaccinated [sub two meters,post two meters]

    run.map.mask_infection_para = None # this is the chance of getting infected if you are wearing a mask [sub two meters,post two meters] 

    run.map.combined_infection_para = None# this is the chance of getting infected if you are vaccinated and wearing a mask [sub two meters,post two meters]
    
    run.map.bounds = (240, 5) #sets the bounds of the simulation where bounds are [x_max,y_max] in meters

    run.map.upper_bound_people = 20 #sets the upper bound of number of people in the simulation

    run.map.contagious_rate = 0.01 #sets the rate of contagious people in the simulation

    data = run.run()
    run.save_animation("covid_simulation.mp4", fps=20, dpi=100)
    #comparison = Comparison((240, 5) ,20, 3, 10)
    #comparison.run_full_comparison()