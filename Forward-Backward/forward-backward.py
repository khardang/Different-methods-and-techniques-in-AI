import numpy as np

#Start_probability is for if it rain or not today given the last day
#Transition is the probability for rain or sun, given umberella.
rain = [True, False]  #different state it can be, in this case it is rain or not rain.
start_prob = np.array([0.5, 0.5])     #Start probability P(X1) = 0.5
transition_prob = np.array([[0.7, 0.3],[0.3, 0.7]]) #Probability matrix for if the day before was rain or not
umbrella = np.array([0.9, 0.2]) #Probability for carrying an umbrella if it is rain or sun
no_umbrella = np.array([0.1, 0.8]) #probability for NOT carrying an umbrella if it is rain or sun
observation = np.array([True,True,False,True,True]) #Observations of carrying umbrella. True = umbrella, False = No umbrella


def forward(observation):

    forward_list=np.ones((len(observation)+1,len(rain)))
    rain_given_observation=start_prob
    forward_list[0]=rain_given_observation
    t=0
    for i in observation:
        t += 1
        rain_given_observation = np.dot(np.transpose(transition_prob),rain_given_observation)
        if i: #if he carrying an umbrella
            rain_given_observation = (rain_given_observation * umbrella)
            rain_given_observation = np.divide(rain_given_observation, rain_given_observation.sum()) #Normalization


        else: #if he is not carrying an umbrella
            rain_given_observation = (rain_given_observation * no_umbrella)
            rain_given_observation = np.divide(rain_given_observation, rain_given_observation.sum())  #Normalization
        forward_list[t]= rain_given_observation
    print("forward_list=",forward_list)
    return forward_list

def backward(observation):
    backward_list = np.ones((len(observation)+1,len(rain)))
    rain_given_observation=np.ones(len(rain))
    t=len(observation)
    reverse = reversed(observation) #reverse the list of observation, uposit of forward.
    for i in reverse:
        t -= 1
        if i: # if carrying umbrella
           rain_given_observation = umbrella * rain_given_observation
        else: #if not carrying umbrella
            rain_given_observation = no_umbrella * rain_given_observation
        rain_given_observation = np.dot(transition_prob, rain_given_observation)
        backward_list[t] = rain_given_observation
    print("backward_list=",backward_list)
    return backward_list


def forward_backward(observation):
    forward_list=forward(observation)
    backward_list=backward(observation)
    forward_backward_list = []
    combine = np.multiply(forward_list,backward_list) # this combines both forward and backward list and is an unnormalized list of forward_backward

    for i in range(len(forward_list)):
        value = combine[i]/combine[i].sum()  #Normalize
        forward_backward_list.append(value)
    print("forward_backward_list=",forward_backward_list)
    return forward_backward_list

forward_backward(observation)
