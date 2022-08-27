import pandas as pd
import matplotlib.pyplot as plt

# dataset consists of a csv containing hours of study and scores of students. 
data = pd.read_csv('data.csv')

# gradient descent algorithm 
# parameters: actual m and b, sample of data and learning rate of algorithm. 
def gradient_descent(m_now, b_now, points, learning_rate):

    # partial derivatives initialized on 0. 
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        # get values of each sample.
        x = points.iloc[i].study 
        y = points.iloc[i].score

        # partial derivative to e respect to m. 
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))

        # partial derivative to e respect to b. 
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    # m equals current m minus partial derivative of e respect to m or m_gradient multiplied by the learning rate.
    m = m_now - m_gradient * learning_rate

    # b equals current b minus partial derivative of e respect to b or b_gradient multiplied by the learning rate.
    b = b_now - b_gradient * learning_rate

    return m, b

m = 0
b = 0
learning_rate = 0.0001 # define a  slow learning rate .
epochs = 300 # define # of epochs.

for i in range(epochs):
    if i % 50 == 0: # just to visualize the road of the epochs.
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learning_rate) # calculate the gradient descent for m,b and passing the samples and the learning rate.

# print final m and b
print("m: ",m," b:", b)

# scatter the samples and the returned function
plt.scatter(data.study, data.score)
plt.plot(list(range(20,80)), [m * x + b for x in range (20,80)], color="red")
plt.legend(['fitted function', 'samples'])
plt.show()

