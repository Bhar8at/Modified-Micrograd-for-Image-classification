from Neuron import MLP
from onehotencoding import x_train, y_train, x_test, y_test
# Initializing MLP
n = MLP(784, [10,1])


# Training Values
x =  x_train.tolist()
ypred = y_train.tolist()






for i in range(200):

    # forward pass
    yout = [n(x[i]) for i in range(len(x))]
    loss = sum([(yo - yp)**2 for yo, yp in zip(yout,ypred)])

    # backward pass 
    for p in n.parameters():
        p.grad = 0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05*p.grad
        
        
    print("Step ",i,loss.data,)