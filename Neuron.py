import random
from Value import Value


class Neuron():

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        o = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        o = o.tanh()
        return o

    def parameters(self):
        return self.w + [self.b,]

class Layer():

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP():

    def __init__(self, nin, nouts):
        s = [nin, ] + nouts 
        self.layers = [Layer(s[i], s[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for Layer in self.layers for p in Layer.parameters()]
    




