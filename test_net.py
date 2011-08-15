execfile("__init__.py")
#import pyrecurrentnet
import numpy as np
import time

net = RNN(2, 5, 2)

W_shape = net.W.shape
net.W = 2.0 * np.random.rand(*W_shape) -1.0

b_shape = net.bias.shape
net.bias = 2.0 * np.random.rand(*b_shape) - 1.0
net.randomiseState()
net_state = net.x

steps = 100000
x = np.random.rand(net.input_size)
all_outs = net.evolve(x,steps)
print "With evolve: ", all_outs

net.set_x(net_state)
for _ in xrange(steps):
    all_outs = net(x)

print "without evolve: ", all_outs

print "\nTrying the attractors"
for _ in xrange(10):
    x = np.random.rand(net.input_size)
    net.randomiseState()
    all_outs = net.evolve(x,steps)
    print "Net settles in ", all_outs

print "Measuring time"

x = np.random.rand(net.input_size)
net.randomiseState()
now = time.time()
for _ in xrange(steps):
    net(x)
print "Time without evolve: ", time.time() - now

x = np.random.rand(net.input_size)
net.randomiseState()
now = time.time()
net.evolve(x, steps)
print "Time with evolve: ", time.time() - now


