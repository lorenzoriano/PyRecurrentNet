__all__ = ["RNN"]
import libcrnn
import numpy as np

class RNN(libcrnn.RNN):
    def __call__(self, inpt):
        inpt = np.array(inpt, dtype = np.float64).ravel()
        return libcrnn.RNN.__call__(self, inpt)
     
    x = property(libcrnn.RNN.get_x, libcrnn.RNN.set_x)
    bias = property(libcrnn.RNN.get_bias, libcrnn.RNN.set_bias)
    W = property(libcrnn.RNN.get_W, libcrnn.RNN.set_W)
    input_size = property(libcrnn.RNN.input_size)
    output_size = property(libcrnn.RNN.output_size)
    hidden_size = property(libcrnn.RNN.hidden_size)
    size = property(libcrnn.RNN.size)
    
    def __str__(self):
        ret = "Size: " + str( self.size)
        ret += " input_size: "  +str(self.input_size)
        ret += " output_size: " +str(self.output_size)
        ret += "\n"
        ret += "W = \n" + str(self.W) + "\nbias = \n" + str(self.bias)
        return ret

def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    net = RNN(hidden_size, input_size, output_size)
    array_W = np.array(chromosome.genomeList[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(chromosome.genomeList[len(chromosome.genomeList) - bias_size:] )
    
    W = np.zeros((net.size, net.size))
    bias = np.zeros(net.size)

    W[net.input_size:, :] = array_W
    bias[net.input_size:] = array_bias
    
    net.W = W
    net.bias = bias
        
    return net

def list_convert(params, input_size, output_size, hidden_size):
    bias_size = hidden_size + output_size
    
    net = RNN(hidden_size, input_size, output_size)
    array_W = np.array(params[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(params[len(params) - bias_size:] )
    
    W = np.zeros((net.size, net.size))
    bias = np.zeros(net.size)
    
    W[net.input_size:, :] = array_W
    bias[net.input_size:] = array_bias
    
    net.W = W
    net.bias = bias
    
    return net