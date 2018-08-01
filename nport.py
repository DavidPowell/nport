#!/usr/bin/env python

"""
Classes for dealing with N-port linear network parameters

The definitions for these parameters are mostly taken from Pozar
"""

from __future__ import division
import os.path
import re
from numpy import pi, dot
from numpy.linalg import solve, inv
import numpy as np

# QUERY: should Z0 default to None?

# TODO: allow impedance to vary per port and to have dispersion

def offset_key_element(element):
    "An internal routine which converts intuitive indices (e.g. S[1,1]) into python zero-based indices"
    
    if type(element) == slice:
        if element.start is None:
            start = None
        else:
            start = element.start-1
            assert(start >= 0)
    
        if element.stop is None:
            stop = None
        else:
            stop = element.stop-1
            assert (stop >= 0)
          
        return slice(start, stop, element.step)
    elif type(element) == int:
        assert(element > 0)
        return element-1
    else:
        raise TypeError("Unknown key type")

class network_params(object):
    """
    An abstract base class for N-port network parameters.  
    
    All network types include the N-port data, and the frequency information
  
    They can be accessed in the following ways:
    p[2,1] - parameter 2,1 over all frequencies
    p[f]   - all parameters at frequency f
    p[f,2,1] - parameter 2,1 at frequency f
    """
  
    def __init__(self, f, data = None, ports = 0):
        """
        f - the frequency range of the data
        data - the data to store
        ports - the number of ports
        """
        
        if data is None:
            # we are creating an empty set of parameters, 
            # so we should allocate them as zeros
            if ports <= 0:
                raise ValueError(
                "If data is not specified, then ports must be specified")
            
            self.__data = np.zeros((len(f), ports, ports), dtype=complex)
            self.ports = ports
        else:
            # we are using already existing data
            if (not data.shape[0] == len(f)):
                raise ValueError(
                "N-port data length does not match frequency length")
    
            if len(data.shape) == 1:
                # we have a 1D array, in which case create 1-port data
                self.__data = np.zeros((len(data), 1, 1))
                self.__data[:, 0, 0] = data
                self.ports = 1
            else:   
                if len(data.shape) != 3:
                    raise ValueError(
                    "N-port data must be either 1 or 3-dimensional")
                elif data.shape[1] != data.shape[2]:
                    raise ValueError("N-port data is not square")
                self.__data = data
        
                self.ports = data.shape[2]
            
        self.f = f

        # parameters required to reconstruct, apart from data
        self.init_params = {'f' : f}
    


    @staticmethod
    def interpret(key, shape):
        """interpret the key used to get or set network parameters"""
    
        #print key
        
        if type(key) == int or type(key) == slice:
            # a single index or range should be interpreted as a frequency
            return (key, slice(None), slice(None))
        elif type(key) == tuple:
            key = list(key)
            # a tuple, work out whether it has 2 or 3 indices
            if len(key) == 2:
                return (slice(None), offset_key_element(key[0]), offset_key_element(key[1]))
            elif len(key) == 3:
                return (key[0], offset_key_element(key[1]), offset_key_element(key[2]))
             
        raise KeyError("Unknown key "+repr(key))

    
    def __getitem__(self, key):
        """
        Returns the slice of the N-port parameters which the user asks for.
        Converts from the engineering format e.g. S[2, 1] to the internal
        zero based format of numpy arrays
        """
        return self.__data.__getitem__(
            network_params.interpret(key, self.__data.shape))
    
    def __setitem__(self, key, data):
        self.__data.__setitem__(
            network_params.interpret(key, self.__data.shape), data)
  
    def __len__(self):
        """Returns the number of frequencies contained in these parameters"""
        return len(self.f)
  
    @property
    def shape(self):
        return self.__data.shape
  
    def same_freq(self, other):
        """
        Check if two sets of network parameters operate on the same
        frequency data
        """
        if self.f is other.f:
            return True
        elif len(self.f) != len(other.f):
            return False
        else:
            return all(self.f == other.f)
            
    def select_ports(self, ports):
        """
        Selects only the parameters corresponding to the given list of ports
        
        Returns a new set of parameters of the same type
        Note that the effect of 'dropping' other ports depends on the type
        of parameters - for S parameters they are terminated in Z0, for
        Y parameters they are short circuit, for Z parameters open circuit
        
        Can also be used to:
            renumber ports, by specifying all ports in a new order
            add new ports, by specifying zeros in the list of new ports
            (new ports will have zero corresponding parameters)
        
        """
        
        #if len(ports) > self.ports:
        #    raise ValueError, "Cannot select more ports than original data"
            
        setup_params = {}
        setup_params.update(self.init_params)
        setup_params['ports'] = len(ports)
            
        new_params = self.__class__(**setup_params)
        
        for outer_new, outer_orig in enumerate(ports):
            for inner_new, inner_orig in enumerate(ports):
                if inner_orig and outer_orig:
                    new_params[outer_new+1, inner_new+1] = self[outer_orig, inner_orig]
                else:
                    new_params[outer_new+1, inner_new+1] = 0.0
                
        return new_params

class wave_params(network_params):
    """
    A common base class for S and T parameters which holds information about the
    port impedances
    
    Z0 - can be None to indicate that reference impedance is not known, implying
    that conversion to non wave parameters should be prohibited
    """
    def __init__(self, f, data = None, ports = 0, Z0 = None):
        network_params.__init__(self, f, data, ports)
        #self.set_reference_impedance(Z0)
    
        self.Z0 = Z0
        self.init_params['Z0'] = self.Z0
    

class port_impedance(object):
    "Stores a port impedance which may depend on frequency and port number"
    def __init__(self, Z0):
        """Set the reference impedance for the structure, and try to guess
        whether it varies with frequency, port number or both.
        
        Note that this assumes that impedance is not set or is invalid, as 
        it will not convert to another reference impedance
        """
        
        if type(Z0) == type(1) or type(Z0) == type(1.0):
            # a simple scalar
            self.Z0 = np.array(Z0)
        elif isinstance(Z0, np.ndarray):
            # a multi-dimensional array
            pass
        else:
            raise AttributeError("Z0 is of unknown type %s", str(type(Z0)))
    

class S_params(wave_params):
    """
    A class to hold scattering parameters, as well as the associated frequency
    and impedance data.
  
    Parameters for a given frequency can be accessed as
  
    S[f]
  
    where f is the index and S is an sparams object.  For a particular
    parameter at all frequencies, use:
  
    S[m, n]
  
    where m and n are currently restricted to being either 1 or 2
  
    |b[1]|   | S[1,1]  S[1,2] | |a[1]|
    |    | = |                | |    |
    |b[2]|   | S[2,1]  S[2,2] | |a[2]|

    """
    def __init__(self, f, data = None, ports = 0, Z0 = 50.0):
        wave_params.__init__(self, f, data, ports, Z0)
  
    def writesnp(self, filename):
        """
        Write the network parameters to a Touchstone .snp file
        """
        if self.shape[2] != 2:
            raise Exception("Can only write s2p")
        
        with open(filename, "wt") as output_file:
            output_file.write("# HZ S RI R 50\n")
            
            for count in range(len(self)):
                S = self[count]
                output_file.write(
                    str(self.f[count])+" "+str(S[0, 0].real)+" "
                    +str(S[0, 0].imag)+" "+str(S[0, 1].real)+" "
                    +str(S[0, 1].imag)+" "+str(S[1, 0].real)+" "
                    +str(S[1, 0].imag)+" "+str(S[1, 1].real)+" "
                    +str(S[1, 1].imag)+"\n")
              
    
    def T_params(self, reverse = False):
        """
        Calculate the T (scattering transfer) parameters 
        
        Parameters
        ----------
        reverse : boolean, optional
            if True, will calculate the transfer
            parameters for the reverse orientation of this network, which can be
            useful for networks such as amplifiers which are essentially unilateral
        
        Returns
        -------
        T : T_params
            The scattering transfer parameters
        
        For more than 2 ports, assumes that the first n ports are input, 
        the second n ports are output
        """
        S = self

        T = T_params(f=self.f, Z0=self.Z0, ports=self.ports)

        if self.ports == 2:
            #raise Exception, "T parameters are only valid for 2-port networks"
            
            #T = T_params(f=self.f, Z0=self.Z0)
            
            if reverse:
                T[1,1] = -(S[2,2]*S[1,1] - S[2,1]*S[1,2])/S[1,2]
                T[1,2] = S[2,2]/S[1,2]
                T[2,1] = -S[1,1]/S[1,2]
                T[2,2] = 1.0/S[1,2]
            else:
                T[1,1] = -(S[1,1]*S[2,2] - S[1,2]*S[2,1])/S[2,1]
                T[1,2] = S[1,1]/S[2,1]
                T[2,1] = -S[2,2]/S[2,1]
                T[2,2] = 1.0/S[2,1]
            
        else:
            if self.ports %2 != 0:
                raise ValueError("Must have an even number of ports for scattering transfer parameters")
                
            if reverse:
                raise NotImplementedError                
                
            n = self.ports//2
            #import numpy.linalg as la

#            M1 = np.zeros((self.ports, self.ports), np.complex128)
#            M2 = np.zeros((self.ports, self.ports), np.complex128)
#
#
#            M1[n:2*n, 0:n] = np.eye(n)
#            M2[0:n, n:2*n] = np.eye(n)

#           T_data = np.empty((len(self.f), self.ports, self.ports), np.complex128)

            for count in range(self.shape[0]):
                # create sub-matrices
                S11_mat = S[count, 1:n+1, 1:n+1]
                S12_mat = S[count, 1:n+1, n+1:]
                S21_mat = S[count, n+1:, 1:n+1]
                S22_mat = S[count, n+1:, n+1:]
                
                S21_inv_S22 = solve(S21_mat, S22_mat)
                S21_inv = inv(S21_mat)
                
                T[count, 1:n+1, 1:n+1] = S12_mat - np.dot(S11_mat, S21_inv_S22)
                T[count, 1:n+1, n+1:] = np.dot(S21_inv, S11_mat)
                T[count, n+1:, 1:n+1] = -S21_inv_S22
                T[count, n+1:, n+1:] = S21_inv

               #M1[:n, :] = S[count, 1:n+1, :]
                #M2[n:2*n, :] = S[count, n+1:2*n+1, :]
                
                #T[count] = solve(M2.T, M1.T).T
                
                ## define the vector S-parameters
                #S_11 = S[count, 1:n+1,1:n+1]
                #S_12 = S[count, 1:n+1,n+1:2*n+1]
                #S_21 = S[count, n+1:2*n+1,1:n+1]
                #S_22 = S[count, n+1:2*n+1,n+1:2*n+1]
                    #
                #T[count, 1:n+1, 1:n+1] = S_12 - np.dot(S_11, la.solve(S_21, S_22))
                #T[count, 1:n+1, n+1:2*n+1] = np.dot(S_11, la.inv(S_21))
                #T[count, n+1:2*n+1, 1:n+1] = -la.solve(S_21, S_22)
                #T[count, n+1:2*n+1, n+1:2*n+1] = la.inv(S_21)
            
            

        return T


    def S_params(self, Z0 = None):
        if Z0 != self.Z0 and Z0 != None:
            raise NotImplementedError("Change impedance via converting to a different type of parameters")
        else:
            return self
      
    def ABCD_params(self, Z0 = None):
        """
        Calculate the ABCD parameters of this structure
        
        Z0 - if specified, over-rides the previously specified port
        impedances of these S-parameters
        """
        # TODO: update ACBD and T parameters to allow multiple ports        
        
        if self.shape[1] != 2:
            raise Exception("ABCD parameters only valid for 2-port networks")
    
        ABCD = ABCD_params(f=self.f)
        S = self
        if Z0 == None:
            Z0 = self.Z0
        
        ABCD[1,1] =        ((1.0 + S[1,1])*(1.0 - S[2,2]) 
                            + S[2,1]*S[1,2])/(2.0*S[2,1])
        ABCD[1,2] =     Z0*((1.0 + S[1,1])*(1.0 + S[2,2]) 
                            - S[2,1]*S[1,2])/(2.0*S[2,1])
        ABCD[2,1] = 1.0/Z0*((1.0 - S[1,1])*(1.0 - S[2,2]) 
                            - S[2,1]*S[1,2])/(2.0*S[2,1])
        ABCD[2,2] =        ((1.0 - S[1,1])*(1.0 + S[2,2]) 
                            + S[2,1]*S[1,2])/(2.0*S[2,1])
    
        return ABCD
    
    def Y_params(self, Z0 = None):
        """
        Calculate the Y parameters of this structure
        
        Z0 - if specified, over-rides the previously specified port
        impedances of these S-parameters, can be a vector to have
        different impedance for each port
        """
        if Z0 == None:
            Z0 = self.Z0
          
        S = self
        Y = Y_params(f = self.f, ports = self.ports)
        
        if False: #self.ports == 2:
            den = ((1.0 - S[1,1])*(1.0 - S[2,2]) - S[1,2]*S[2,1])*Z0
            Y[1,1] = ((1 - S[1,1])*(1 + S[2,2]) + S[1,2]*S[2,1])/den
            Y[1,2] = -2*S[1,2]/den
            Y[2,1] = -2*S[2,1]/den
            Y[2,2] = ((1 + S[1,1])*(1 - S[2,2]) + S[1,2]*S[2,1])/den
        else:
            I = np.eye(self.ports)
            Z_fact = I*1.0/np.sqrt(np.array(Z0))
            for count in range(len(self.f)):
                Y[count] = dot(Z_fact, dot(solve(I + S[count], I - S[count]), Z_fact))
        
        return Y

    def Z_params(self, Z0 = None):
        """
        Covert S-parameters to Y-parameters
        
        Z0 - if specified, over-rides the previously specified port
        impedances of these S-parameters, can be a vector to have
        different impedance for each port
        """
        if Z0 == None:
            Z0 = self.Z0
          
        S = self
        Z = Z_params(f = self.f, ports = self.ports)
        
        I = np.eye(self.ports)
        Z_fact = I*np.sqrt(np.array(Z0))
        
        for count in range(len(self.f)):
            Z[count] = dot(Z_fact, dot(solve(I - S[count], I + S[count]), Z_fact))
        
        return Z



class ABCD_params(network_params):
    """
    ABCD (voltage-current transfer) parameters

    Definition of parameters is:
  
    |V[1]|   | A[1,1]  B[1,2] | | V[2]|
    |    | = |                | |     |
    |I[1]|   | C[2,1]  D[2,2] | |-I[2]|

    Note that these parameters are only valid for 2-port networks
  
    """
    def __init__(self, f, data = None):
        if data != None and data.shape[1] != 2:
            raise Exception("ABCD parameters only valid for 2-port networks")
        network_params.__init__(self, f=f, data=data, ports=2)
    
    def cascade(self, next_network):
        """
        Cascades this network with next_network, which will have it's port
        1 connected to port 2 of this network
        """
        if not self.same_freq(next_network):
            raise Exception(
            "Cannot cascade networks with different number of frequencies")
        result = ABCD_params(f=self.f)
        for count in range(len(self.f)):
            result[count] = np.dot(self[count], next_network[count])
            # ** check order!!!
        return result
    
    
    def self_cascade(self, N):
        """
        Cascades this network with itself N times
        """
        result = []
        
        for abcd in self[:]:
            result.append(np.linalg.matrix_power(abcd,N))
            
        return ABCD_params(f=self.f, data=np.array(result))
  
    def S_params(self, Z0 = 50.0):
        "Convert to S parameters"
  
        A = self[1, 1]
        B = self[1, 2]
        C = self[2, 1]
        D = self[2, 2]
        
        S = S_params(self.f, ports = 2) #zeros((len(self), 2, 2), complex)
        S[1,1] = (A + B/Z0 - C*Z0 - D)  / (A + B/Z0 + C*Z0 + D) 
        S[1,2] =       2.0*(A*D - B*C)  / (A + B/Z0 + C*Z0 + D) 
        S[2,1] =                   2.0  / (A + B/Z0 + C*Z0 + D) 
        S[2,2] = (-A + B/Z0 - C*Z0 + D) / (A + B/Z0 + C*Z0 + D)
        
        return S #S_params(data, self.f, Z0)
  
    def terminate_impedance(self, Z, port = 2):
        """Terminate this ABCD network in an impedance Z at port 2,
        and return the resultant impedance"""
        A = self[1, 1]
        B = self[1, 2]
        C = self[2, 1]
        D = self[2, 2]
    
        if port == 1:
            return (D*Z + B)/(C*Z+A)
        else:
            return (A*Z + B)/(C*Z+D)
 
def cascade(*networks):
    "Cascade a several ABCD networks, listed starting from the input network"

    result = ABCD_params(networks[0].f)
    result[1,1] = 1.0
    result[2,2] = 1.0
  
    for network in networks: #[-2:1:-1]:
        result = result.cascade(network)

    return result
    
class T_params(wave_params):
    """
    Scattering Transfer (T) parameters
  
    Definition for a 2-port network is:
  
    |b[1]|   | T[1,1]  T[1,2] | |a[2]|
    |    | = |                | |    |
    |a[1]|   | T[2,1]  T[2,2] | |b[2]|  
  
    Definition for a 2n-port network:
        
    |b[1]|   | T[1,1] ... T[1,2n] | |a[n+1]|
    | .  |   |   .           .    | |      |
    | .  |   |   .           .    | |      |
    | .  |   |   .           .    | |      |
    |b[n]|   | T[n,1] ... T[n,2]  | |a[2n] |
    |    | = |                    | |      |
    |a[1]|   | T[n+1,1]  T[2,2]   | |b[n+1]|  
    | .  |   |   .         .      | |      |
    | .  |   |   .         .      | |      |
    | .  |   |   .         .      | |      |
    |a[n]|   | T[2n,1]  T[2n,2n]  | |b[2n] |
    
    """
    def __init__(self, f, data = None, ports=2, Z0 = None):
        if data != None and data.shape[1] %2 != 0:
            raise ValueError("T parameters are only valid for networks with an even number of ports")
        network_params.__init__(self, f=f, data=data, ports=ports)
        self.Z0 = Z0
    
    def T_params(self, Z0 = None):
        if Z0 == self.Z0 or Z0 is None:
            return self
        else:
            raise NotImplementedError("Impedance conversion not defined")
      
    def S_params(self, Z0 = None):
        if self.ports != 2:
            raise NotImplementedError
            
        S = S_params(f=self.f, ports=2)
        T = self
        
        S[1,1] = T[1,2]/T[2,2]
        S[1,2] = (T[1,1]*T[2,2] - T[1,2]*T[2,1])/T[2,2]
        S[2,1] = 1.0/T[2,2]
        S[2,2] = -T[2,1]/T[2,2]
        return S
        
    def cascade(self, next_network):
        """
        Cascades this network with next_network, which will have it's port
        1 connected to port 2 of this network
        """
        if not self.same_freq(next_network):
            raise ValueError("Cannot cascade networks with different frequencies")
        result = T_params(f=self.f, ports=self.ports)
        for count in range(len(self.f)):
            result[count] = np.dot(self[count], next_network[count])
            # ** check order!!!
        return result        
  

class Z_params(network_params):
    """
    Impedance (Z) parameters
  
    definition for a two port network is:
  
    |V[1]|   | Z[1,1]  Z[1,2] | |I[1]|
    |    | = |                | |    |
    |V[2]|   | Z[2,1]  Z[2,2] | |I[2]|

    """
    def __init__(self, f, data = None, ports = 0):
        network_params.__init__(self, f=f, data=data, ports=ports)

    def S_params(self, Z0):
        if self.shape[1] == 1:
            S = S_params(f=self.f, Z0=Z0, ports=1)
            Z = self
            S[:] = (Z[:]-Z0)/(Z[:]+Z0)
            return S
        else:
            raise Exception(
            "Converting Z params to S params not fully implemented")

class Y_params(network_params):
    """
    Admittance (Y) parameters
  
    definition for a two port network is:
  
    |I[1]|   | Y[1,1]  Y[1,2] | |V[1]|
    |    | = |                | |    |
    |I[2]|   | Y[2,1]  Y[2,2] | |V[2]|

    """
    def __init__(self, f, data = None, ports = 0):
        network_params.__init__(self, f=f, data=data, ports=ports)

    def S_params(self, Z0):
        Z0 = np.array(Z0)
        Y = self
        S = S_params(f = self.f, ports = self.ports, Z0=Z0)
        
        I = np.eye(self.ports)
        Z_fact = I*1.0/np.sqrt(Z0)
        for count in range(len(self.f)):
            S[count] = 2*dot(Z_fact, dot(np.linalg.inv(Y[count] + I*1/Z0), Z_fact)) - I
        return S
        
    def Z_params(self):
        Z = Z_params(f = self.f, ports = self.ports)
        
        for count in range(len(self.f)):
            Z[count] = inv(self[count])
        
        return Z

def loadsnp(filename, force_format = None, Z0 = None, n = None, return_comments=False):
    """
    Load the network parameters from a Touchstone format .snp file
  
    The object will be returned in the format specified in the file, 
    however with the force_format argument you can ensure that a 
    particular type (e.g. S-parameters) will be returned
  
    Parameters
    ----------
    Z0 : number, optional
        force characteristic impedance Z0 to be a particular value
    n : integer, optional
        override detection of number of ports
    force_format : string, optional
        if not None, automatically this particular type of n-port object
        valid types are 's', 'z' or 'y' parameters    
    Returns
    -------
    S : S_params
        The scattering parameters of the object. If `force_format` is a parameter
        type, then the corresponding object type will be returned instead
    """

    freq_mult = { 'hz' : 1.0, 'khz' : 1e3,  'mhz' : 1e6,
                 'ghz' : 1e9, 'thz' : 1e12, 'phz' : 1e15}
    param_class = {'s' : S_params, 'z' : Z_params, 'y' : Y_params}

    # around 10% of time spent in this function
    def get_line(file_obj):
        """gets a line from the file object, ignoring comment or blank lines
        returns it as a list of strings"""
        while True:
            l = file_obj.__next__().split()
            if len(l) != 0 and l[0][0] != '!':
                break
                
        return l

    def get_line_real(file_obj):
        """gets a line from the file object, ignoring comment or blank lines
        returns it as a list of reals"""
        while True:
            l = file_obj.__next__().split()
            if len(l) != 0 and l[0] != '!':
                break
        return [float(a) for a in l]

    def get_header_comments(file_obj):
        comments = []
        while True:
            l = file_obj.__next__()
            if l[0] == '!':
                comments.append(l)
            elif l[0] == '#':
                return l.split(), comments
            else:
                raise ValueError("Invalid line contents: "+l)

    # around 75% of time spend in these functions
    # These functions are designed to operate on array after it is completely loaded
    
    # TODO: check that refactoring has worked
    def ri_to_complex(cdata):
        return cdata

    def ma_to_complex(cdata):
        return cdata.real*np.exp(1j*cdata.imag/180*pi) #+1j*cdata.real*np.sin(angle)

    def db_to_complex(cdata):       
        return 10.0**(cdata.real/20.0)*np.exp(1j*cdata.imag/180*pi)


    format_decoders = {'ri': ri_to_complex, 'ma': ma_to_complex, 'db': db_to_complex}

    # from filename, work out the number of ports
    if n is None:
        n = int(re.findall(r'\.[Ss]([\d]+)[Pp]', 
                           os.path.splitext(filename)[1])[0])
  
    f = []
    S = []
    S_temp = np.zeros((n, n), complex)

    with open(filename, "rt") as input_file:
        
        # get the header line first
        header, comments = get_header_comments(input_file)
        if header[0] != '#':
            raise ValueError("First non-blank line in snp file must be header")
        
        multiplier = freq_mult[header[1].lower()]
        which_class = param_class[header[2].lower()]
        
        try:
            convert_complex = format_decoders[header[3].lower()]
        except KeyError:
            raise ValueError("Unkown S parameters format code: %s" %header[3])
      
        if Z0 == None and len(header) > 4:
            Z0 = float(header[5])
            
        while True:
        
            # Try to get another line from the file.If fail then its the end of 
            # file in a logical spot, so just finish iterating.
            try:
                data = get_line_real(input_file)
            except StopIteration:
                break
                
            if n < 3:
                # for 1 port or 2 port data, everything is all on the same line
                
                f.append(data[0]*multiplier)
              
                line_pos = 1
              
                # NB: order is deliberately switched, S11, S21, S12, S22
                for inner_count in range(n):
                    for outer_count in range(n):
                        S_temp[outer_count, inner_count] = \
                            complex(data[line_pos], data[line_pos+1])
                            #get_complex(data[line_pos], data[line_pos+1])
                        line_pos += 2
                S.append(S_temp.copy())

            else:
                # 3 port networks and larger are in "matrix" form
               
                # get the frequency from the first line
                f.append(data[0]*multiplier)
                data = data[1:]
                
                # each line corresponds to a measuring port
                for meas_count in range(n):
                    if meas_count > 0:
                        data = get_line_real(input_file)
                    
                    # grab further lines if necessary to make up enough data
                    while len(data) < 2*n:
                        data.extend(get_line_real(input_file))
                
                    for source_count in range(n):
                        # check that this is the correct order of indices
                        S_temp[meas_count, source_count] = \
                            complex(data[2*source_count], data[2*source_count+1])
                
                S.append(S_temp.copy())
               
        # previous routines all assume real and imaginary data
        # it is quicker to convert other forms at the very end in one array operation        
        S = convert_complex(np.array(S))
        
    if return_comments:
        return which_class(f=np.array(f), data = S, Z0=Z0), comments
    else:
        return which_class(f=np.array(f), data = S, Z0=Z0)
    

def reduce_all_snp(directory, input_extension, selected_ports, input_force_ports = None):
    """Go through all files in a directory with the given extension.  Load them
    as .snp files, select the particular ports which are desired, then dump to
    another file (with appropriate extension for the number of ports)
    
    dir - the directory where the source and destination files are
    input_extension - all files with this extension will be processed
    selected_ports - a list of the numbers of the ports to be kept
    input_force_ports (optional) - in case the input file name does not indicate
    the number of ports, specify it here  
    """
    import os
    from os.path import join
    
    for input_filename in os.listdir(directory):
        full_name = join(directory, input_filename)
        split_input = os.path.splitext(full_name)
        if split_input[1] != input_extension:
            continue
        
        S = loadsnp(full_name, n=input_force_ports)
        S_new = S.select_ports(selected_ports)
        output_extension = ".s%dp" % len(selected_ports)
        output_name = join(directory, split_input[0]+output_extension)
        S_new.writesnp(output_name)

def linear_to_circular(pxx, pxy, pyx, pyy):
    """Convert parameters in a linear polarisation basis to a circular polarisation basis
    
    Parameters
    ----------
    pxx, pxy, pyx, pyy: array
        The transmission components in a cartesian basis
        
    Returns
    -------
    prr, prl, plr, pll : array
        The transmission components in a circular basis
        
    The definition of Right and left-handed is ???????
    """
    # TODO:check definitions of left and right handed convention        
    
    prr = 0.5*(pxx + pyy + 1j*(pxy - pyx))
    prl = 0.5*(pxx - pyy - 1j*(pxy + pyx))
    plr = 0.5*(pxx - pyy + 1j*(pxy + pyx))
    pll = 0.5*(pxx + pyy - 1j*(pxy - pyx))
    
    return prr, prl, plr, pll


