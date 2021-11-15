import numpy as np
import pint 
import matplotlib.pyplot as plt
import findiff as fd

# Initialise the standard Pint Unit Registry
ureg = pint.UnitRegistry()

# Top tips with Pint:
# Check dimensionality with a.dimensionality
# Convert units using a.to method, e.g. speed.to('inch/minute')

# Define base units
meter = ureg.meter
second = ureg.second
kelvin = ureg.kelvin
kPa = ureg.kPa
kg = ureg.kg
mol = ureg.mol
kJ = ureg.kJ

# Compound units
area = meter**2
cube = meter**3

# Attribute of heap: array to keep track of all the variables of time - gets updated with heap.record method...
history = np.empty( (1,3)) 

# Store differential operators in a dictionary?

# Define constants
c = {
    'M_Ch':[ (159.16/1000) * kg/mol, 'Molar Mass of Chalcocite (Cu2S)' ],
    'M_Py':[ (119.98/1000) * kg/mol, 'Molar Mass of Pyrite (FeS2)' ],
    'M_Ox':[ (31.99/1000) * kg/mol, 'Molar Mass of Oxygen (O2)' ],
    }

# Set model parameters
params = { 
    'T_atmos':[298*kelvin, 'Atmospheric temperature'],
    'P_atmos':[101*kPa, 'Atmospheric pressure'],
    'X':[5*10**(13)/cube, 'Bacterial population density'],
    'K':[5*10**(-10)/area, 'Bed Permeability'],
    'GCu':[0.5, 'Copper grade: percentage by weight'],
    'G^0':[ 0.63, 'Chalcocite grade: percentage by weight'],
    'KPY':[0, 'Pyrite factor: kg pyrite leached / kg chalcocite leached'],
    'T_L':[298*kelvin, 'Liquid Temperature'],
    'q_L':[1.4*10**(-6)*cube/(second*area), 'Volume flow rate of Liquid per unit area'],
    'O2g':[0.26*kg/cube, 'Oxygen concentration at T_atmos and P_atmos'],
    'sigma_1':[ c['M_Ch'][0]*c['M_Py'][0]/( (5/2)*c['M_Ox'][0]*c['M_Py'][0] + (7/2)*c['M_Ox'][0]*c['M_Ch'][0] ), 'Stoichimoetric factor (see bioleaching model by Casas et al)' ],
    'Vm_p1':[6.8*10**(-13)*kg/(second*kelvin*cube), 'Bacterial respiration rate - param 1'],
    'Vm_p2':[7000*kelvin, 'Bacterial respiration rate - param 2'],
    'Vm_p3':[74000*kelvin, 'Bacterial respiration rate - param 3'],
    'rho_B':[1800*kg/cube, 'Average density of the ore Bed or dump'],
    'K_m':[10**(-3)*kg/cube, 'Michaelis constant (see Michaelis-Menten equation)'],
    'eps_L':[0.12, 'Volumetric fraction of liquid phase within the ore bed'], # Is this parameter fraction of all matter in ore bed or fraction of the fluid matter only? 
    'eps_g':[0.88, 'Volumetric fraction of gas phase within the ore bed'], # See comment above - i.e. is this deduction valid?
    'D_g':[1.5*10**(-5)*area/second, 'Diffusion coefficient for oxygen in the has phase'],
    'k_B':[ 2.1*10**(-3)*kJ/(meter*kelvin*second), 'Thermal conductivity of the ore bed']
    }

dt = 5/100 # input into heap leach method? 
# Collection of differential operators
# diffops = { 
#     'Le':[fd.FinDiff(0, 1)] }

def see_params(values=False):
    for key in params:
        print(key, '->', params[key][1])
        if values:
            print('; value =', params[key][0], '\n')

# Define the methods

# Maximum respiration rate of the bacteria 
def Vm(T):
    ''' The maximum specific bacterial respiration rate as a function of temperature (Vm). 
    Units are kg's (of liquid Oxygen consumed) per second per number of bacteria per cubic meter. Input must be in Kelvin.'''  
    if hasattr(T, 'dimensionality'):
        assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
        return params['Vm_p1'][0] * T * np.exp( -params['Vm_p2'][0]/T ) /  (1 + np.exp( 236 - params['Vm_p3'][0]/T ) )
    else:
        print('Input T must be in Kelvin!')

# Reaction rate - Michaelis-Menten equation with Oxygen as the limiting substrate 
def alpha_dot(T, C_L):
    ''' Velocity of the combined chemical and bacterial reactions as a function of temperature (T) and liquid
    Oxygen concentration (C_L); i.e. the rate of copper-sulfide dissolution. 
    Valid only when Oxygen is the limiting substrate (i.e. sufficient bacteria and ferrous Fe ions).'''
    if hasattr(C_L, 'dimensionality'):
        assert C_L.dimensionality == params['K_m'][0].dimensionality, 'Input C_L must be in {}!'.format(params['K_m'][0].units)
        factor = params['sigma_1'][0]*params['X'][0]/(params['rho_B'][0]*params['G^0'][0])
        return factor * Vm(T) * C_L/(params['K_m'][0] + C_L )
    else:
        print('Input C_L must be in {}!'.format(params['K_m'][0].units))
   
  
# heap takes in a meshgrid of XX, YY,  (this will be useful for plotting surfaces later on)
def stack( d , N, symmetric = True, info = False ): 
    ''' Returns a numpy.meshgrid with number of lattice nodes set by N 
    and spatial dimensions set by d.'''
    if symmetric:
        x = np.linspace(-d[0]/2, d[0]/2, N[0])
        y = np.linspace(-d[1]/2, d[1]/2, N[1])
        if info:
            print( '\n Heap succesfully stacked! ')
            print( '\t Shape = ({}, {})'.format(N[0], N[1]) )
            print( '\t Dimensions (relative to origin):' )
            print( '\t'*2,  'd = (x, y) = {}'.format(d))
            print( '\t',' => spacetime bounds (symmetric):')
            print( '\t'*2,  'x_a -> x_b = {} -> {}'.format( -d[0]/2, d[0]/2))
            print( '\t'*2,  'y_a -> y_b = {} -> {}'.format( -d[1]/2, d[1]/2))
            print( '\t Number of nodes (per dimension):'  )
            print( '\t'*2,  'N = (Nx, Ny , Nt) = {}'.format(N)) 
        return np.array( np.meshgrid( x, y , indexing = 'ij') )
    else:
        print('complete non-symmetric version later')
    
N = (6, 8)
d = (5, 7)
stk =  stack( d , N )

def space_grain(stack):
        Delta_X = abs(stack[0][1][0] - stack[0][0][0])
        Delta_Y = abs(stack[1][0][0] - stack[1][0][1])
        return Delta_X, Delta_Y
      
def plt_stack(stack):
    plt.figure()
    plt.plot(stack[0], stack[1], marker='.', color='k', linestyle='none')

def leach(a_dot, dt):
    pass
''' Returns alpha (the molar concentration of Cu2S04) at time t = t + Delta_t
for all heap grid-points (x, y), given liquid oxygen concentration C_L(x, y, t)
and temperature T(x, y, t). '''
    
# C_L = np.full(N, 0.006) * kg/cube
# CL_mol = C_L/c['M_Ox'][0]/mol # dividing by mol because I still need to figure out what the units of alpha should be....
# alpha = np.zeros(N) / cube
# T = np.full(N, 250) * kelvin
# dt = 5/100*second

# # Reaction rate
# a_dot = alpha_dot(T, C_L) # outside of the leach method because it will be needed elsewhere
# alpha_formed = a_dot*dt
# alpha += alpha_formed
# ox_to_alpha = 2.5/2 # Oxygen to Cu2SO4 based on the stoiciometric coefficients
# # oxygen_lost = ox_to_alpha*alpha_formed
# CL_mol -= ox_to_alpha*alpha_formed


# for i in range(0,10):
#     alpha += alpha_formed
#     CL_mol -= ox_to_alpha*alpha_formed
#     print('alpha =', alpha[0][0])
#     print('Ox =', CL_mol[0][0])


stk = stack( (10, 10), (11, 21)  )
DX, DY = space_grain(stk)
sub_grain = 5
dx = DX/sub_grain
dy = DY/sub_grain

# axis 0 = x
# axis 1 = y
# axis 2 = t # no need for third access, just use a_dot()

k_B = params['k_B'][0]
L_EB = k_B * (fd.FinDiff(0, dx, 2) + fd.FinDiff(1, dy, 2 ) )




# Variable dictionary: a description of each variable used in the model
var_dict = {
    'alpha':'The product balance resulting from the combined chemical and bacterial reactions', 
    'alpha_dot':alpha_dot.__doc__,
    'C_L':'The concentration of liquid Oxygen',
    'Vm':Vm.__doc__
    }











def slicer():
    '''slice heap along chosen access to get associated variables '''
    pass




class Heap():
    pass

class Humple():
    
    # Class Object Attribute (e.g. attributes belonging to ALL Humples)
    example = 'constants of nature'
    
    def __init__(self, substance):

        self.substance = substance
        
        
    def __str__(self):
        return "Describition of the humple:" 



def how_to_run():
    print('Run using script x... TBC')

# if __name__ == "__main__":
#     how_to_run()

