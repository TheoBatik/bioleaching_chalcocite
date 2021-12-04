##################### IMPORTS ################################################

import numpy as np
import pint
import matplotlib.pyplot as plt
import findiff as fd


##################### UNITS ##################################################

# Initialise the standard Pint Unit Registry
ureg = pint.UnitRegistry()

# Base units
meter = ureg.meter
second = ureg.second
kelvin = ureg.kelvin
kPa = ureg.kPa
kg = ureg.kg
mol = ureg.mol
kJ = ureg.kJ

# Derived units
area = meter**2
cube = meter**3

# Conversions
hour = 1* ureg.hour 
sec_in_hour = hour.to('seconds')
day = 1 * ureg.day
sec_in_day = day.to('seconds')

class Heap():
    
    ####################### ATTRIBUTES ########################################
    
    # Constants
    c = {
        'M_Ch':[ (159.16/1000) * kg/mol, 'Molar Mass of Chalcocite (Cu2S)' ],
        'M_Py':[ (119.98/1000) * kg/mol, 'Molar Mass of Pyrite (FeS2)' ],
        'M_Ox':[ (31.99/1000) * kg/mol, 'Molar Mass of Oxygen (O2)' ],
        }

    # Parameters
    params = { 
        'T_atmos':[298*kelvin, 'Atmospheric temperature'],
        'P_atmos':[101*kPa, 'Atmospheric pressure'],
        'X':[5*10**(13)/cube, 'Bacterial population density'],
        'K':[5*10**(-10)/area, 'Bed Permeability'],
        'GCu':[0.5, 'Copper grade: percentage by weight'],
        'G^0':[ 0.63, 'Chalcocite grade: percentage by weight'],
        'G':[ 1.67*kg/(area*sec_in_hour), 'Mass flow rate of dry air per unit area - lowest rate used in Dixon'],
        'sigma_1':[ None , 'Stoichimoetric factor (see bioleaching model by Casas et al)' ],
        'FPY':[0, 'Pyrite factor: kg pyrite leached / kg chalcocite leached'],
        'T_L':[298*kelvin, 'Liquid Temperature'],
        'q_L':[1.4*10**(-6)*cube/(second*area), 'Volume flow rate of Liquid per unit area'],
        'O2g':[0.26*kg/cube, 'Oxygen concentration at T_atmos and P_atmos'],
        'Vm_p1':[6.8*10**(-13)*kg/(second*kelvin*cube), 'Bacterial respiration rate - param 1'],
        'Vm_p2':[7000*kelvin, 'Bacterial respiration rate - param 2'],
        'Vm_p3':[74000*kelvin, 'Bacterial respiration rate - param 3'],
        'rho_B':[1800*kg/cube, 'Average density of the ore Bed or dump'],
        'rho_air':[1.16*kg/cube, 'Average density of the air'],
        'rho_L':[1050*kg/cube, 'Average density - liquid phase'],
        'rho_S':[2700*kg/cube, 'Average density - solid phase'],
        'K_m':[10**(-3)*kg/cube, 'Michaelis constant (see Michaelis-Menten equation)'],
        'eps_L':[0.12, 'Volumetric fraction of liquid phase within the ore bed'], # Is this parameter fraction of all matter in ore bed or fraction of the fluid matter only? 
        'eps_g':[0.88, 'Volumetric fraction of gas phase within the ore bed'], # See comment above - i.e. is this deduction valid?
        'D_g':[1.5*10**(-5)*area/second, 'Diffusion coefficient for oxygen in the gas phase'],
        'k_B':[ 2.1*10**(-3)*kJ/(meter*kelvin*second), 'Thermal conductivity of the ore bed'],
        'Delta_H_Ch':[-6000*kJ/kg, 'Chalcocite reaction enthalpy'],
        'Delta_H_Py':[-12600*kJ/kg, 'Pyrite reaction enthalpy'],
        'ASH_G':[1.0*kJ/(kg*kelvin), 'Average Specific Heat - Gas'],
        'ASH_L':[4.0*kJ/(kg*kelvin), 'Average Specific Heat - Liquid'],
        'ASH_S':[1.172*kJ/(kg*kelvin),'Average Specific Heat - Solid'],
        'ASH_V':[1.1864*kJ/(kg*kelvin), 'Average Specific Heat - Vapour water'],
        'lambda':[583*kJ/kg, 'Heat of water vapourisation'],
        'H_air':[0.5, 'Average air humidity', 'Use Antoine equation for more accurate result.'],
        'Henry_1':[ 21.312  , 'Source 18, 28 in Casas bib' ],
        'Henry_2':[ 0.784 , 'Source 18, 28 in Casas bib' ],
        'Henry_3':[ 0.00383 , 'Source 18, 28 in Casas bib' ], 
        'Ox_in_air':[ 0.21 , 'Percentage composition of oxygen in air.' ],
        'coxg_fac':[ None , 'Concentration of gaseous oxygen factor.' ]
        }
    
    # Initialisation
    def __init__(self, dim = (10, 10, 5), ms = (21, 21, 50) ):
        
        ##### Spacetime ######
        
        # General
        self.dim = dim
        self.ms = ms
        self.mesh = None
        self.history = np.empty(self.ms) 
       
        # Space 
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None
        
        # Time
        self.dt = None
        self.Te = None
        
        ##### Energy Exchange Operators ######
        
        self.Te_fac = None # fix        
        # Conduction operators
        self.Ec = None
        # Liquid Flow Operator
        self.EL_fac = None
        self.EL = None
        # Gas flow Operator
        self.Eg_fac = None
        self.Eg = None
        # Energy Exchange Operator
        self.Ex = None
        # Heat of reaction
        self.DeltaH_R = self.params['Delta_H_Ch'][0] + self.params['FPY'][0] * self.params['Delta_H_Py'][0]
       
        ##### Derived paraeters  ######
        self.params['sigma_1'][0] = self.c['M_Ch'][0]*self.c['M_Py'][0]/( (5/2)*self.c['M_Ox'][0]*self.c['M_Py'][0] + (7/2)* self.params['FPY'][0] * self.c['M_Ox'][0]*self.c['M_Ch'][0] )
        self.coxg_fac = self.params['Ox_in_air'][0] * self.params['rho_air'][0] * kg/cube # add to params dictionary 
        self.source_fac = self.params['rho_B'][0] * self.params['G^0'][0] / ( self.params['sigma_1'][0] *  self.params['X'][0] ) # add to params dictionary 
        
       
    ####################### Methods ##########################################     

    
    ## Prepare heap space: return meshgrid and 1-D arrays x, y, t
    def stack( self, symmetric = True, info = False ): 
        ''' Returns a numpy.meshgrid with number of lattice nodes set by N 
        and spatial dimensions set by d.'''
        d = self.dim 
        N = self.ms
        if symmetric:
            x = np.linspace(-d[0]/2, d[0]/2, N[0])
        else:
            x = np.linspace(0, d[0], N[0])
        y = np.linspace(0, d[1], N[1])
        t = np.linspace(0, d[2], N[2])
        mesh = np.array( np.meshgrid( x, y , t, indexing = 'ij') )
        # Update Heap attributes
        setattr(self, 'mesh', mesh)
        setattr(self, 'x', x)
        setattr(self, 'y', y)
        setattr(self, 't', t)
        if info:
            print( '\n Heap succesfully stacked! ')
            print( '\n\t Dimensions (relative to origin):' )
            print( '\t'*2,  'dim = (x, y, t) = {}'.format(d))
            print( '\n\t Mesh shape:'  )
            print( '\t'*2,  'ms = (Nx, Ny , Nt) = {}'.format(N)) 
        return x, y, t, mesh

          
    def plot_mesh(self, mesh):
        plt.figure()
        plt.plot(mesh[0], mesh[1], marker='.', color='k', linestyle='none')
        
    def diffs(self, x, y, t):
        ''' Returns the differentials dx and dy, given the 1-D spatial arrays x and y.''' 
        dx = abs(x[1] - x[0]) 
        dy = abs(y[1] - y[0])
        dt = abs(t[1] - t[0])
        setattr(self, 'dx', dx)
        setattr(self, 'dy', dy)
        setattr(self, 'dt', dt)
        return dx, dy, dt  
    
    def dissolve(self, gas, temp):
        ''' Dissolution of gaseous oxygen by Henry's law at temperature T''' 
        if hasattr(temp, 'dimensionality'):
            assert temp.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
        else:
            raise('Input T must be in Kelvin!') 
        assert gas.shape == self.ms, 'Gas shape must be {}'.format(self.ms) 
        celsius = temp.magnitude - 274
        henry = lambda T: self.params['Henry_1'][0]  + self.params['Henry_2'][0] * T -  self.params['Henry_3'][0] * T **2 
        liquid = henry(celsius) * gas
        return liquid
    
    
    def init_ops(self, accuracy = 4):
        ''' Initiate all differential operators.'''
        if getattr(self, 'mesh') is None:
            raise('Heap must be stacked, before differential operators can be initiated.')
        elif getattr(self, 'dx') is None or getattr(self, 'dy') is None or getattr(self, 'dt') is None:
            raise('Differentials (dx, dy, dt) required before operators can be initiated.')
        else:
            # Energy Exchange:
            ## Conduction Operator 
            setattr(self, 'Ec', self.params['k_B'][0] * (fd.FinDiff(0, self.dx, 2, acc = accuracy)  + fd.FinDiff(1, self.dy, 2, acc = accuracy )  ) * meter ** (-2) )
            ## Liquid Flow Operator
            setattr(self, 'EL_fac', -  self.params['q_L'][0] * self.params['rho_L'][0] * self.params['ASH_L'][0] )
            setattr(self, 'EL',  self.EL_fac * fd.FinDiff(1, self.dy, 1, acc = accuracy) * meter ** (-1) )
            ## Gas flow Operator
            setattr(self, 'Eg_fac', self.params['G'][0]*(  self.params['ASH_G'][0]  + self.params['ASH_V'][0] * self.params['H_air'][0]  ) )
            setattr(self, 'Eg',  self.Eg_fac * (fd.FinDiff(0, self.dx, 1, acc = accuracy) + fd.FinDiff(1, self.dy, 1, acc = accuracy) ) * meter ** (-1) )
            ## Energy Exchange Operator
            setattr(self, 'Ex', self.Ec -  self.EL - self.Eg ) # Check minus sign on EL
            # Time-evolution operator:
            setattr(self, 'Te_fac',  self.params['rho_B'][0] * self.params['ASH_S'][0] )
            setattr(self, 'Te',  self.Te_fac * fd.FinDiff(2, self.dt, 1, acc = accuracy) * second ** (-1) )
        
# ################### OXYGEN BALANCE OPERATORS #################################

# # Diffusion Operator
# Od_factor = params['eps_g'][0] * params['D_g'][0]
# Od =  Od_factor *  (  fd.FinDiff(0, DX, 2)  + fd.FinDiff(1, DY, 2 )  ) * meter ** (-2)
# # Convection Operator
# Oc = params['eps_g'][0] * (fd.FinDiff(0, DX, 1) + fd.FinDiff(1, DY, 1) ) * meter ** (-1) 
# # Oxygen Balance Operator
# #Ob = Od - Oc 

        
        
   ## Maximum respiration rate of the bacteria 
    def Vm(self, T):
        ''' The maximum specific bacterial respiration rate as a function of temperature (Vm). 
        Units are kg's (of liquid Oxygen or iron or both?) per second per number of bacteria per cubic meter. Input must be in Kelvin.'''  
        if hasattr(T, 'dimensionality'):
            assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
            return  self.params['Vm_p1'][0] * T * np.exp( -self.params['Vm_p2'][0]/T ) /  (1 + np.exp( 236 - self.params['Vm_p3'][0]/T ) )
        else:
            print('Input T must be in Kelvin!')
    
    ## Reaction rate - Michaelis-Menten kinetics with liquid Oxygen as the limiting substrate 
    def ccu_dot(self, T, coxl):
        ''' Velocity of the combined chemical and bacterial reactions as a function of temperature (T) and liquid
        Oxygen concentration (coxl); i.e. the rate of copper-sulfide dissolution. 
        Valid only when Oxygen is the limiting substrate (i.e. sufficient bacteria and ferrous Fe ions).'''
        if hasattr(coxl, 'units'):
            assert coxl.units == self.params['K_m'][0].units, 'Concentration liquid oxygen (coxl) must be in {}!'.format(self.params['K_m'][0].units)
            cu_dot_fac = self.params['sigma_1'][0]*self.params['X'][0]/(self.params['rho_B'][0]*self.params['G^0'][0]) 
            return cu_dot_fac * self.Vm(T) * coxl/(self.params['K_m'][0] + coxl )
        else:
            print('Concentration liquid oxygen (coxl) must be in {}!'.format(self.params['K_m'][0].units) )
                       
    ## Print all parameters
    def print_params(self, values=False):
        ''' Print a readable list of model parameter names with descriptions (and values if True).''' 
        for key in self.params:
            print(key, '->', self.params[key][1])
            if values:
                print('; value =', self.params[key][0], '\n')               
        

        
    

    # Exchange energy given source function, boundary conditions and differential operator
    def energy_exchange(self, source, bc, operator = None ):
        ''' Solves the energy exchange PDE, given source function (source) and boundary conditions
        (bc). Returns the temperature distribution at current time slice.'''
        if operator is None:
            if getattr(self, 'Ex') is None:
                raise( 'First initiate the differential operators by running init_ops()')
            else: 
                operator = self.Ex
                print('Using default energy exchange operator.')
        pde = fd.PDE(operator.magnitude, source.magnitude, bc)
        T = pde.solve()
        return T
    
## Doc_string        
    def __str__(self, methods = True):
        # Attributes:
        s = '##### Attributes ##### \n'
        ats = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        for a in ats:
            s += '\n ==> ' + a
        # Methods:
        s += '##### Methods ##### \n' 
        meth = [m for m in dir(self) if not a.startswith('__') and callable(getattr(self, a))] 
        for m in meth:
            s += '\n ==> ' + m 
            if methods:
                s += ' = ' + m.__doc__
        return s
            



    








# # Variable dictionary: a description of each variable used in the model
# var_dict = {
#     'alpha':'The product balance resulting from the combined chemical and bacterial reactions', 
#     'alpha_dot':alpha_dot.__doc__,
#     'C_L':'The concentration of liquid Oxygen',
#     'Vm':Vm.__doc__
#     }


def gas_flow():
    pass

def oxygen_balance():
    pass

# def how_to_run():
#     print('Run using script x... TBC')

# if __name__ == "__main__":
#     how_to_run()

