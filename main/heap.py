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
    'G':[ 1.67*kg/(area*sec_in_hour), 'Mass flow rate of dry air per unit area - lowest rate used in Dixon'],
    'FPY':[0, 'Pyrite factor: kg pyrite leached / kg chalcocite leached'],
    'T_L':[298*kelvin, 'Liquid Temperature'],
    'q_L':[1.4*10**(-6)*cube/(second*area), 'Volume flow rate of Liquid per unit area'],
    'O2g':[0.26*kg/cube, 'Oxygen concentration at T_atmos and P_atmos'],
    'sigma_1':[ c['M_Ch'][0]*c['M_Py'][0]/( (5/2)*c['M_Ox'][0]*c['M_Py'][0] + (7/2)*c['M_Ox'][0]*c['M_Ch'][0] ), 'Stoichimoetric factor (see bioleaching model by Casas et al)' ],
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
     #   'Henry_1':[ (21.312 + 274)  , 'Source 18, 28 in Casas bib' ],
     # 'Henry_2':[ (0.784 + 274)/kelvin , 'Source 18, 28 in Casas bib' ],
     # 'Henry_3':[ (0.00383 + 274)/(kelvin**2) , 'Source 18, 28 in Casas bib' ]
      'Ox_in_air':[ 0.21 , 'Percentage composition of oxygen in air.' ],
    }


################### INITIAL CONDITIONS ########################################

N = (21, 41) # N = size of heap
d = (10, 20)

# Concentrations 

# henry = lambda T: params['Henry_1'][0]  + params['Henry_2'][0] * T -  params['Henry_3'][0] * T **2 
# # print(henry(params['T_atmos'][0]))
# a = 0.006/henry(20)

# Gasesous oxygen
coxg_fac = params['Ox_in_air'][0] * params['rho_air'][0] * kg/cube
coxg = np.full(N, coxg_fac)

## Copper-sulphide
CuS_fac = params['rho_B'][0] * params['G^0'][0]
CuS = np.full(N, CuS_fac ) * kg/cube

# Copper-sulphate
cu = np.zeros(N)

# Thermodynamic #

## temperature
T = np.full(N, params['T_atmos'][0] )
## gas density
rho_non_ox = np.full( N, 0.24 * params['rho_B'][0].units )
gd = Coxg + rho_non_ox


##################### Heap Class #############################################
class Heap():
    
    # Class Object Attributes:

        
    ## constants
    c = {
        'M_Ch':[ (159.16/1000) * kg/mol, 'Molar Mass of Chalcocite (Cu2S)' ],
        'M_Py':[ (119.98/1000) * kg/mol, 'Molar Mass of Pyrite (FeS2)' ],
        'M_Ox':[ (31.99/1000) * kg/mol, 'Molar Mass of Oxygen (O2)' ],
        }

    ## model parameters
    params = { 
        'T_atmos':[298*kelvin, 'Atmospheric temperature'],
        'P_atmos':[101*kPa, 'Atmospheric pressure'],
        'X':[5*10**(13)/cube, 'Bacterial population density'],
        'K':[5*10**(-10)/area, 'Bed Permeability'],
        'GCu':[0.5, 'Copper grade: percentage by weight'],
        'G^0':[ 0.63, 'Chalcocite grade: percentage by weight'],
        'G':[ 1.67*kg/(area*sec_in_hour), 'Mass flow rate of dry air per unit area - lowest rate used in Dixon'],
        'FPY':[0, 'Pyrite factor: kg pyrite leached / kg chalcocite leached'],
        'T_L':[298*kelvin, 'Liquid Temperature'],
        'q_L':[1.4*10**(-6)*cube/(second*area), 'Volume flow rate of Liquid per unit area'],
        'O2g':[0.26*kg/cube, 'Oxygen concentration at T_atmos and P_atmos'],
        'sigma_1':[ c['M_Ch'][0]*c['M_Py'][0]/( (5/2)*c['M_Ox'][0]*c['M_Py'][0] + (7/2)*c['M_Ox'][0]*c['M_Ch'][0] ), 'Stoichimoetric factor (see bioleaching model by Casas et al)' ],
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
        }

    def __init__(self, shape = (10, 10), nodes = (21, 21) ):
        self.shape = shape
        self.nodes = nodes
        # array to keep track of all the variables over time - gets updated with heap.record method...
        self.history = np.empty(self.nodes) 
        
        
####################### Heap Methods #########################################     
   
   ## Maximum respiration rate of the bacteria 
    def Vm(self, T):
        ''' The maximum specific bacterial respiration rate as a function of temperature (Vm). 
        Units are kg's (of liquid Oxygen or iron or both?) per second per number of bacteria per cubic meter. Input must be in Kelvin.'''  
        if hasattr(T, 'dimensionality'):
            assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
            return self.params['Vm_p1'][0] * T * np.exp( -self.params['Vm_p2'][0]/T ) /  (1 + np.exp( 236 - self.params['Vm_p3'][0]/T ) )
        else:
            print('Input T must be in Kelvin!')
    
    ## Reaction rate - Michaelis-Menten kinetics with liquid Oxygen as the limiting substrate 
    def cu_dot(self, T, coxl):
        ''' Velocity of the combined chemical and bacterial reactions as a function of temperature (T) and liquid
        Oxygen concentration (C_L); i.e. the rate of copper-sulfide dissolution. 
        Valid only when Oxygen is the limiting substrate (i.e. sufficient bacteria and ferrous Fe ions).'''
        if hasattr(coxl, 'dimensionality'):
            assert coxl.dimensionality == self.params['K_m'][0].dimensionality, 'Concentration liquid oxygen (coxl) must be in {}!'.format(self.params['K_m'][0].units)
            cu_dot_fac = self.params['sigma_1'][0]*self.params['X'][0]/(self.params['rho_B'][0]*self.params['G^0'][0]) 
            return cu_dot_fac * Vm(T) * coxl/(self.params['K_m'][0] + coxl )
        else:
            'Concentration liquid oxygen (coxl) must be in {}!'.format(self.params['K_m'][0].units)
            
            
    ## Print all parameters
    def see_params(self, values=False):
        ''' Print a readable list of model parameter names with descriptions (and values if True).''' 
        for key in self.params:
            print(key, '->', self.params[key][1])
            if values:
                print('; value =', self.params[key][0], '\n')   
               
                
    ## Prepare heap space: return meshgrid and 1-D arrays x, y
    def stack( self, symmetric = True, info = False ): 
        ''' Returns a numpy.meshgrid with number of lattice nodes set by N 
        and spatial dimensions set by d.'''
        d = self.shape 
        N = self.nodes
        mesh = np.array( np.meshgrid( x, y , indexing = 'ij') )
        self.mesh = mesh
        if symmetric:
            x = np.linspace(-d[0]/2, d[0]/2, N[0])
            y = np.linspace(0, d[1], N[1])
            # update heap size attribute to N
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
            return x, y, mesh
        else:
            print('complete non-symmetric version later')
    
    def space_grain(self, x, y):
        ''' Returns the differentials dx and dy, given the 1-D spatial arrays x and y.''' 
        dx = abs(x[1] - x[0]) 
        dy = abs(y[1] - y[0])
        self.dx = dx
        self.dy = dy
        return dx, dy
          
    def plot_mesh(self, mesh):
        plt.figure()
        plt.plot(mesh[0], mesh[1], marker='.', color='k', linestyle='none')
        
        ## Doc_string        
    def __str__(self):
        return "Describition of the humple:" 
        
    def get_coxl(self, coxg, T):
        ''' Dissolution of gaseous oxygen by Henry's law at temperature T''' 
        if hasattr(T, 'dimensionality'):
            assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
        else:
            print('Input T must be in Kelvin!') 
        assert gas.shape == self.nodes, 'Shape of input gas (coxg) must be {}.'.format(self.nodes) # self.shape
        celsius = temp.magnitude - 274
        henry = lambda T: params['Henry_1'][0]  + params['Henry_2'][0] * T -  params['Henry_3'][0] * T **2 
        liquid = henry(celsius) * gas
        return liquid


##################### RUN ####################################################

heap = Heap(d, N)
T = 1 * kelvin
Vmax = heap.Vm(T)
 
# a_dot = heap.cu_dot(T, )

x, y, mesh = heap.stack()
dx, dy = heap.space_grain(x, y)

p = heap.params

heap.plot_mesh(mesh)

##################### Heap Attributes ########################################
    


# Store differential operators in a dictionary?





dt = 5/100 # input into heap leach method? 


# Collection of differential operators
# diffops = { 
#     'Le':[fd.FinDiff(0, 1)] }




# Print parameters
def see_params(values=False):
    ''' Print a readable list of model parameter names with descriptions (and values if True).''' 
    for key in params:
        print(key, '->', params[key][1])
        if values:
            print('; value =', params[key][0], '\n')

def Vm(T):
        ''' The maximum specific bacterial respiration rate as a function of temperature (Vm). 
        Units are kg's (of liquid Oxygen or iron or both?) per second per number of bacteria per cubic meter. Input must be in Kelvin.'''  
        if hasattr(T, 'dimensionality'):
            assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
            return params['Vm_p1'][0] * T * np.exp( -params['Vm_p2'][0]/T ) /  (1 + np.exp( 236 - params['Vm_p3'][0]/T ) )
        else:
            print('Input T must be in Kelvin!')
 
        
 
# T = 300*kelvin
# factor = T * np.exp( -params['Vm_p2'][0]/T ) /  (1 + np.exp( 236 - params['Vm_p3'][0]/T ) )
# print(factor, 'vs', params['Vm_p1'][0])
# v = Vm(T)
# print(v)


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
   
# factor = params['sigma_1'][0]*params['X'][0]/(params['rho_B'][0]*params['G^0'][0])
# print('factor = ', factor)




# heap takes in a meshgrid of XX, YY,  (this will be useful for plotting surfaces later on)
def stack( d , N, symmetric = True, info = False ): 
    ''' Returns a numpy.meshgrid with number of lattice nodes set by N 
    and spatial dimensions set by d.'''
    if symmetric:
        x = np.linspace(-d[0]/2, d[0]/2, N[0])
        y = np.linspace(0, d[1], N[1])
        # update heap size attribute to N
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
        return x, y, np.array( np.meshgrid( x, y , indexing = 'ij') )
    else:
        print('complete non-symmetric version later')

def space_grain(mesh):
        dx = abs(mesh[0][1][0] - mesh[0][0][0])
        dy = abs(mesh[1][0][0] - mesh[1][0][1])
        return dx, dy
      
def plt_mesh(mesh):
    plt.figure()
    plt.plot(mesh[0], mesh[1], marker='.', color='k', linestyle='none')

def leach(a_dot, dt):
    pass
''' Returns alpha (the molar concentration of Cu2S04) at time t = t + Delta_t
for all heap grid-points (x, y), given liquid oxygen concentration C_L(x, y, t)
and temperature T(x, y, t). '''


def dissolve(gas, temp):
    if hasattr(temp, 'dimensionality'):
        assert T.dimensionality == kelvin.dimensionality, 'Input T must be in Kelvin!'
    else:
        print('Input T must be in Kelvin!') 
    assert gas.shape == N, 'Gas shape must fit into heap shape.' # self.shape
    celsius = temp.magnitude - 274
    henry = lambda T: params['Henry_1'][0]  + params['Henry_2'][0] * T -  params['Henry_3'][0] * T **2 
    liquid = henry(celsius) * gas
    return liquid






N = (21, 41)
d = (10, 20)


x, y, mesh  =  stack( d , N )


C_L = np.full(N, 0.006) * kg/cube # N = size of heap
CL_mol = C_L/c['M_Ox'][0]/mol # dividing by mol because I still need to figure out what the units of alpha should be....
alpha = np.zeros(N) / cube
T = np.full(N, 250) * kelvin
dt = 5/100*second

# Reaction rate
a_dot = alpha_dot(T, C_L) # outside of the leach method because it will be needed elsewhere
alpha_formed = a_dot*dt
alpha += alpha_formed
ox_to_alpha = 2.5/2 # Oxygen to Cu2SO4 based on the stoiciometric coefficients
# oxygen_lost = ox_to_alpha*alpha_formed
CL_mol -= ox_to_alpha*alpha_formed


# for i in range(0,10):
#     alpha += alpha_formed
#     CL_mol -= ox_to_alpha*alpha_formed
#     print('alpha =', alpha[0][0])
#     print('Ox =', CL_mol[0][0])



DX, DY = space_grain(mesh)
# sub_grain = 5
# dx = DX/sub_grain
# dy = DY/sub_grain

# axis 0 = x
# axis 1 = y
# axis 2 = t # no need for third access, just use a_dot()

k_B = params['k_B'][0]
# Energy Exchange Operator
Le = k_B * (fd.FinDiff(0, DX, 2) + fd.FinDiff(1, DY, 2 ) )
shape = N
f = a_dot.magnitude

# X and Y
Y = mesh[1][:][:]
X = mesh[0][:][:]


# bc = fd.BoundaryConditions(N)
# # Dirichlet BC
# bc[0,:] = params['T_atmos'][0] # left 
# bc[-1,:] = params['T_atmos'][0] # right
# bc[:,-1] = params['T_atmos'][0] # top
# # Neumann BC
# bc[:, 0] = fd.FinDiff(1, DY, 1), 0 # bottom
# # mid = round(N[0]/2) 
# # bc[mid, 1:-1] = fd.FinDiff(0, DX, 1), 0

# pde = fd.PDE(Le, f, bc)
# u = pde.solve()

# fig, ax = plt.subplots()
# cs = plt.contourf(x,y,u.T)
# fig.colorbar(cs,  orientation='vertical')
# fig.show()


    









# Non-homogoneous:
source_factor  = (params['rho_B'][0]*params['G^0'][0]) / (params['sigma_1'][0] *params['X'][0] )
source =  source_factor * a_dot

################### GAS FLOW OPERATORS #################################





################### ENERGY EXCHANGE OPERATORS #################################

# Conduction Operator 
E_c = k_B * (fd.FinDiff(0, DX, 2)  + fd.FinDiff(1, DY, 2 )  ) * meter ** (-2)
# Liquid Flow Operator
E_L_factor = - params['q_L'][0] * params['rho_L'][0] * params['ASH_L'][0]  # liquid enthalpy flow per unit area per temperature difference. 
E_L =  E_L_factor * fd.FinDiff(1, DY, 1) * meter ** (-1) 
# Gas flow Operator
E_g_factor = params['G'][0]*(  params['ASH_G'][0]  + params['ASH_V'][0] * params['H_air'][0]  ) # -params['lambda'][0] * params['H_air'][0] 
E_g = E_g_factor * (fd.FinDiff(0, DX, 1) + fd.FinDiff(1, DY, 1) ) * meter ** (-1) 
# Heat of reaction (source function)
Delta_H_R = params['Delta_H_Ch'][0] + params['FPY'][0] * params['Delta_H_Py'][0]
E_source = - Delta_H_R * source
# Energy Exchange Operator
Ex = E_c -  E_L - E_g

################### ENERGY EXCHANGE BC's ######################################

bc = fd.BoundaryConditions(N)

# Assuming a rectangular heap: bottom insulated & top under irrigation 

## Dirichlet BC
bc[0,:] = params['T_atmos'][0] # left 
bc[-1,:] = params['T_atmos'][0] # right
bc[:,-1] = params['T_L'][0] # top ########## CHANGE TO T = T_L 

## Neumann BC
bc[:, 0] = fd.FinDiff(1, DY, 1), 0 # bottom
mid = round(N[0]/2) # middle_x
bc[mid, 1:-1] = fd.FinDiff(0, DX, 1), 0

################### ENERGY EXCHANGE PDE SOVLE #################################

# Solve
pde = fd.PDE(Ex.magnitude, E_source.magnitude, bc)
T = pde.solve()

# Plot
fig, ax = plt.subplots()
cs = plt.contourf(x,y,T.T)
fig.colorbar(cs,  orientation='vertical')
fig.show()


################### OXYGEN BALANCE OPERATORS #################################

# Diffusion Operator
Od_factor = params['eps_g'][0] * params['D_g'][0]
Od =  Od_factor *  (  fd.FinDiff(0, DX, 2)  + fd.FinDiff(1, DY, 2 )  ) * meter ** (-2)
# Convection Operator
Oc = params['eps_g'][0] * (fd.FinDiff(0, DX, 1) + fd.FinDiff(1, DY, 1) ) * meter ** (-1) 
# Oxygen Balance Operator
#Ob = Od - Oc 
















# Variable dictionary: a description of each variable used in the model
var_dict = {
    'alpha':'The product balance resulting from the combined chemical and bacterial reactions', 
    'alpha_dot':alpha_dot.__doc__,
    'C_L':'The concentration of liquid Oxygen',
    'Vm':Vm.__doc__
    }


def gas_flow():
    pass

def oxygen_balance():
    pass

def energy_exchange():
    pass









def slicer():
    '''slice heap along chosen access to get associated variables '''
    pass





def how_to_run():
    print('Run using script x... TBC')

# if __name__ == "__main__":
#     how_to_run()

