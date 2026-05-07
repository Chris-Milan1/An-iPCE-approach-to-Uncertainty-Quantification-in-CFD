""" This module serves as the implementation of Burgers' Equation Uncertainty Quantification where viscosity is random"""
import math as m 
import numpy as np 
from numba import njit
import matplotlib.pyplot as plt 
import pandas as pd
import time
from scipy.optimize import curve_fit

# Define the hermite polynomails on a recurrence relationship
def hermite_poly(n, x): 
    """
    Generate the Probabalist hermite polynomials using a recurrence relation
    
    :int n: Order of the polynomial
    :array x: Input variable
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    He_prev = 1
    He_curr = x
    for k in range(1, n):
        He_next = x * He_curr - k * He_prev
        He_prev, He_curr = He_curr, He_next
    return He_curr 

# Define a function to compute the nu expansion
def nu(mu, sigma, P, zeta):
    """
    Build the summation of viscosity ratio, known
    v0 = mu, v1 = sigma, vk = 0
    vhat = sum_{i=0}^P vi phi_i(zeta)
    
    mu: Accepted value for viscosity 
    sigma: Uncertainty in the model
    P: Order of Expansion
    zeta: Random variable (Germ)
    """
    # ln the mu because in the log normal transform, the mean is the log of the accepted value. This happens other wise the mean of the distribution is not the accepted value but rather the exponential of the accepted value
    v0 = np.log(mu)  - 0.5 * sigma ** 2
    v1 = sigma
    nu_1 = 0
    for i in range(P+1):
        if i == 0:
            nu_1+= v0 * hermite_poly(i, zeta)
        if i == 1:
            nu_1+= v1 * hermite_poly(i, zeta)
        if i > 1:
            nu_1+= 0 * hermite_poly(i, zeta)

    return nu_1

# Define a function to compute the z_k coefficient matrix
def z_k(viscosity_mu, sigma, P, zeta, w):

    # Matrix to store the results 
    z_matrix = np.zeros((P+1))

    # Get the nu expansion and take it to the exponential 
    nu_e = np.exp(nu(viscosity_mu, sigma, P, zeta))

    for k in range(P+1):

        # Numerator function
        num = np.sum(w * nu_e * hermite_poly(k, zeta))

        # Denominator function 
        den = np.sum(w * hermite_poly(k, zeta) ** 2)

        # Assign to matrix 
        z_matrix[k] = num/den
    
    return z_matrix

# Define a function to compute the C_ijk matrix 
def C_ijk(zeta, w, P):
    C = np.zeros((P+1, P+1, P+1))

    for k in range(P+1):
        # denom 
        den = np.sum(w* hermite_poly(k, zeta)**2)
        for i in range(P+1):
            for j in range(P+1):
                num = np.sum(w * hermite_poly(k, zeta) * hermite_poly(i, zeta) * hermite_poly(j, zeta))

                x = num/den

                # Append to matrix 
                C[i,j,k] = x
    
    return C

# Define a function to determine tensor dimensions
def dimensions(d, M):
    """    
    :int d: Number of random variables 
    :int M: Degree of Expansion 
    """
    P = int(m.factorial(d + M) / (m.factorial(M) * m.factorial(d)))
    return P

# Define a function for Gauss-Hermite Quadraure
def gauss_hermite_quad(n):
    """
    A function to compute the integral of the inner product of hermite polynomials using guass hermite quadrature
    
    n: The order of the quadrature, which will determine the number of points and weights used in the quadrature
    """
    # Get the roots and weights for the hermite polynomials 
    x, w =np.polynomial.hermite_e.hermegauss(n)
    return x, w

# Loss function vector of size (Nodes * (P+1)), using kron_delta_{k0} - sum_{ij} beta_i beta_recip_j C_ijk = 0 (beta and beta_recip how spatial nodes)
def loss_func(beta, beta_recip, C):
    
    # Solve for the second term 
    t2 = np.einsum('ijk,is,sj -> sk', C, beta, beta_recip)         

    # Perform the subtraction 1- t2 fpr k =0 and 0 - t2 for k>0 across all spatial nodes
    r = -t2
    r[:, 0] += 1

    return r

# Jacobian matrix of size (nodes, (P+1), (P+1)), J_{Kj} = sum_{i=0}^P beta_i C_ijk
def jacobian(beta, C):
    # Calculate the summation
    sum_jaco = - np.einsum('ijk, is -> skj', C, beta)
    return sum_jaco

# Each Spatial node must get its own optimized PCE coefficients
def newton_raphson(beta, guess, C):
    tol = 1e-6
    max_iter = 1000
    a = guess.copy()
    for _ in range(max_iter):
        grad = jacobian(beta, C)
        # Be sure to reshape the loss
        loss = loss_func(beta, a, C)
        if np.linalg.norm(loss) <tol:
            break
        a -= np.linalg.solve(grad, loss)
    return a

# Define a WENO function
def weno_pce(u_curr, u_right, u_2right, u_left, u_2left, C, P, spatial_nodes):
     
        # Compute the flux (f = U^2/2)
        f_x_minus_2 = 0.5 * np.einsum('ijk,is,js->ks', C, u_2left, u_2left)
        f_x_minus_1 = 0.5 * np.einsum('ijk,is,js->ks', C, u_left, u_left)
        f_x_i = 0.5 * np.einsum('ijk,is,js->ks', C, u_curr, u_curr)
        f_x_plus_1 = 0.5 * np.einsum('ijk,is,js->ks', C, u_right, u_right)
        f_x_plus_2 = 0.5 * np.einsum('ijk,is,js->ks', C, u_2right, u_2right)

        # f_k for each stencil, where k = 0, 1, 2
        f0 = 1/6 * (2*f_x_minus_2 - 7*f_x_minus_1 + 11*f_x_i)
        f1 = 1/6 * (- f_x_minus_1 + 5*f_x_i + 2*f_x_plus_1)
        f2 = 1/6 * (2*f_x_i + 5*f_x_plus_1 - f_x_plus_2)

        # Compute the smoothness for each substencil, β_k = (13/12) * (f_{j-2} - 2*f_{j-1} + f_j)**2 + (1/4) * (f_{j-2} - 4*f_{j-1} + 3*f_j)**2
        # Squaring the fluxes which are random, requires special attention. 
        # f_k = sum{i=0}^P sum{j=0}^P C_ijk u_i u_j, f_k^2 = sum{i=0}^P sum{j=0}^P sum{m=0}^P sum{n=0}^P C_ijk C_mnq u_i u_j u_m u_n, where q is the index of the flux after squaring
        # This is a 4th order tensor contraction, so einsum will be used, with a final shape of (P+1, spatial_nodes)

        # B0 expanded: (4/3)*f_x_minus_2**2 - (19/3)*f_x_minus_2*f_x_minus_1 + (11/3)*f_x_minus_2*f_x_i + (25/3)*f_x_minus_1**2 - (31/3)*f_x_minus_1*f_x_i + (10/3)*f_x_i**2
        # B1 expanded: (4/3)*f_x_minus_1**2 - (13/3)*f_x_minus_1*f_x_i + (5/3)*f_x_minus_1*f_x_plus_1 + (13/3)*f_x_i**2 - (13/3)*f_x_i*f_x_plus_1 + (4/3)*f_x_plus_1**2
        # B2 expanded: (10/3)*f_x_i**2 - (31/3)*f_x_i*f_x_plus_1 + (11/3)*f_x_i*f_x_plus_2 + (25/3)*f_x_plus_1**2 - (19/3)*f_x_plus_1*f_x_plus_2 + (4/3)*f_x_plus_2**2

        # 2 random fluxes will need to be mutiplied together as well, this is similar to squareing, in fact squaring is a special case of this wher the 2 fluxes are the same. 
        # The muliplication will be done using the same method as above, but with different C_ijk tensors, which we will call D_ijkl = sum{q=0}^P C_ijq C_klq / sum{q=0}^P C_0jq C_0jq, where the denominator is a normalisation factor to ensure the weights sum to 1. 
        # This will give us a new tensor D_ijkl which can be used to compute the product of 2 fluxes, f_k * f_m = sum{i=0}^P sum{j=0}^P sum{k=0}^P sum{l=0}^P D_ijkl u_i u_j u_k u_l 

        # Compute the smoothness for each substencil
        # Beta 0 
        B0 = (4/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_2, f_x_minus_2) - (19/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_2, f_x_minus_1) + (11/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_2, f_x_i) + (25/3)*np.einsum('ijk,is,js->ks', C, f_x_minus_1, f_x_minus_1) - (31/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_1, f_x_i) + (10/3) * np.einsum('ijk,is,js->ks', C, f_x_i, f_x_i)
        B1 = (4/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_1, f_x_minus_1) - (13/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_1, f_x_i) + (5/3) * np.einsum('ijk,is,js->ks', C, f_x_minus_1, f_x_plus_1) + (13/3)*np.einsum('ijk,is,js->ks', C, f_x_i, f_x_i) - (13/3) * np.einsum('ijk,is,js->ks', C, f_x_i, f_x_plus_1) + (4/3) * np.einsum('ijk,is,js->ks', C, f_x_plus_1, f_x_plus_1)
        B2 = (10/3) * np.einsum('ijk,is,js->ks', C, f_x_i, f_x_i) - (31/3) * np.einsum('ijk,is,js->ks', C, f_x_i, f_x_plus_1) + (11/3) * np.einsum('ijk,is,js->ks', C, f_x_i, f_x_plus_2) + (25/3)*np.einsum('ijk,is,js->ks', C, f_x_plus_1, f_x_plus_1) - (19/3) * np.einsum('ijk,is,js->ks', C, f_x_plus_1, f_x_plus_2) + (4/3) * np.einsum('ijk,is,js->ks', C, f_x_plus_2, f_x_plus_2)


        # Compute the nonlinear weights, a_k = d_k /(e + b_k)^p
        # The big issue here is that the smoothness beta's are random variables, so the weights will be random. 
        # Also a betas are PCE coefficients they cannot be used directly for division.
        # There are 2 solutions to the division issue both require a transformation. 
        # 1. A Galerkin projection to find a new set of coefficients that represent the reciprocal of the smoothness, done by multiplying by hermite polynomials and integrating using gauss quad
        # 2. An optimization problem to find the coefficients that minimise the difference between the original smoothness and the smoothness reconstructed from the new PCES, 
        # The Newton-Raphson method will be used, to find the coefficients that satify teh reicprocal relationship. 
    
        d0 = 0.1
        d1 = 0.6
        d2 = 0.3 
        p = 2
        e = 10**-6        

        # Set the first coefficient, a[0] = 1/(mean(beta) + e)** p and the rest to 0 (a0 is an array of size P+1, spatial nodes)
        coeff =  1/((np.mean(B0, axis = 1))**p)
        a0 = np.zeros((spatial_nodes, P+1))
        a0[0:] = coeff

        a1 =  np.zeros((spatial_nodes, P+1))
        a1[0:] = 1/((np.mean(B1, axis = 1))**p)

        a2 =  np.zeros((spatial_nodes, P+1))
        a2[0:] = 1/((np.mean(B2, axis = 1))**p)

        
        B0_recip = newton_raphson(B0, a0, C)
        B1_recip = newton_raphson(B1, a1, C)
        B2_recip = newton_raphson(B2, a2, C)

        # Weights 
        aw_0 = d0 * B0_recip
        aw_0 = np.transpose(aw_0)

        aw_1 = d1 * B1_recip
        aw_1 = np.transpose(aw_1)

        aw_2 = d2 * B2_recip
        aw_2 = np.transpose(aw_2)
        

        # Normalise the weights w = a_k / sum_{l=0}^2 a_l
        a_sum = aw_0 + aw_1 + aw_2

        # Due to the division the same newton raphson method will need to be implemenated again to find the reciprocal 
        guess = np.zeros((spatial_nodes, P+1))
        
        guess[:, 0] = 1.0/np.mean(a_sum)
        a_sum_recip = newton_raphson(a_sum, guess, C)
        a_sum_recip = np.transpose(a_sum_recip)


        # To find the weights a0 and asum_recip must be mutlipled as both are pCEs, the require einsum for each of the stencils
        w0 = np.einsum('ijk,is,js->ks', C, aw_0, a_sum_recip)
        w1 = np.einsum('ijk,is,js->ks', C, aw_1, a_sum_recip)
        w2 = np.einsum('ijk,is,js->ks', C, aw_2, a_sum_recip)
    

        # Compute the final flux (f_{j+1/2})
        # Once again, the flux and weights are pces so einsum is required 

        wf0 = np.einsum('ijk,is,js->ks', C, w0, f0)
        wf1 = np.einsum('ijk,is,js->ks', C, w1, f1)
        wf2 = np.einsum('ijk,is,js->ks', C, w2, f2)


        f_i_plus_half = wf0 + wf1 + wf2

        # compute the flux at left interface
        f_i_minus_half = np.roll(f_i_plus_half, 1, axis = 1)

        return f_i_plus_half, f_i_minus_half

# Define a function to solve the FVM
def FVM(U, delta_x, delta_t, timesteps, C, z, P, zeta, spatial_nodes):
       
    # Main simulation solver loop
    # n is the time step
    for n in range(timesteps-1):
        # print(n)
        if n % (timesteps // 10) == 0:
         print(n)
        # Extract the current state and neighbors
        u_curr = U[:, :, n]
        
        # This handles the periodic boundary conditions manually
        u_right = np.empty_like(u_curr)
        u_right[:, :-1] = u_curr[:, 1:]
        u_right[:, -1]  = u_curr[:, 0]

        # Shift 2 right
        u_2right = np.empty_like(u_curr)
        u_2right[:, :-2] = u_curr[:, 2:]
        u_2right[:, -2:] = u_curr[:, :2]

        # Shift left (axis=1, +1)
        u_left = np.empty_like(u_curr)
        u_left[:, 1:] = u_curr[:, :-1]
        u_left[:, 0]  = u_curr[:, -1]

        # Shift 2 left
        u_2left = np.empty_like(u_curr)
        u_2left[:, 2:] = u_curr[:, :-2]
        u_2left[:, :2] = u_curr[:, -2:]

        f_i_plus_half, f_i_minus_half = weno_pce(u_curr, u_right, u_2right, u_left, u_2left, C, P, spatial_nodes)

        # Apply second order Runge Kutta method
        
        # Stage 1:
        t1 = u_curr

        # Diffusion
        u_s = u_right - 2*u_curr + u_left
        C_z = np.einsum('ijk, i, js-> ks', C, z, u_s)
        diff = (1/(delta_x**2)) * C_z 

        l = -(1/delta_x) * (f_i_plus_half - f_i_minus_half) + diff
        first_u = t1 + delta_t * l
        
        
        # Calculate stage 2
        # Extract the current state and neighbors
        u_curr_1 = first_u
        
        # This handles the periodic boundary conditions manually
        u_right_1 = np.empty_like(u_curr_1)
        u_right_1[:, :-1] = u_curr_1[:, 1:]
        u_right_1[:, -1]  = u_curr_1[:, 0]

        # Shift 2 right
        u_2right_1 = np.empty_like(u_curr_1)
        u_2right_1[:, :-2] = u_curr_1[:, 2:]
        u_2right_1[:, -2:] = u_curr_1[:, :2]

        # Shift left (axis=1, +1)
        u_left_1 = np.empty_like(u_curr_1)
        u_left_1[:, 1:] = u_curr_1[:, :-1]
        u_left_1[:, 0]  = u_curr_1[:, -1]

        # Shift 2 left
        u_2left_1 = np.empty_like(u_curr_1)
        u_2left_1[:, 2:] = u_curr_1[:, :-2]
        u_2left_1[:, :2] = u_curr_1[:, -2:]

        f_i_plus_half, f_i_minus_half = weno_pce(u_curr_1, u_right_1, u_2right_1, u_left_1, u_2left_1, C, P, spatial_nodes)

        t1 = u_curr

        # Diffusion
        u_s = u_right_1 - 2*u_curr_1 + u_left_1
        C_z = np.einsum('ijk, i, js-> ks', C, z, u_s)
        diff = (1/(delta_x**2)) * C_z 

        l = -(1/delta_x) * (f_i_plus_half - f_i_minus_half) + diff
        second_u = t1 + delta_t * l


        # Total
        u_k_e_1 = 0.5 * u_curr + 0.5 * second_u

        # Append to U
        U[:, :, n+1] = u_k_e_1
    
    return U

# Define a weno function for monte carlo
@njit
def weno_mc(u_curr, u_right, u_2right, u_left, u_2left):
    # Compute the flux (f = U^2/2)
    f_x_minus_2 = (u_2left**2)/2
    f_x_minus_1 = (u_left**2)/2
    f_x_i = (u_curr**2)/2
    f_x_plus_1 = (u_right**2)/2
    f_x_plus_2 = (u_2right**2)/2

    # f_k for each stencil, where k = 0, 1, 2
    f0 = 1/6 * (2*f_x_minus_2 - 7*f_x_minus_1 + 11*f_x_i)
    f1 = 1/6 * (- f_x_minus_1 + 5*f_x_i + 2*f_x_plus_1)
    f2 = 1/6 * (2*f_x_i + 5*f_x_plus_1 - f_x_plus_2)

    # Compute the smoothness for each substencil, β_k = (13/12) * (f_{j-2} - 2*f_{j-1} + f_j)**2 + (1/4) * (f_{j-2} - 4*f_{j-1} + 3*f_j)**2
    beta_0 = (13/12) * (f_x_minus_2 - 2*f_x_minus_1 + f_x_i)**2 + (1/4) * (f_x_minus_2 - 4*f_x_minus_1 + 3*f_x_i)**2
    beta_1 = (13/12) * (f_x_minus_1 - 2*f_x_i + f_x_plus_1)**2 + (1/4) * (f_x_minus_1 - f_x_plus_1)**2
    beta_2 = (13/12) * (f_x_i - 2*f_x_plus_1 + f_x_plus_2)**2 + (1/4) * (3*f_x_i - 4*f_x_plus_1 + f_x_plus_2)**2

    # Compute the nonlinear weights, a_k = d_k /(e + b_k)^p
    d0 = 0.1
    d1 = 0.6
    d2 = 0.3 
    p = 2
    e = 10**-6
    a0 = d0/(e + beta_0)**p
    a1 = d1/(e + beta_1)**p
    a2 = d2/(e + beta_2)**p

    # Normalise the weights w = a_k / sum_{l=0}^2 a_l
    a_sum = a0 + a1 + a2
    w0 = a0/a_sum
    w1 = a1/a_sum
    w2 = a2/a_sum

    # Compute the final flux (f_{j+1/2})
    f_i_plus_half = w0*f0 + w1*f1 + w2*f2

    # compute the flux at left interface
    f_i_minus_half = np.empty_like(f_i_plus_half)
    f_i_minus_half[0] = f_i_plus_half[-1]
    f_i_minus_half[1:] = f_i_plus_half[:-1]
    return f_i_plus_half, f_i_minus_half

# Define a support solution function for the monte carlo
@njit
def solution(viscosity, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude):
    #  Initialize Matrix
    U = np.zeros((spatial_nodes, timesteps))
    
    # Define delta_x and x properly for Numba
    delta_x = length / spatial_nodes
    
    # We create the x array so that x[i] is defined for the loop below
    x = np.arange(spatial_nodes) * delta_x
    
    # Initial Condition
    for i in range(spatial_nodes):
        U[i, 0] = mean_flow + amplitude * np.sin(x[i]) 
    
    delta_t = sim_time / timesteps

    # Implememant the cfl condition

    for n in range(timesteps - 1):
        u_curr = U[:, n]
        
        # This handles the periodic boundary conditions manually
        u_right = np.empty_like(u_curr)
        u_right[:-1] = u_curr[1:]
        u_right[-1]  = u_curr[0]
        
        u_2right = np.empty_like(u_curr)
        u_2right[:-2] = u_curr[2:]
        u_2right[-2:] = u_curr[:2]
        
        u_left = np.empty_like(u_curr)
        u_left[1:] = u_curr[:-1]
        u_left[0]  = u_curr[-1]
        
        u_2left = np.empty_like(u_curr)
        u_2left[2:] = u_curr[:-2]
        u_2left[:2] = u_curr[-2:]

        # There are 3 stencils:
        # S0 = {X_i-2, X_i-1, X_i}
        # S1 = {X_i-1, X_i, X_i+1}
        # S2 = {X_i, X_i+1, X_i+2}

        f_i_plus_half, f_i_minus_half = weno_mc(u_curr, u_right, u_2right, u_left, u_2left)

        diff = (viscosity)/(delta_x**2) * (u_right - 2*u_curr + u_left)

        # Apply second order Runge Kutta method
        
        # Stage 1:
        t1 = u_curr

        l = -(1/delta_x) * (f_i_plus_half - f_i_minus_half) + diff
       
        first_u = t1 + delta_t * l
        
        
        # Calculate stage 2
        # Extract the current state and neighbors
        u_curr_1 = first_u
        
        # This handles the periodic boundary conditions manually
        u_right_1 = np.empty_like(u_curr_1)
        u_right_1[:-1] = u_curr_1[1:]
        u_right_1[-1]  = u_curr_1[0]
        
        u_2right_1 = np.empty_like(u_curr_1)
        u_2right_1[:-2] = u_curr_1[2:]
        u_2right_1[-2:] = u_curr_1[:2]
        
        u_left_1 = np.empty_like(u_curr_1)
        u_left_1[1:] = u_curr_1[:-1]
        u_left_1[0]  = u_curr_1[-1]
        
        u_2left_1 = np.empty_like(u_curr_1)
        u_2left_1[2:] = u_curr_1[:-2]
        u_2left_1[:2] = u_curr_1[-2:]

        f_i_plus_half, f_i_minus_half = weno_mc(u_curr_1, u_right_1, u_2right_1, u_left_1, u_2left_1)

        diff = (viscosity)/(delta_x**2) * (u_right_1 - 2*u_curr_1 + u_left_1)
        t1 = u_curr
        l = -(1/delta_x) * (f_i_plus_half - f_i_minus_half) + diff
        second_u = t1 + delta_t * l


        # Total
        u_next = 0.5 * u_curr + 0.5 * second_u

        U[:, n+1] = u_next

    return U

# Define a function to run a Monte Carlo simulation
def monte_carlo(mu, sigma, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, n_simulations, plot_norm = False):
    #sigma = sigma*mu    
    if plot_norm == True:
        # Plot the normal distribution 
        x_visc = np.linspace(0, 2*mu, 1000)
        pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_visc - mu)/sigma)**2)
        plt.figure()
        plt.plot(x_visc, pdf, label='PDF')
        plt.axvline(mu, color='r', linestyle='--', label = 'Mean')
        plt.xlabel('Viscosity')
        plt.ylabel('Density')
        plt.title('Distribution of Viscosity')
        plt.legend()
        plt.grid()

    # Generate samples
    solution_store = np.zeros((int(n_simulations), (spatial_nodes), timesteps))

    mu_log = np.log(mu) - 0.5 * sigma**2
    sigma_log = sigma

    viscosity_samples = np.random.lognormal(mu_log, sigma_log, int(n_simulations))

    # Begin the Monte carlo loop 
    for i, nu in enumerate(viscosity_samples):
        
        print('iteration', i)
        sol = solution(nu,length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude)
        solution_store[i] = sol
    
    # CONVERGENCE (relative error)
    # Initialise arrays to hold all information
    x = np.zeros(solution_store.shape[1:])
    x_sq = np.zeros(solution_store.shape[1:])
    errors = np.zeros(n_simulations)

    for i in range(n_simulations):
        curr_mean = solution_store[i]

        x = x + curr_mean
        x_sq = x_sq + curr_mean**2

        # Calculate the error 
        n = i + 1
        if n > 1:
            mean = x/n
            mean_sq = x_sq/n

            var = np.maximum(mean_sq - mean**2, 0)
            sem = np.sqrt(var / n)
            error = np.linalg.norm(sem) / np.linalg.norm(curr_mean)
            errors[i] = error

    df = pd.DataFrame({
        'Simulation_Number': np.arange(len(errors)),
        'Result': errors
    })

    print(df.head())

    # Save the results to a csv file 
    df.to_excel("Monte_results.xlsx", index=False)
    print("Monte results saved to excel")
    
    return solution_store, viscosity_samples

# Define a function to plot the results 
def post_process_dynamic(solution_pce, solution_mc, order, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, P, 
                         plots_to_show=None):
    """
    Generates static plots at every 10% of the total timesteps and saves them as PDFs.
    Ensures all axes are labeled with Quantity and Units.
    """
    if plots_to_show is None:
        plots_to_show = [1, 2, 3, 4, 5, 6]   
    
    t_vals = np.linspace(0, sim_time, timesteps)
    x = np.linspace(0, 2*np.pi, spatial_nodes, endpoint=False)
    fact_weights = np.array([m.factorial(i) for i in range(1, order + 1)]).reshape(-1, 1)
    
    # Define the indices for 0%, 10%, ..., 100%
    #snapshot_indices = [timesteps - 1]

    snapshot_indices = np.linspace(0, timesteps - 1, 11, dtype=int)

    # --- 2. Snapshot Loop ---
    for frame in snapshot_indices:
        num_plots = len(plots_to_show)
        if num_plots == 0: break
        
        cols = min(num_plots, 3) if num_plots != 4 else 2
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
        plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.1, top=0.9)
        ax_flat = axes.flatten()
        
        for i in range(num_plots, len(ax_flat)):
            ax_flat[i].axis('off')

        # --- 3. Data Calculation ---
        eps = 1e-10
        m_pce = solution_pce[0, :, frame]
        m_mc = np.mean(solution_mc[:, :, frame], axis=0)
        v_pce = np.maximum(np.sum(solution_pce[1:order+1, :, frame]**2 * fact_weights, axis=0), 0)
        v_mc = np.var(solution_mc[:, :, frame], axis=0)
        std_pce = np.sqrt(v_pce)

        # --- 4. Plotting Logic ---
        for idx, plot_num in enumerate(plots_to_show):
            ax = ax_flat[idx]
            # Unified X-Axis Label
            ax.set_xlabel("Spatial Coordinate $x$ ($m$)")
            ax.grid(True, which="both", alpha=0.3)
            
            if plot_num == 1:
                ax.plot(x, m_pce, 'b-', lw=2, label='PCE Mean')
                ax.plot(x, m_mc, 'r--', lw=1.5, label='MC Mean')
                ax.fill_between(x, m_pce - std_pce, m_pce + std_pce, color='gray', alpha=0.2, label='PCE ± std')
                ax.set_ylim(mean_flow - amplitude*2, mean_flow + amplitude*2)
                ax.set_title("Mean Velocity & Uncertainty")
                ax.set_ylabel("Velocity $u$ ($m/s$)")
                
            elif plot_num == 2:
                ax.plot(x, v_pce, 'b-', lw=2, label='PCE Var')
                ax.plot(x, v_mc, 'k--', lw=1.5, label='MC Var')
                ax.set_ylim(0, max(v_pce.max(), v_mc.max()) * 1.1 + eps)
                ax.set_title("Variance Comparison")
                ax.set_ylabel("Variance $\sigma^2$ ($(m/s)^2$)")
                
            elif plot_num == 3:
                err = np.nan_to_num(np.abs(m_mc - m_pce) / (np.abs(m_mc) + eps) * 100)
                ax.plot(x, err, 'g-', lw=1.5, label='% Error Mean')
                ax.set_ylim(0, err.max() * 1.1 + eps)
                ax.set_title("Mean Percentage Error")
                ax.set_ylabel("Relative Error (%)")
                
            elif plot_num == 4:
                ax.plot(x, m_pce, 'b-', lw=2, label='PCE Mean')
                ax.plot(x, m_mc, 'r--', lw=1.5, label='MC Mean')
                ax_twin = ax.twinx()
                ax_twin.plot(x, v_pce, 'c-', lw=2, label='PCE Var')
                ax_twin.plot(x, v_mc, 'k--', lw=1.5, label='MC Var')
                ax.set_ylim(mean_flow - amplitude*2, mean_flow + amplitude*2)
                ax_twin.set_ylim(0, max(v_pce.max(), v_mc.max()) * 1.1 + eps)
                ax.set_title("Mean & Variance Dual Axis")
                ax.set_ylabel("Velocity $u$ ($m/s$)")
                ax_twin.set_ylabel("Variance $\sigma^2$ ($(m/s)^2$)")
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax_twin.get_legend_handles_labels()
                ax_twin.legend(h1+h2, l1+l2, loc='upper right', fontsize='x-small')
                
            elif plot_num == 5:
                v_err = np.nan_to_num(np.abs(v_mc - v_pce) / (v_mc + eps) * 100)
                ax.plot(x, v_err, 'm-', lw=1.5, label='% Error Var')
                ax.set_ylim(0, v_err.max() * 1.1 + eps)
                ax.set_title("Variance Percentage Error")
                ax.set_ylabel("Relative Error (%)")
                
            elif plot_num == 6:
                cv = std_pce / (m_pce + eps)
                ax.plot(x, cv, 'darkorange', lw=2, label='Coeff. of Variation')
                ax.set_ylim(0, max(cv.max() * 1.1, 1.1))
                ax.set_title("Coeff. of Variation")
                ax.set_ylabel("CV (unitless)")

            if plot_num != 4:
                ax.legend(loc='upper right', fontsize='small')

        percentage = int((frame/(timesteps-1))*100)
        fig.suptitle(f"Snapshot at Time: {t_vals[frame]:.2f}s ({percentage}%) | PCE Order: {order}", fontsize=14)
        
        fig.savefig(f"snapshot_{percentage:03d}pct.pdf", bbox_inches='tight')
        plt.close(fig) 
    
    # --- 5. [SPATIO-TEMPORAL HEATMAPS - INDIVIDUAL PLOTS] ---
    # 1. Calculate Means
    mean_evolution_pce = solution_pce[0, :, :]
    mean_evolution_mc = np.mean(solution_mc, axis=0) 

    # 2. Calculate Variances and Standard Deviation
    fact_weights_3d = fact_weights.reshape(-1, 1, 1)
    var_evolution_pce = np.maximum(np.sum(solution_pce[1:order+1, :, :]**2 * fact_weights_3d, axis=0), 0)
    std_evolution_pce = np.sqrt(var_evolution_pce)
    var_evolution_mc = np.var(solution_mc, axis=0)

    # 3. Calculate Variance Error
    var_error_evolution = np.abs(var_evolution_pce - var_evolution_mc)

    # Shared color limits for the mean plots
    v_min_mean = min(mean_evolution_pce.min(), mean_evolution_mc.min())
    v_max_mean = max(mean_evolution_pce.max(), mean_evolution_mc.max())

    # --- Plot 1: PCE Mean ---
    plt.figure(figsize=(10, 8))
    cp1 = plt.pcolormesh(x, t_vals, mean_evolution_pce.T, shading='gouraud', cmap='viridis', vmin=v_min_mean, vmax=v_max_mean)
    plt.colorbar(cp1, label='Mean Velocity $u$ ($m/s$)')
    plt.title(f"PCE Mean Evolution (Order {order})", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Coordinate $x$ ($m$)", fontsize=12)
    plt.ylabel("Time $t$ ($s$)", fontsize=12)
    plt.savefig("heatmap_pce_mean.pdf", bbox_inches='tight')
    plt.close()

    # --- Plot 2: MC Mean ---
    plt.figure(figsize=(10, 8))
    cp2 = plt.pcolormesh(x, t_vals, mean_evolution_mc.T, shading='gouraud', cmap='viridis', vmin=v_min_mean, vmax=v_max_mean)
    plt.colorbar(cp2, label='Mean Velocity $u$ ($m/s$)')
    plt.title("MC Mean Evolution", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Coordinate $x$ ($m$)", fontsize=12)
    plt.ylabel("Time $t$ ($s$)", fontsize=12)
    plt.savefig("heatmap_mc_mean.pdf", bbox_inches='tight')
    plt.close()

    # --- Plot 3: PCE Standard Deviation ---
    plt.figure(figsize=(10, 8))
    cp3 = plt.pcolormesh(x, t_vals, std_evolution_pce.T, shading='gouraud', cmap='plasma')
    plt.colorbar(cp3, label='Standard Deviation $\sigma$ ($m/s$)')
    plt.title("PCE Standard Deviation", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Coordinate $x$ ($m$)", fontsize=12)
    plt.ylabel("Time $t$ ($s$)", fontsize=12)
    plt.savefig("heatmap_pce_std_dev.pdf", bbox_inches='tight')
    plt.close()

    # --- Plot 4: Variance Error ---
    plt.figure(figsize=(10, 8))
    cp4 = plt.pcolormesh(x, t_vals, var_error_evolution.T, shading='gouraud', cmap='plasma')
    plt.colorbar(cp4, label='Variance Error $\sigma^2$ ($(m/s)^2$)')
    plt.title("Absolute Error |PCE Var - MC Var|", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Coordinate $x$ ($m$)", fontsize=12)
    plt.ylabel("Time $t$ ($s$)", fontsize=12)
    plt.savefig("heatmap_variance_error.pdf", bbox_inches='tight')
    plt.close()

# Define a function to perform a convergence test on the PCE 
def convergence_pce(mu, sigma, orders, length, spatial_nodes, sim_time, timesteps,
                    mean_flow, amplitude, n_simulations, t_point, cfl, scale):

    # STORAGE
    mean_solutions = np.zeros((len(orders), spatial_nodes, timesteps))

    x = np.linspace(0, length, spatial_nodes, endpoint=False)

    # SOLVE FOR ALL ORDERS
    for idx, ord in enumerate(orders):

        print("Order", ord)

        start_time = time.perf_counter() 

        P = dimensions(1, ord) - 1

        delta_x = length / spatial_nodes
        delta_t = sim_time / timesteps

        zeta, w = gauss_hermite_quad(20 * P)

        sol = np.zeros((P + 1, spatial_nodes, timesteps))

        # initial condition (only mean mode)
        for i in range(spatial_nodes):
            sol[0, i, 0] = mean_flow + amplitude * np.sin(x[i])

        current_sol = FVM(
            sol, delta_x, delta_t, timesteps,
            C_ijk(zeta, w, P),
            z_k(mu, sigma, P, zeta, w),
            P, zeta, spatial_nodes
        )

        end_time = time.perf_counter()

        # store mean field
        mean_solutions[idx] = current_sol[0]

    # CONVERGENCE 
    ref_mean = mean_solutions[-1]

    errors = {}

    for i, ord in enumerate(orders[:-1]):

        curr_mean = mean_solutions[i]

        rel_error = np.linalg.norm(curr_mean - ref_mean) / np.linalg.norm(ref_mean)

        errors[ord] = rel_error

    orders_list = np.array(sorted(errors.keys()))
    
    # Ensure we are taking the norm here so we are plotting and comparing scalars
    error_values = np.array([float(np.linalg.norm(errors[o])) for o in orders_list])

    df = pd.DataFrame({
        'Simulation_Number': np.arange(len(error_values)),
        'Result': error_values
    })

    # Save the results to a csv file 
    df.to_excel("PCE_conv.xlsx", index=False)
    print("PCE Conv saved to excel")

    return errors

# Define a function for the workflow
def solve(mu, sigma, order, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, n_simulations, t_point, cfl, scale, optimise=False):
    
    # PCE Simulation Timing Start 
    start_pce = time.perf_counter()

    P = dimensions(1, order) - 1
    delta_x = length/spatial_nodes
    delta_t = sim_time/timesteps
    
    zeta, w = gauss_hermite_quad(20*P)

    # CFL conditions
    worst_u = mean_flow + amplitude
    limit_adv = (cfl * delta_x)/worst_u
    upper_nu = np.exp(np.log(mu) + scale*sigma)
    limit_diff = (cfl * delta_x ** 2) / (2 * upper_nu)

    while delta_t > limit_adv or delta_t > limit_diff:
        timesteps += 10
        delta_t = sim_time/timesteps

    sol = np.zeros((P+1, spatial_nodes, timesteps))
    z = z_k(mu, sigma, P, zeta, w)
    C = C_ijk(zeta, w, P)

    x = np.linspace(0, length, spatial_nodes, endpoint=False)
    for k in range(P+1):
        for e in range(spatial_nodes):
            sol[k, e, 0] = mean_flow * (1 if k == 0 else 0) + amplitude * np.sin(x[e]) * (1 if k == 0 else 0)

    solution_pce = FVM(sol, delta_x, delta_t, timesteps, C, z, P, zeta, spatial_nodes)
    
    end_pce = time.perf_counter()
    times_pce = end_pce - start_pce
    #  PCE Simulation Timing End 

    if optimise:
        start_mc = time.perf_counter()
        solution_mc, viscosity_samples = monte_carlo(mu, sigma, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, n_simulations, plot_norm=False)
        times_mc = time.perf_counter() - start_mc
        return solution_pce, solution_mc, times_pce, times_mc
    
    verify = input('Would you like to verify the PCE results with a Monte Carlo simulation ? (y/n)')
    if verify == 'n':
        # Returning 0 or None for MC time if not run
        return solution_pce, None, times_pce, 0
    
    if verify == 'y':
        print('Performing Monte Carlo Simulation')
        # MC Simulation Timing Start 
        start_mc = time.perf_counter()
        solution_mc, viscosity_samples = monte_carlo(mu, sigma, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, n_simulations, plot_norm=False)
        times_mc = time.perf_counter() - start_mc
        # MC Simulation Timing End 
        
        print(f"MC Shape: {solution_mc.shape}")
        # 1 = mean, 2 = variance comparision, 3 = mean % error, 4= mean and var dual axis, 5 = variance % error, 6 = coeff
        #post_process_dynamic(solution_pce, solution_mc, P, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, P, plots_to_show=[4])
        
        return solution_pce, solution_mc, times_pce, times_mc
    
# Define a function to enforce the CFL condition
def CFL(length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, cfl, mu, scale):
    delta_x = length/spatial_nodes
    delta_t = sim_time/timesteps

    # Implementant a cfl condition, based on the worst case 
    # Advection limit 
    worst_u = mean_flow + amplitude
    limit_adv = (cfl * delta_x)/worst_u

    # Diffusion limit
    upper_nu = np.exp(np.log(mu) + scale*sigma)
    limit_diff = (cfl * delta_x ** 2) / (2 * upper_nu)

    # Check and update
    while delta_t > limit_adv or delta_t > limit_diff:
        timesteps += 10
        delta_t = sim_time/timesteps
    return timesteps

mu = 1.32e-6
sigma = 0.99
order = 2
length = m.pi
spatial_nodes = 100
sim_time = 5
timesteps = 100
mean_flow = 6
amplitude = 3
n_simulations = 90000
t_point = 100
cfl = 1 # adjust for numerical stability
scale = 2


mode = input("Choose mode: (g)eneral, (p)ce convergence").lower()

# CFL stability
timesteps = CFL(length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, cfl, mu, scale)

# ================================
#  GENERAL SIMULATION
# ================================
if mode == 'g':
    pce, mc, times_pce, times_mc = solve(mu, sigma, order, length, spatial_nodes, sim_time, timesteps,
          mean_flow, amplitude, n_simulations, t_point, cfl, scale, optimise=False)

    print('PCE times', times_pce)
    print('MC times', times_mc)
# ================================
# PCE CONVERGENCE
# ================================
elif mode == 'p':
    orders = [1, 2, 3,4,5,6,7,8,9]
    errors = convergence_pce(mu, sigma, orders, length, spatial_nodes, sim_time, timesteps, mean_flow, amplitude, n_simulations, t_point, cfl, scale)
