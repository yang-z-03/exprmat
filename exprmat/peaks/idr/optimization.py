
import os, sys
from scipy.optimize import fminbound
import numpy
import math

from exprmat.peaks.idr.constants import *
from exprmat.ansi import info
from exprmat.peaks.idr.utils import (
    simulate_values,
    compute_pseudo_values, 
    calc_post_membership_prbs, 
    calc_gaussian_mix_log_lhd,
    calc_gaussian_mix_log_lhd_gradient
)


def log_lhd_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p)
    z2 = compute_pseudo_values(r2, mu, sigma, p)
    return -calc_gaussian_mix_log_lhd(theta, z1, z2)


# set the loss function to log_lhd
calc_loss = log_lhd_loss


def EM_step(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    ez = calc_post_membership_prbs(starting_point, z1, z2)
    
    # just a small optimization
    ez_sum = ez.sum()
        
    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2
    
    weighted_sum_sqs_1 = (ez*((z1-mu)**2)).sum()
    weighted_sum_sqs_2 = (ez*((z2-mu)**2)).sum()
    weighted_sum_prod = (ez*(z2-mu)*(z1-mu)).sum()

    sigma = math.sqrt((weighted_sum_sqs_1+weighted_sum_sqs_2)/(2*ez_sum))
    
    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)

    p = ez_sum/len(ez)
    
    return numpy.array([mu, sigma, rho, p])


def grid_search(r1, r2):
    res = []
    best_theta = None
    max_log_lhd = -1e100
    for mu in numpy.linspace(0.1, 5, num=10):
        for sigma in numpy.linspace(0.5, 3, num=10):
            for rho in numpy.linspace(0.1, 0.9, num=10):
                for pi in numpy.linspace(0.1, 0.9, num=10):
                    z1 = compute_pseudo_values(r1, mu, sigma, pi)
                    z2 = compute_pseudo_values(r2, mu, sigma, pi)
                    log_lhd = calc_gaussian_mix_log_lhd((mu, sigma, rho, pi), z1, z2)
                    if log_lhd > max_log_lhd:
                        best_theta = ((mu,mu), (sigma,sigma), rho, pi)
                        max_log_lhd = log_lhd
    
    return best_theta


def find_max_step_size(param_val, grad_val, limit_to_1 = False, MIN_VAL = 1e-6):
    if grad_val < 0 and param_val < MIN_VAL: return 0
    if limit_to_1 and grad_val > 0 and param_val > MIN_VAL: return 0
    
    max_alpha = 10
    if grad_val > 1e-6:
        max_alpha = min(max_alpha, (param_val - MIN_VAL) / grad_val)
    elif grad_val < -1e-6:
        max_alpha = min(max_alpha, (MIN_VAL-param_val) / grad_val)

    if limit_to_1:
        if grad_val > 1e-6:
            max_alpha = min(max_alpha, (1 - param_val - MIN_VAL) / grad_val)
        elif grad_val < -1e-6:
            max_alpha = min(max_alpha, (param_val + MIN_VAL - 1) / grad_val)

    return max_alpha    


def coordinate_ascent(
    r1, r2, theta, gradient_magnitude, 
    fix_mu = False, fix_sigma = False
):
    for j in range(len(theta)):
        if fix_mu and j == 0: continue
        if fix_sigma and j == 1: continue
    
        prev_loss = calc_loss(r1, r2, theta)
        # find the direction of the gradient
        gradient = numpy.zeros(len(theta))
        gradient[j] = gradient_magnitude
        init_alpha = 5e-12
        while init_alpha < 1e-2:
            pos = calc_loss( r1, r2, theta - init_alpha*gradient )
            neg = calc_loss( r1, r2, theta + init_alpha*gradient )
            if neg < prev_loss < pos:
                gradient[j] = gradient[j]
                break
            elif neg > prev_loss > pos:
                gradient[j] = -gradient[j]
                break
            else: init_alpha *= 10         

        assert init_alpha < 1e-1
        
        min_step = 0
        max_step = find_max_step_size(
            theta[j], gradient[j], (False if j in (0,1) else True))

        if max_step < 1e-12: continue

        alpha = fminbound(
            lambda x: calc_loss( r1, r2, theta + x*gradient ),
            min_step, max_step)
        
        loss = calc_loss( r1, r2, theta + alpha*gradient )
        if loss < prev_loss: theta += alpha * gradient

    return theta


def clip_model_params(init_theta):
    theta_changed = False
    theta = init_theta.copy()
    if theta[0] < MIN_MU:
        theta[0] = MIN_MU
        theta_changed = True

    if theta[1] < MIN_SIGMA:
        theta[1] = MIN_SIGMA
        theta_changed = True

    if theta[2] < MIN_RHO:
        theta[2] = MIN_RHO
        theta_changed = True
    elif theta[2] > MAX_RHO:
        theta[2] = MAX_RHO
        theta_changed = True

    if theta[3] < MIN_MIX_PARAM:
        theta[3] = MIN_MIX_PARAM
        theta_changed = True
    elif theta[3] > MAX_MIX_PARAM:
        theta[3] = MAX_MIX_PARAM
        theta_changed = True
        
    return theta, theta_changed


def step_ca(z1, z2, theta, index, min_val, max_val):

    inner_theta = theta.copy()
    def f(alpha):
        inner_theta[index] = theta[index] + alpha
        return -calc_gaussian_mix_log_lhd(inner_theta, z1, z2)

    assert theta[index] >= min_val
    min_step_size = min_val - theta[index]
    assert theta[index] <= max_val
    max_step_size = max_val - theta[index]

    alpha = fminbound(f, min_step_size, max_step_size)
    prev_lhd = -f(0)
    new_lhd = -f(alpha)
    if new_lhd > prev_lhd: theta[index] += alpha
    else: new_lhd = prev_lhd
    return theta, new_lhd


def iter_ca(z1, z2, prev_theta, max_iter,
                 fix_mu=False, fix_sigma=False, eps=1e-12):
    """
    Fit the gaussian model params via coordinate ascent.
    """
    init_lhd = calc_gaussian_mix_log_lhd(prev_theta, z1, z2)
    prev_lhd = init_lhd
    min_vals = [MIN_MU, MIN_SIGMA, MIN_RHO, MIN_MIX_PARAM]
    max_vals = [MAX_MU, MAX_SIGMA, MAX_RHO, MAX_MIX_PARAM]
    theta = numpy.array(prev_theta).copy()
    for i in range(max_iter):
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            if index == 0 and fix_mu: continue
            if index == 1 and fix_sigma: continue
            theta, new_lhd = step_ca(z1, z2, theta, index, min_val, max_val)
        
        theta, changed_params = clip_model_params(theta)
        assert changed_params == False
            
        if not changed_params:
            assert new_lhd + 1e-6 >= prev_lhd
            if new_lhd - prev_lhd < eps:
                return theta, new_lhd
        
        prev_theta = theta
        prev_lhd = new_lhd
    
    return theta, new_lhd


def iter_em(
    z1, z2, prev_theta, max_iter,
    fix_mu = False, fix_sigma = False, eps = 1e-12
):
    """
    Fit the gaussian model params via EM.
    """
    init_lhd = calc_gaussian_mix_log_lhd(prev_theta, z1, z2)
    prev_lhd = init_lhd
    for i in range(max_iter):
        theta = EM_step(z1, z2, prev_theta)
        theta, changed_params = clip_model_params(theta)
        new_lhd = calc_gaussian_mix_log_lhd(theta, z1, z2)
        # if the model is at the boundary, abort
        if changed_params:
            return theta, new_lhd, True

        assert new_lhd + 1e-6 >= prev_lhd
        if new_lhd - prev_lhd < eps:
            return theta, new_lhd, False
        
        prev_theta = theta
        prev_lhd = new_lhd
    
    return theta, new_lhd, False


def emp_with_pseudo_value(
    r1, r2, theta_0, N = 100, EPS = 1e-4, 
    fix_mu = False, fix_sigma = False
):
    theta = theta_0
    z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
    z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])

    max_num_EM_iter = 30
    
    for i in range(N):
        prev_theta = theta
        # EM only works in the unconstrained case
        if not fix_mu and not fix_sigma:
            theta, new_lhd, changed_params = iter_em(
                z1, z2, prev_theta, max_num_EM_iter, 
                fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10)
        
        if fix_mu or fix_sigma or changed_params:
            theta = prev_theta
            theta, new_lhd = iter_ca(
                z1, z2, prev_theta, max_num_EM_iter, 
                fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10)
        
        sum_param_change = numpy.abs(theta - prev_theta).sum()

        prev_z1 = z1
        z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
        prev_z2 = z2
        z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])
        mean_pseudo_val_change = (
            numpy.abs(prev_z1-z1).mean() + numpy.abs(prev_z2-z2).mean())
        
        # info(
        #     ('iter ' + str(i)).ljust(12) + 
        #     ' dsum: ' + ("%.2e" % sum_param_change) + 
        #     ' dpsv: ' + ("%.2e" % mean_pseudo_val_change) + 
        #     ' theta: ' + theta
        # )
        
        if i > 3 and (sum_param_change < EPS and mean_pseudo_val_change < EPS): 
            break
    
    return theta, log_lhd_loss(r1, r2, theta)


def estimate_model_params(
    r1, r2, theta_0, max_iter = 5000, convergence_eps = 1e-10, 
    fix_mu = False, fix_sigma = False
):

    theta, loss = emp_with_pseudo_value(
        r1, r2, theta_0, N=max_iter, EPS=convergence_eps, 
        fix_mu=fix_mu, fix_sigma=fix_sigma)
    
    return theta, loss
