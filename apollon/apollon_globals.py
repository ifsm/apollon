#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Global definitions"""

# possible guessing methods for the inital Poisson means
_lambda_guess_methods = ['quantile', 'linear']

# standard plot styles
class plot_params:
		signal = {'linewidth':1, 
				  'linestyle':'solid', 
				  'color':'b', 
				  'alpha':.7, 
				  'zorder':0}

		sig_ons = {'linewidth':1, 
				  'linestyle':'solid', 
				  'color':'k', 
				  'alpha':.3, 
				  'zorder':0}

		onset = {'linewidth':2,
				 'linestyle':'dashed',
				 'color':'r',
				 'alpha':.9,
				 'zorder':10}