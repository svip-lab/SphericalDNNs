'''
% Produce an approximately uniform sampling of a unit sphere using a spiral
% method [1]. According to this approach particle (i.e., sample) longitudes 
% are proportional to particle rank (1 to N) and latitudes are assigned to
% ensure uniform sampling density.  
%
% INPUT:
%   - N     : desired number of particles. N=200 is the default setting.
%             Note that sampling becomes more uniform with increasing N. 
%   - vis   : optional logical input argument specifying if you the 
%             spiral-based sampling should be visualized. vis=false is the 
%             default setting.
%
% OUTPUT:  
%   - V     : N-by-3 array of vertex (i.e., sample) co-ordinates.
%   - Tri   : M-by-3 list of face-vertex connectivities. 
%
%
% REFERENCES:
% [1] Christopher Carlson, 'How I Made Wine Glasses from Sunflowers', 
%     July 8, 2011. url: http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
%
% AUTHOR: Anton Semechko (a.semechko@gmail.com)
% DATE: March, 2015  
%
'''
import numpy as np
import math

def SpiralSampleSphere(N):
	'''
	[V,Tri]=SpiralSampleSphere(N,vis)
	'''
	gr = (1 + math.sqrt(5))/2	#gr=(1+sqrt(5))/2;       % golden ratio
	ga = (2 * np.pi * (1 - 1 / gr)) 	#ga=2*pi*(1-1/gr);       % golden angle

	i = np.array(list(range(N-1)))	#i=0:(N-1);              % particle (i.e., point sample) index
	lat = np.arccos(1-2*i/(N-1))	#lat=acos(1-2*i/(N-1));  % latitude is defined so that particle index is proportional to surface area between 0 and lat
	lon=i*ga 						#lon=i*ga;               % position particles at even intervals along longitude

	#% Convert from spherical to Cartesian co-ordinates
	x = np.multiply(np.sin(lat), np.cos(lon))		#x=sin(lat).*cos(lon);
	y = np.multiply(np.sin(lat), np.sin(lon))		#y=sin(lat).*sin(lon);
	z = np.cos(lat)									#z=cos(lat);
	x, y, z = x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis]
	V = np.concatenate((x,y,z),axis=1) 				#V=[x(:) y(:) z(:)];

	return V