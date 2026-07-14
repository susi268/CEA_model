import numpy as np
import matplotlib.pyplot as plt

def visu_map(radius, theta, phi, qty, rad=1.5, cmap='seismic'):
  # Function to see any output quantity from PLUTO as a map
  # at any chosen radius
  # Input : radius is the list of radii (Rs)
  #         theta is the list of colatitudes (rad)
  #         phi is the list of longitudes (rad)
  #         qty is the physical quantity to plot (PLUTO)
  #         rad is the radius at which to plot (Rs)
  #         cmap is the colormap chosen for the figure
  # Output : the function returns a plot
  # ex : from visu_map import visu_map
  #      visu_map(d.x1, d.x2, d.x3, d.bx1)

  # Convert to latitude and longitude  
  lat = 90. - 180.*theta/np.pi
  longi = 180.*phi/np.pi
  Lat, Long = np.meshgrid(lat, longi, indexing='ij')

  # Find radius
  n1 = len(radius)
  idx_rad = 0
  while ((radius[idx_rad] < rad) & (idx_rad < n1-1)):  
    idx_rad = idx_rad + 1

  # Plot
  fig = plt.figure()
  plt.pcolor(Long, Lat, qty[idx_rad,:,:], cmap=cmap)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title(r'R = {} $R_s$'.format(radius[idx_rad]))
  fig.show()
