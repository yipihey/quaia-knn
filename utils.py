import fitsio
import numpy as np
import k3d
import pandas as pd
import SEdist
import pygwalker as pyg
import matplotlib.pyplot as plt
import scienceplots
import scipy.spatial
import itertools
from scipy.interpolate import interp1d
import healpy
plt.style.use(['science','ieee'])
color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as default_cosmo
import astropy.units as u
import astropy.cosmology.units as cu


def add_cartesian_coordinates(df, cosmo=default_cosmo, redshifts=None):
    # Extract RA, Dec, and redshift from the DataFrame
    ra = df['ra'].values
    dec = df['dec'].values
    if redshifts is None:
        redshifts = df['redshift_quaia'].values 
    else:
        df['redshifts'] = redshifts

    # Calculate comoving distance using the cosmology model
    comoving_distance = cosmo.comoving_distance(redshifts)

    # Create SkyCoord object
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=comoving_distance)

    # Convert to Cartesian coordinates
    x = coords.cartesian.x.value
    y = coords.cartesian.y.value
    z = coords.cartesian.z.value

    # Add Cartesian coordinates to the DataFrame
    df['x'] = np.float32(x)
    df['y'] = np.float32(y)
    df['z'] = np.float32(z)

    return df


# Helper function to generate a mesh for a sphere geometry to display with k3d
def generate_sphere(radius=1.0, center=(0, 0, 0), resolution=30):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    # Flatten arrays
    vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Create faces (triangular mesh)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = p1 + 1
            p3 = (i + 1) * resolution + j
            p4 = p3 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    faces = np.array(faces, dtype=np.uint32)

    return vertices, faces
