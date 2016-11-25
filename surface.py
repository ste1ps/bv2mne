#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import warnings

from mayavi import mlab

import numpy as np

import mne
from mne import Label
from mne.surface import _complete_surface_info
from mne.io.constants import FIFF
from nibabel import gifti
from nibabel.freesurfer.io import write_geometry
from nibabel.gifti.gifti import GiftiImage, GiftiDataArray
from mne.source_space import SourceSpaces
from surfer import Brain

from scipy.stats import rankdata

import source

from data import (read_texture_info,
                  read_serialize,
                  compute_trans)

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
import networkx as nx
import gdist


class Surface(object):

    def __init__(self, pos, triangles, subject, label=None, hemi='lh', color=(0.8, 0.2, 0.1),
                 name='no_name', lobe='no_name', normals=None, surface_master=None,
                 trans=None):
        """constructor of Surface class

        Surface contains informations about a surface region

        Parameters
        ----------
        pos : array
            Positions of each points of mesh
        triangles : array
            Mesh triangles
        subject : str
            The name of the subject
        label : mne.Label object
            The Label that corresponds to this surface
        hemi : 'lh' | 'rh'
            The name of the hemisphere
        color : array(3)
            The color of the surface
        name: str
            The name of the surface
        lobe : str
            The lobe of the surface
        normals : array(2)
            Normals on the surface
        surface_master : Surface object
            The reference surface
        trans : str | array | None
            The matrix transformation or the filename to get this

        Returns
        -------
        None
        -------
        Author : Alexandre Fabre
        """
        
        self.pos = np.array(pos)
        self.pos_length = len(pos)
        self.triangles = np.array(triangles)
        self.triangles_length = len(triangles)
        self.label = label
        self.subject = subject
        self.hemi = hemi
        self.color = color
        self.name = name
        self.lobe = lobe
        self.surface_master = surface_master

        # positions of its sources in sources list of hemisphere after they are packed
        self.index_pack_src = None

        if trans is not None:
            self.pos = compute_trans(self.pos, trans)


        # compute normals on the surface
        if normals is None:
            _dict = dict(rr=self.pos, tris=self.triangles,
                         ntri=self.triangles_length, np=self.pos_length)
            self.normals = _complete_surface_info(_dict)['nn']
        else:
            self.normals = normals

    @property
    def get_values(self):
        """return interesting values """
        _dict = {'rr': self.pos, 'tris': self.triangles, 'nn': self.normals}
        return _dict

    def __repr__(self):
        """ Overload the method that return a printable representation of the object  """

        return self.to_string

    @property
    def to_string(self):
        """ Return the information about instance of this class """

        return ("\nvertices : {}\ntriangles : {}\nnumber of vertices : {}\nnumber of triangles : {}".\
                format(self.pos, self.triangles, self.pos_length, self.triangles_length))

    def save_as_ref(self, fileformat='white'):
        self.save_surface(filename='{0}\surf\{1}.{2}'.format(self.subject,
                                                             self.hemi,
                                                             fileformat))

    def save_surface(self, filename):
        """ save the surface """

        mne.write_surface(filename, self.pos, self.triangles, create_stamp=self.subject)

    def save_label(self, filename):
        """ save the label of the surface """

        mne.write_label(filename, self.label)

    def save_gifti_texture(self, filename):
        """ save the texture of the surface in a gifti (BrainVisa) format"""

        gifti_image = GiftiImage()

        if surface_master is None:
            vertices = np.zeros(np.max(self.label.vertices))
        else:
            vertices = np.zeros(self.pos_length)

        vertices[list(self.label.vertices)] = 1

        # create gifti information
        darray = GiftiDataArray.from_array(vertices, intent='NIFTI_INTENT_LABEL',
                                           endian='LittleEndian')
        # location
        darray.intent = 0

        giftiImage.add_gifti_data_array(darray)

        gifti.giftiio.write(gifti_image, filename)

    def get_sources(self, space=5, distance='euclidean', remains=None):
        """get sources of the surface

        Parameters
        ----------
        space : float
            The distance between sources
        distance : 'euclidean' | 'dijkstra' | 'continuous' | None
            The distance used to compute distance on surface
        remains : None | int
            The number of sources that we want to keep

        Returns
        -------
        src : SourceSpaces
        -------
        Author : Alexandre Fabre
        """

        src = source.get_surface_sources(self, space, distance, remains)
        return src
    
    def show(self, figure=None, color=None, opacity=1, bgcolor=(1, 1, 1), size=(600, 600),
             show_brain=False, brain=None, get_brain=False, fileformat='white'):
        """show surface

        Parameters
        ----------
        figure : None | Figure object
        color : list(3) | None
            Color of the surface
        opacity : float | None
            Opacity of the surface. It's must be between 0 and 1
        bgcolor : list(3)
            Color of the background color
        size : list(2)
            Size of the window that displays figures
        show_brain : Boolean
            If True, display regions on brain hemispheres
        brain : Brain surfer instance
            The basic hemisphere
        get_brain : Boolean
            If True, return the last Brain surfer instance
        fileformat : str
            The format of file surface hemisphere. It's used only if show_brain is True

        Returns
        -------
        figure : Figure object
        brain : Brain surfer object
            It returned if show_brain is True
        -------
        Author : Alexandre Fabre
        """
        

        if figure is None:
            figure = mlab.figure(size=size, bgcolor=bgcolor)

        if show_brain:
            if brain is None:
                brain = Brain(self.label.subject, self.label.hemi, fileformat, subjects_dir='.',
                              figure=figure, background='white', curv=False, show_toolbar=True)
            if color is None:
                color = self.label.color
            if self.label is not None:
                brain.add_label(self.label, color=color)
            else:
                raise Exception('instance of Surface has not associate a label')

        else:
            if color is None:
                color = self.color
            x, y, z = np.transpose(self.pos)

            mesh = mlab.pipeline.triangular_mesh_source(x, y, z, self.triangles, opacity=opacity)
            mlab.pipeline.surface(mesh, color=color, opacity=opacity)

        if get_brain:
            return figure, brain
        else:
            return figure

    def show_sources(self, src, figure=None, color=None, opacity=1, bgcolor=(1, 1, 1), size=(600, 600),
                     show=True, show_brain=False, brain=None, get_brain=False, fileformat='white',
                     sphere_color=(0.7, 1, 0.7), scale_factor=1, resolution=8):
        """show surface sources with or without surface

        Parameters
        ----------
        src : SourceSpaces object
        figure : None | Figure object
        color : list(3) | None
            Color of the surface
        opacity : float | None
            Opacity of the surface. It's must be between 0 and 1
        bgcolor : list(3)
            Color of the background color
        size : list(2)
            Size of the window that displays figures
        show : Boolean
            If True show the surface
        show_brain : Boolean
            If True, display regions on brain hemispheres
        brain : Brain surfer instance
            The basic hemisphere
        fileformat : str
            The format of file surface hemisphere. It's used only if show_brain is True
        sphere_color : list(3)
            Color of the sphere that represents the source
        scale_factor : int
            Dimension of the sphere
        resolution : int
            Resolution of the sphere

        Returns
        -------
        figure : Figure object
        brain : Brain surfer object
            If get_brain is True, brain is returned
        -------
        Author : Alexandre Fabre
        """

        if (isinstance(src, SourceSpaces)):
            src = src[0]
        else:
            try:
                src=np.array(src)
            except TypeError:
                raise TypeError('src must be SourceSpaces or array')

        if show:
            figure, brain = self.show(figure=figure, color=color, opacity=opacity, size=size,
                                      bgcolor=bgcolor, brain=brain, show_brain=show_brain,
                                      get_brain=True, fileformat='white')

        for s in src:

            if not show and figure is None:
                figure= mlab.figure(size=size, bgcolor=bgcolor)
            x, y, z = s
            mlab.points3d(x, y, z, color=sphere_color, figure=figure, resolution=resolution,
                          line_width=2, scale_factor=scale_factor)

        if get_brain:
            return figure, brain
        else:
            return figure


def get_surface(fname, subject='S4', hemi='lh', trans=None):
    """get surface whith a file

    Parameters
    ----------
    fname : float
        Filename of the surface
    subject : float
        Name of the subject
    hemi : 'lh' | 'rh'
        Hemisphere of interest
    trans : str | array | None
        The matrix transformation or the filename to get this

    Returns
    -------
    surface : instance of Surface
    -------
    Author : Alexandre Fabre
    """

    try:
        coords, triangles = mne.read_surface(fname)
    except Exception:
        try:
            giftiImage = gifti.giftiio.read(fname)

            coords = giftiImage.darrays[0].data
            triangles = giftiImage.darrays[1].data
        except Exception:
            raise Exception('surface file must be in FreeSurfer or BrainVisa format')

    surface = Surface(coords, triangles, subject=subject, hemi=hemi, trans=trans)

    return surface


def read_color_area(filename, index):
    """get color for points

    Parameters
    ----------
    fname : float
        filename of the color atlas
    index : array
        Lines to extract

    Returns
    -------
    color_list : list
        Selected colors
    -------
    Author : Alexandre Fabre
    """

    color_list = None
    if filename is not None:
        with open(filename, 'r') as textfile:
            # get color in list of string
            color_lines = textfile.read().strip().split("\n")
            # transform string in list : '(100,38,53)' --> [100,38,53]
            color = list(map(eval, color_lines))
            try:
                # set color value between 0 and 1
                color_list = list(map(tuple, (np.dot(np.take(color, index, axis=0), 1 / 255))))
            except ValueError:
                raise ('values don\'t match with lines')

    return color_list


def get_surface_areas(surface, texture, subject='S4', hemi='lh',
                      fname_atlas=None, fname_color=None):
    """get areas on the surface

    Parameters
    ----------
    surface : instance of Surface
    texture : str | array
        Array to get areas or the filename to get this
    subject : str
        Name of the subject
    hemi : 'lh' | 'rh'
        Name of the hemisphere
    fname_atlas : str | None
        Filename for area atlas
    fname_color : str | None
        Filename for area color

    Returns
    -------
    areas : list of Surface object
    -------
    Author : Alexandre Fabre
    """

    areas = []

    rr = surface.pos
    normals = surface.normals

    # get texture with gifti format (BainVisa)
    if isinstance(texture, str):
        giftiImage = gifti.giftiio.read(texture)
        base_values = giftiImage.darrays[0].data

    else:
        base_values = texture

    # sort indices thanks to texture
    argsort = np.argsort(base_values, kind='mergesort', axis=0)

    values = base_values[argsort]

    # get parcels and count their number
    parcels, counts = np.unique(values, return_counts=True)

    vertices = argsort

    rr = rr[argsort, :]
    normals = normals[argsort, :]

    # get parcels information
    info = read_texture_info(fname_atlas, hemi)
    color = read_color_area(fname_color, index=np.unique(values))

    parcels_length = len(parcels)
    cour = 0

    # number of points in a mesh face
    nb_iter = 3

    triangles = surface.triangles

    for pos, val in enumerate(counts):

        name_process = info.get(parcels[pos], False)
        if not name_process:
            name = 'no_name'
            lobe = 'no_name'
        else:
            name = name_process[0]
            lobe = name_process[1]

        end = cour + val

        vertices_cour = vertices[cour: end]

        # each point of parcel must be associated with a face (triangle) in its parcel

        # get triangles where points of the parcel are
        triangles_bool = []
        for i in range(nb_iter):
            triangles_bool += [np.logical_or.reduce([triangles[:, i] == val_vertices
                                                     for val_vertices in vertices_cour])]
        triangles_bool = np.transpose(triangles_bool)

        # counting the number of True per lines --> True : 1 , False : 0
        # to know how many points of the parcel are in each face
        counts = triangles_bool.sum(1)

        # indices of each triangles that contains 3 points of the parcel
        ind_all = np.where(counts == 3)
        tris_cour = triangles[ind_all]

        # indices of each faces that contains 2 points of the parcel
        ind = np.where(counts == 2)

        tris_modif = triangles[ind]
        triangles_bool = triangles_bool[ind]

        delete = np.append(ind_all, ind)
        triangles = np.delete(triangles, delete, axis=0)

        lines, cols = tris_modif.shape

        # for the case of a triangle that contains 2 points of the parcel
        # replace the bad point (not in the parcel) with one of the two good points
        for i in range(lines):
            ind = np.where(triangles_bool[i, :] == False)[0]
            tris_modif[i, ind] = tris_modif[i, (ind + 1) % cols]

        triangles_cour = np.vstack((tris_cour, tris_modif))

        # to associate triangles to the new index of point positions in the area
        triangles_cour = rankdata(triangles_cour, method='dense').reshape(-1, 3) - 1

        if color is None:
            color_cour = (0.8, 0.2, 0.1)
        else:
            color_cour = color[pos]

        rr_cour = rr[cour:end]

        # locations in meters
        rr_label = rr_cour * 1e-3

        label = Label(vertices_cour, rr_label, values=values[cour:end], hemi=hemi,
                      comment=name, subject=subject, name=name,
                      color=color_cour, verbose=None)

        cour_surface = Surface(rr_cour, triangles_cour, subject=subject, label=label, hemi=hemi,
                               name=name, lobe=lobe, color=color_cour,
                               normals=normals[cour:end], surface_master=surface)

        areas.append(cour_surface)
        cour += val

    return areas

def get_surf_distance(pos, triangles=None, distance='euclidean'):
    """get distance on the surface

    Parameters
    ----------
    pos : array
        Point positions
    triangles : array | None
        Index of points that form triangles
    distance : 'euclidean' | 'dijkstra' | 'continuous'
        Method to compute distances. dijkstra use the shortest path on the surface.
        continuous use dijkstra continuous algorithm.

    Returns
    -------
    dist : array
        distances between points
    -------
    Ref
    -------
        Mitchell, J.S., Mount, D.M., and Papadimitriou, C.H. (1987). The discrete geodesic problem
    -------
    Author : Alexandre Fabre
    """

    if distance == 'euclidean':
        # this distance don't use mesh 
        dist = euclidean_distances(pos, pos)
    elif distance == 'dijkstra':
        dist = dijkstra_distances(pos, triangles)
    elif distance == 'continuous':
        dist= geodesic_distances(pos, triangles)    

    return dist

def get_graph(pos, triangles):
    """get mesh graph

    Parameters
    ----------
    pos : array
        Point positions
    triangles : array | None
        Index of points that form triangles

    Returns
    -------
    dist : array
        distances between points
    -------
    Author : Alexandre Fabre
    """

    G = nx.Graph()
    length = len(pos)

    # create nodes
    nodes = np.arange(length)

    # set nodes
    G.add_nodes_from(nodes)
    col = len(triangles[0])

    # for each triangles, create 3 edges
    for triangle in triangles:
        for k in range(col):
            i = triangle[k]
            j = triangle[(k+1)%col]
            if not G.has_edge(i,j):
                x = pos[i]
                y = pos[j]

                # use euclidean distances to know the size of the edge
                dist = euclidean(x,y)
                G.add_edge(i,j, distance=dist)
    return G
        

def dijkstra_distances(pos, triangles):
    """use shortest path to compute distances between points in surface

    Parameters
    ----------
    pos : array
        Point positions
    triangles : array | None
        Index of points that form triangles

    Returns
    -------
    dist : array
        distances between points
    -------
    Author : Alexandre Fabre
    """
    
    pos = np.array(pos)

    # creat mesh graph
    G = get_graph(pos, triangles)
    
    points = []
    distance = []

    # compute shortest path on all pairs
    # it's the fastest method
    path = nx.all_pairs_dijkstra_path_length(G, weight='distance')
    length = len(pos)

    for i in range(length):
        
        # get distances in correct format
        distance.append([path[i][dist] for dist in path[i]])
        
    distance = np.array(distance)

    return distance


def geodesic_distances(pos, triangles):
    """geodesic distances using continuous dijkstra method 

    Parameters
    ----------
    pos : array
        Point positions
    triangles : array | None
        Index of points that form triangles

    Returns
    -------
    dist : array
        distances between points
    -------
    Author : Alexandre Fabre
    """

    pos_length = len(pos)

    # convert data in appropriate format
    pos = np.array(pos, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int32)  

    # index for each points
    index = np.arange(pos_length, dtype=np.int32)

    distance = []

    for i in range(pos_length):
        start = np.array([index[i]], dtype=np.int32)
        targets = index[:i]
        distance.append(gdist.compute_gdist(pos, triangles, start, targets))

    distance = get_full_matrix(distance)
    
    return distance

def get_full_matrix(mat):

    """
    Parameters
    ----------
    mat : array
        Matrix must be in form :
        X = [[], [(x2,x1)], [(x3,x1),(x3,x3)], ...]
        
    Returns
    -------
    full_matrix : array
        Rectangular matrix
    -------
    """
    
    length = len(mat)
    mat = np.array(mat)
    full_matrix = np.zeros((length, length))
    full_matrix[0,0] = 0
    for i in range(1,length):
        for j in range(len(mat[i])):
            full_matrix[i,j] = mat[i][j]
            
    full_matrix = full_matrix + np.transpose(full_matrix)

    return full_matrix

if __name__ == '__main__':
    pass
