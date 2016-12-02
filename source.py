#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import numpy as np
import mne
import warnings
from mne.source_space import SourceSpaces
from mne.surface import complete_surface_info
from mne import write_source_spaces
from sklearn.cluster import MiniBatchKMeans
from math import pi
import Pycluster
from Pycluster import kmedoids

from data import (read_serialize,
                  Master, serialize)

import surface as surf_m

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics.pairwise import euclidean_distances
from mne.source_space import SourceSpaces
from mne.io.constants import FIFF


def get_number_sources(obj, space=5, surface=True):
    """get number sources in region

    Parameters
    ----------
    obj : Surface object | Volume object
        The region where sources is computing
    space : float
        The distance between sources
    surface : bool
        If True, the number of sources is computing on a surface. Else it's computing in volume

    Returns
    -------
    sol : int
        Number of sources
    diff :
        Number points that are not sources
    -------
    Author : Alexandre Fabre
    """

    if space < 0:
        raise ValueError('the space number must be positive')

    # the number of selected vertices mustn't exceed the number of vertices
    _max = obj.pos_length

    # compute on the surface
    if surface:
        # surface area per source in mm2
        space = float(space ** 2)

        # compute the area of the surface with its triangle areas
        sol = 0
        triangles_pos = obj.pos[obj.triangles]
        for points in triangles_pos:
            a, b, c = np.array(points)
            
            tri_area = 0.5 * np.linalg.norm(np.cross(b-a,c-a))
            
            sol += tri_area

    # compute on the volume
    else:
        # we divide the volume by a sphere volume to know the number of sources
        space = float(4 / 3 * pi * ((space / 2) ** 3))
        sol = _max

    # avoid division by zero
    sol = int(sol / (max(space, 1e-5)))

    sol = int(min(_max, sol))

    # number of points that have not been selected
    diff = _max - sol

    return sol, diff


def get_volume_sources(volume, space=5, remains=None):
    """get sources in volume

    Parameters
    ----------
    volume : Volume object
    space : float
        The distance between sources
    remains : None | int
        The number of sources that we want to keep

    Returns
    -------
    src : SourceSpaces object
    -------
    Author : Alexandre Fabre
    """

    if remains is None:
        remains, removes = get_number_sources(volume, space=space,
                                              surface=False)

    else:
        # avoid to have an incorrect number of sources
        remains = max(0, min(volume.pos_length, remains))
        removes = volume.pos_length - remains

    if remains == 0:
        raise ValueError('Error, 0 source created')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # create clusters
        km = MiniBatchKMeans(n_clusters=remains, n_init=10, compute_labels=True)
        
    # get cluster labels
    cluster_id = km.fit(volume.pos).labels_

    # get centroids of clusters
    centroids, _ = Pycluster.clustercentroids(volume.pos, clusterid=cluster_id)

    dist = euclidean_distances(centroids, volume.pos)

    # get indices of closest points of centroids
    arg_min = np.argmin(dist, axis=1)

    inuse = np.zeros(volume.pos_length)
    inuse[arg_min] = 1

    # must be converted to meters
    rr = volume.pos * 1e-3

    if volume.hemi=='lh':
        Id = 101
    elif volume.hemi=='rh':
        Id = 102
    src = [{'rr': rr, 'coord_frame': np.array((FIFF.FIFFV_COORD_MRI,), np.int32), 'type': 'surf', 'id': Id,
            'np': volume.pos_length, 'nn': volume.normals, 'inuse': inuse, 'nuse': remains, 'dist': None,
            'nearest': None, 'use_tris': None, 'nuse_tris': 0, 'vertno': arg_min, 'patch_inds': None,
            'tris': None, 'dist_limit': None, 'pinfo': None, 'ntri': 0, 'nearest_dist': None, 'removes': removes}]

    src = SourceSpaces(src)

    return src


def get_surface_sources(surface, space=5, distance='euclidean', remains=None):
    """get sources in volume

    Parameters
    ----------
    surface : Surface object
    space : float
        The distance between sources
    distance : 'euclidean' | 'dijkstra' | 'continuous'
        The distance used to compute distance on surface
    remains : None | int
        The number of sources that we want to keep

    Returns
    -------
    src : SourceSpaces  object
    -------
    Author : Alexandre Fabre
    """

    if remains is None:
        remains, removes = get_number_sources(surface, space=space,
                                              surface=True)
    else:
        # avoid to have a incorrect number of sources
        remains = max(0, min(surface.pos_length, remains))
        removes = surface.pos_length - remains

    if remains == 0:
        raise ValueError('Error, 0 source created')

    if removes == 0:
        # all points are sources
        # logger.info('all points are remained')
        centroids_id = np.arange(remains)
        inuse = np.ones(surface.pos_length, dtype=int)

    else:
        # connectivity of neighbors points
        n_neighbors = min(50, surface.pos_length)

        # get the matrix that identify neighbors points
        knn_graph = kneighbors_graph(surface.pos, n_neighbors, include_self=False)

        # ward criterion is adapted for a surface clustering
        model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph,
                                        n_clusters=remains)
        # compute clusters
        model.fit(surface.pos)

        # get cluster labels
        cluster_id = model.labels_

        # get the distance between points on the surface with Dijkstra or continuous
        # if distance is euclidean, it just computes euclidean distances between points
        distance = surf_m.get_surf_distance(surface.pos, surface.triangles,
                                            distance=distance)

        # clusters give by AgglomerativeClustering are initial clusters for k-medoids
        # for k-medoids, the centroid is a point in a cluster
        # k-medoids method return clusters that are identified by the index of their centroid point
        cluster_id, _, _ = kmedoids(distance, nclusters=remains, npass=1,
                                    initialid=cluster_id)

        # get the index of centroids
        centroids_id = np.unique(cluster_id)

        inuse = np.zeros(surface.pos_length)
        inuse[centroids_id] = 1

    # must be converted to meters
    rr = surface.pos * 1e-3

    if surface.hemi=='lh':
        Id = 101
    elif surface.hemi=='rh':
        Id = 102
    src = [{'rr': rr, 'coord_frame': np.array((FIFF.FIFFV_COORD_MRI,), np.int32), 'type': 'surf', 'id': Id,
            'np': surface.pos_length, 'nn': surface.normals, 'inuse': inuse, 'nuse': remains, 'dist': None,
            'ntri': surface.triangles_length, 'nearest': None, 'use_tris': None, 'nuse_tris': 0,
            'vertno': centroids_id, 'patch_inds': None, 'tris': surface.triangles, 'dist_limit': None, 'pinfo': None,
            'nearest_dist': None, 'removes': removes}]

    src = SourceSpaces(src)

    return src


def sources_pack(src, pos=None, normals=None, triangles=None, unit='mm'):
    """pack sources

    Parameters
    ----------
    src : list of SourceSpaces object
    pos : array(2) | None
        positions master
    normals : array(2) | None
        normals master
    triangles : array(2) | int
        triangles master
    unit : 'mm' | 'm'
        The unit of positions
    Returns
    -------
    src : SourceSpaces  object
    -------
    Author : Alexandre Fabre
    """

    ind = []
    ntri = 0
    index_pack_src = []

    try :
        if isinstance(src, SourceSpaces) or isinstance(src[0], dict):
            src = [s for s in src]
        elif isinstance(src[0], SourceSpaces):
            src= [s[0] for s in src]
        else:
            raise TypeError('src is incorrect')
    except Exception as msg_error:
        raise Exception(msg_error)

    if not len(src):
        raise ValueError('src is empty')

    get_normals = True
    get_triangles = True
    if normals is None:
        normals = []
        get_normals = False
    if triangles is None:
        triangles = None if src[0]['tris'] is None else []
        get_triangles = False

    # pack sources
    if pos is None:
        pos_nb = 0
        src_nb = 0
        rr = []
        for s in src:
            p = np.add((s['vertno']), pos_nb)
            ind += list(p)
            rr += list(s['rr'])
            index_pack_src.append(np.arange(src_nb, src_nb + s['nuse']).tolist())
            if not get_normals:
                normals += list(s['nn'])
            if triangles is not None:
                triangles += np.add(s['tris'], pos_nb).tolist()
            pos_nb += s['np']
            src_nb += s['nuse']
            ntri += s['ntri']
        remains = src_nb
        surf_length = pos_nb
        rr = np.array(rr)

    # pack sources with surface master
    else:
        if unit == 'mm':
            pos *= 1e-3
        surf_length = len(pos)
        if not get_normals:
            normals = np.zeros(len(pos))
        rr = pos.tolist()

        for s in src:

            normals_cond = (not get_normals and not get_triangles and triangles is None)
            triangles_cond = (not get_triangles and triangles is not None)

            if triangles_cond or normals_cond:
                pos_cour = s['rr'].tolist()
                pos_ind = list(map(rr.index, pos_cour))
            if normals_cond:
                normals[pos_ind] = s['nn']
            if triangles_cond:
                triangles += np.take(pos_ind, s['tris']).tolist()

            pos_src_cour = s['rr'][s['vertno']].tolist()
            ind += list(map(rr.index, pos_src_cour))

        remains = len(ind)
        index = np.sort(ind, axis=0).tolist()
        ntri = len(triangles)

        if not get_normals and triangles is not None:
            _dict = dict(rr=pos, tris=triangles,
                         ntri=ntri, np=surf_length)
            normals = _complete_surface_info(_dict)['nn']

        cour = 0
        for s in src:
            sup = cour + s['nuse']
            select = ind[cour: sup]
            index_pack_src.append(list(map(index.index, select)))
            cour = sup
        ind = index

    src_dict = src[0].copy()
    inuse = np.zeros(surf_length)
    ind = np.array(ind).tolist()
    inuse[ind] = 1

    src_dict.update(dict(rr=rr, inuse=inuse, np=surf_length,
                         nuse=remains, nn=normals,
                         tris=triangles, ntri=ntri,
                         vertno=ind))

    src = SourceSpaces([src_dict])

    return src, index_pack_src


def get_brain_sources(obj, space=5, distance=None, remains=None, pack=True,
                      master=None):
    """get brain sources for each structures

    Parameters
    ----------
    obj : list of Surface object | list of Volume object
    space : float
        Distance between sources
    distance : 'euclidean' | 'dijkstra' | 'continuous' | None
        The distance used on the surface
        If distance is different to None, obj is treated like a Surface object
    remains : int | None
        The number of sources that we want to keep
    pack : Boolean
        If True pack sources
    master: Surface object | Volume object | array | None
        The reference structure used to pack sources or positions
    Returns
    -------
    src : SourceSpaces  object
    index_pack_src : list(2)
         Source indices in the source list for each parcels.
         If pack is None, it's not returned
    -------
    Author : Alexandre Fabre
    """

    remain_nb = 0
    remove_nb = 0
    src = []
    for i, cour_obj in enumerate(obj):

        if distance is not None:
            cour_src = get_surface_sources(cour_obj, space, distance, remains)
        else:
            cour_src = get_volume_sources(cour_obj, space, remains)

        remove_nb += cour_src[0]['removes']
        remain_nb += cour_src[0]['nuse']
        src.append(cour_src)

    if pack:
        pos = None
        normals = None
        triangles = None
        # get attributes from master
        if master is not None:
            if isinstance(master, list) or isinstance(master, np.ndarray):
                pos = master
            else:
                if hasattr(master, 'pos'):
                    pos = master.pos
                elif hasattr(master, 'rr'):
                    pos = master.rr

                if hasattr(master, 'normals'):
                    normals = master.normals
                elif hasattr(master, 'nn'):
                    normals = master.nn

                if hasattr(master, 'triangles'):
                    triangles = master.triangles
                elif hasattr(master, 'tris'):
                    triangles = master.tris

        # pack sources
        src, index_pack_src = sources_pack(src, pos, normals, triangles)
        for i in range(len(obj)):
            if hasattr(obj[i], 'index_pack_src'):
                obj[i].index_pack_src = index_pack_src[i]
    else:
        src = SourceSpaces(src)

    print("\n%d points have been removed, %d points remained for downsample" % (remove_nb, remain_nb))

    if pack:
        return src, index_pack_src
    else:
        return src


def show_surface_sources(src, obj, index_pack_src=None, index=None, figure=None, color=None, bgcolor=(1, 1, 1),
                         size=(600, 600), opacity=1,  show=True, show_brain=False, brain=None, get_brain=False,
                         fileformat='white', sphere_color=(0.7, 1, 0.7), scale_factor=1, resolution=8):
    """show sources with surfaces

    Parameters
    ----------
    src : SourceSpaces object | array
    obj : list of Surface objects | list of Volume objects
    index : None, list of int
        Positions in structure list
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
        If True, display the regions on brain hemispheres
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
        It returned if show_brain is True
    -------
    Author : Alexandre Fabre
    """

    get_pos = False
    if not isinstance(src, SourceSpaces):
        try:
            src = np.array(src)
            get_pos = True
        except Exception:
            raise TypeError('src must be SourceSpaces object | array')
    else:
        if len(src)>1:
            raise ValueError('the src size must be equal to 1')
        src = src[0]

    if index is None:
        index = np.arange(len(obj))
    elif isinstance(index, int):
        index = [index]
    elif not isinstance(index, list):
        raise TypeError('index must be an Integer | List | None')

    # if index is empty
    if not len(index):
        raise ValueError('index is empty')

    brain = None
    for ind in index:
        surface = obj[ind]
        # get directly sources positions
        if not get_pos:
            index_src = None
            if index_pack_src is None : 
                if hasattr(surface, 'index_pack_src'):
                    index_src = surface.index_pack_src
            else:
                index_src = index_pack_src[ind]
                
            # get point positions
            p = np.take(src['vertno'], index_src, axis=0).tolist()

            # get source positions
            pos_cour = np.take(src['rr'], p, axis=0) * 1000
            
        else:
            index_src = False
            pos_cour = src[ind]
            
        if index_src is not None:
            figure, brain = obj[ind].show_sources(pos_cour, figure=figure, color=color, opacity=opacity,
                                                  bgcolor=bgcolor, show=show, sphere_color=sphere_color,
                                                  scale_factor=scale_factor, resolution=resolution, brain=brain,
                                                  show_brain=show_brain, get_brain=True, fileformat=fileformat)

    if get_brain:
        return figure, brain
    else:
        return figure

if __name__ == '__main__':
    pass
