#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

from nibabel.affines import apply_affine
import nibabel as nib
from scipy.ndimage.filters import generic_filter

from data import (read_serialize,
                  read_texture_info)

import numpy as np
from source import get_volume_sources
from sklearn.neighbors import kneighbors_graph
from surface import Surface
import mne


class Volume(object):
    def __init__(self, pos, voxel, label, subject, pix_dim=(1,1,1), hemi='lh', name='no_name', lobe='no_name'):
        """The  volume structure

        Parameters
        ----------
        pos : array(2)
            Positions in volume
        hemi : array(3)
            voxel matrix
        label : list of str
            Filename of area atlas
        subject : str
            The name of the subject
        pix_dim : tuple(3)
            voxel dimension
        hemi : 'all' | 'lh' | 'rh'
            Interest hemisphere
        lobe : None | list of str
            Interest lobes
        name : None | list of str
            Interest names

        Returns
        -------
        None
        -------
        Author : Alexandre Fabre
        """

        self.pos = np.array(pos)
        self.pos_length = len(pos)

        # in volume, there is not directions for normals
        self.normals = np.zeros((self.pos_length, 3))

        self.label = label
        self.voxel = np.array(voxel)
        self.pix_dim = pix_dim
        
        self.subject = subject
        self.hemi = hemi
        self.name = name
        self.lobe = lobe

        # positions of its sources in sources list of hemisphere after they are packed
        self.index_pack_src = None

        self.__surface = None

    def get_sources(self, space=5, remains=None):
        """get sources in volume

        Parameters
        ----------
        space : float
            The distance between sources
        remains : None | int
            The number of sources that we want to keep

        Returns
        -------
        src : SourceSpaces  object
        -------
        Author : Alexandre Fabre
        """

        src = get_volume_sources(self, space, remains)

        return src

    
    def get_surface(self):
        if self.__surface is None:
            self.__surface = self.create_surface()
        return self.__surface
        

    def create_surface(self):
        """ create 3D surface from volume"""
        
        contour = self.select_contour()
        
        triangles = self.triangulate_mesh(contour)
        
        surf = Surface(contour, triangles, subject=self.subject, normals= np.zeros((len(contour), 3)))

        self.__surface = surf

        return surf

    def select_contour(self):

        # set point position in the volume matrix 
        self.count = 0
        
        voxel = self.voxel.copy()
        
        # set a filter
        contour = generic_filter(voxel, self.__filter_fct, footprint=np.ones((3, 3, 3)),
                                 mode='constant', cval=0.0, origin=0.0)
        
        del self.count

        # get indices contour
        contour = contour[contour != -1]

        # get positions
        contour = np.take(self.pos, contour.tolist(), axis=0)

        return contour

    def __filter_fct(self, x):
        """ apply filter """
        res = -1
        
        # if the point of interest is a point of the structure
        if x[len(x) // 2] == self.label:

            if np.any(x != self.label):
                # set index of the contour
                res = self.count
            self.count += 1
        return res

    def triangulate_mesh(self, contour):
        """ create a tringular mesh from point positions """

        # get the max number of neighbors for a point on the contour  --> 26 : 9 * 3 - 1 (include self)
        knn_graph = kneighbors_graph(contour, 26, include_self=True).toarray()

        pos=[]
        # get positons of points that may be construct mesh triangular
        for knn in knn_graph:
            pos.append(np.take(contour, np.where(knn==1)[0], axis=0))

        #get index of each point that forms the triangulation
        triangles=[]
        for i in range(len(pos)):
            pos_length = len(pos)

            if pos_length == 2:
                #flat triangle
                tris=[0, 1, 0]
                
            elif pos_length > 2:
                # create triangles
                neighbors = kneighbors_graph(pos[i], 3, include_self=True).toarray()
                tris = np.where(neighbors==1)[1].reshape(-1,3)

                # sort indices to delete double faces
                tris.sort(axis=1)
            else:
                raise ValueError('point is alone')

            # get indices from all points
            triangles += np.where(knn_graph[i]==1)[0][tris].tolist()

        triangles = np.array(triangles)

        # view turns each row into a single element of np.void dtype
        cour = triangles.ravel().view(np.dtype((np.void, triangles.dtype.itemsize * triangles.shape[1])))

        # full rows can be compared for equality by np.unique
        _, unique_index = np.unique(cour, return_index = True)

        # delete double faces
        triangles= triangles[np.sort(unique_index)]
        
        return triangles

def get_volume(mri, fname_atlas, lobe_name, subject , hemi='lh', reduce_volume=True):
    """get volumes

        Parameters
        ----------
        mri : None | str
            The filename of mri labelized
        fname_atlas : str | None
            The filename of the area atlas
        lobe_name : float | None
            Interest lobe names
        subject : str
            The name of the subject
        hemi : str
            The name of hemi  of interest
        reduce_volume : bool
            If True, get just the volume including
            the strucure of interest

        Returns
        -------
        vol : instance of Volume
        -------
        Author : Alexandre Fabre
        """

    # load the volume
    img = nib.load(mri)

    # get 3D matrix
    voxels = img.get_data()

    affine = img.get_affine()

    header = img.get_header()

    # get voxel dimension
    pix_dim= header['pixdim'][2:5]   

    n_sag, n_axi, n_cor = voxels.shape

    # hack to get a correct translation
    affine[:3, -1] = [n_sag // 2, -n_axi // 2, n_cor // 2]

    volumes = []
    
    if isinstance(lobe_name, str):
        lobe_name = [lobe_name]
        
    info = read_texture_info(fname_atlas, hemi)

    label = []

    for name in lobe_name:

        # get dictionary
        lab = np.take(list(info.keys()),
                      np.where(np.array(list(info.values()))[:, -1] == name))[0].tolist()
        label += lab


    for lab in label:

        # reverse arrays
        point_pos = list(zip(*np.where(voxels == lab)))

        if point_pos:
            # get structure positions
            pos = nib.affines.apply_affine(affine, point_pos)

            vox = np.where(voxels == lab, lab, 0)
            
            if reduce_volume:
                x, y, z = np.nonzero(vox)
                vox = vox[x.min():x.max() + 1,
                          y.min():y.max() + 1,
                          z.min():z.max() + 1]
                
            name, lobe = info[lab]
                
            vol = Volume(pos, voxels, lab, name=name, lobe=lobe,
                         subject=subject, pix_dim=pix_dim, hemi=hemi)
            
            volumes.append(vol)

    else:
        return volumes

def test(subject):
    """show the surface of the thalamus in the left hemisphere"""

    figure = None

    fname_atlas = 'label/MarsAtlas_BV_2015.xls'

    fname_vol = '{0}/vol/{0}_gyriVolume_deepStruct.nii.gz'.format(subject)

    fname_surf = '{0}/vol/{0}_gyriVolume_deepStruct_210_0.gii'.format(subject)

    lobe = ['Subcortical']

    vol = get_volume(fname_vol, fname_atlas, lobe, subject)

    # get surface of first volume structure
    surf = vol[0].get_surface()

    # display the surface
    surf.show(figure = figure)
    

if __name__ == '__main__':
    test('S4')
