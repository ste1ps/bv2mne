#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import changepath
from surface import (get_surface,
                     get_surface_areas)
from volume import get_volume
from source import (get_brain_sources,
                    show_surface_sources)
from mne.source_space import SourceSpaces
from data import (read_serialize,
                  Master,
                  serialize,
                  create_param_dict)
import numpy as np


class Brain(object):
    def __init__(self, surfaces=None, volumes=None, surface_master=None):

        """The brain anatomy

        Brain may be composed to surfaces and volumes
        
        Parameters
        ----------
        surfaces : list of Surface object
        volumes : list of Volume object
        surface_master : list of Surface object | dict of Surface object
            The surface that include other surface

        Returns
        -------
        None
        -------
        Author : Alexandre Fabre
        """

        self.surfaces = None
        self.volumes = None
        self.surface_master = None
        self.subject = None

        if surfaces is not None:
            
            # create dictionnary where keys are hemispheres
            self.surfaces = create_param_dict(surfaces)
            
            list_hemi = list(self.surfaces.keys())
            
            # get the first surface in the first hemipshere
            self.subject = self.surfaces[list_hemi[0]][0].subject

            if surface_master is not None:
                self.surface_master = create_param_dict(surface_master)
            else:
                for hemi in self.surfaces:
                    get_master = False
                    i = 0
                    while not get_master and i < len(self.surfaces[hemi]):
                        if hasattr(self.surfaces[hemi][i], 'surface_master'):
                            get_master = True
                            if self.surface_master is None:
                                self.surface_master={}
                            self.surface_master[hemi] = self.surfaces[hemi][i].surface_master
                        i += 1

        if volumes is not None:
            self.volumes = create_param_dict(volumes)
            if self.subject is None:
                self.subject = self.volumes[list(self.volumes.keys())[0]][0].subject

        # structure index
        if self.surfaces is not None:
            self.obj = self.surfaces
            self.name_obj = 'surface'
        elif self.volumes is not None:
            self.obj = self.volumes
            self.name_obj = 'volume'
        else:
            raise Exception('The instance of Bran is empty')

    def __getitem__(self, ind):
        """get anatomy elements with indices

        Parameters
        ----------
        ind : ('lh' | 'rh') | ( ('lh' | 'rh'), int )

        Returns
        -------
        element : list of Surface object | Surface object | list of Volume object | Volume object
        -------
        Author : Alexandre Fabre
        """

        if isinstance(ind, tuple):
            if len(ind) != 2:
                raise ValueError('There are too many indices')
            else:
                key = ind[0]
                val = ind[1]
            element = self.obj[key][val]
        else:
            element = self.obj[ind]

        return element

    def set_index(self, index=None):
        """set structures index

        Parameters
        ----------
        index : 'surface' | 'volume' | None
            set structures of interest. If None change selected structures with others
            
        Returns
        -------
        None
        -------
        Author : Alexandre Fabre
        """
        
        struct = np.array(['surface', 'volume'])
        if index is None:
            # to select the other object
            index = struct[struct!=self.name_obj]
        elif not index in values:
            raise ValueError('index must be \'surface\' or \'volume\'')
        
        if index == 'surface' and self.surfaces is not None:
            self.obj = self.surfaces
            self.name_obj = 'surface'
        elif index == 'volume' and self.volumes is not None:
            self.obj = self.volumes
            self.name_obj = 'volume'

    def get_sources(self, space=5, distance='euclidean', remains=None):
        """get sources of structures

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
        src : list of SourceSpaces object | None
            If there are surfaces and volumes in Brain :
                 src : [ surface sources (lh, rh) , volume sources (lh, rh) ]
                 src size = 2 x 2
            If there is no surfaces or volumes : src size = 2
        -------
        Author : Alexandre Fabre
        """

        src = [[], []]

        for hemi in ['lh', 'rh']:

            if self.surfaces is not None:
                if hemi in self.surfaces:
                    master = None
                    if self.surface_master is not None:
                        if hemi in self.surface_master:
                            master = self.surface_master[hemi]
                            
                    # get surface sources on the hemisphere
                    # we use the surface_master to keep a realist repartition of sources
                    # pack = True : sources surfaces will be packed to get sources for the hemisphere
                    # sources will be packed in using the surface_master
                    sources_surface, _ = get_brain_sources(self.surfaces[hemi], space,
                                                           distance, remains, pack=True,
                                                           master=master)
                    src[0].append(sources_surface)

            if self.volumes is not None:
                if hemi in self.volumes:
                    # get volume sources
                    # pack = True, sources will be packed
                    sources_volume, _ = get_brain_sources(self.volumes[hemi], space, remains, pack=True)
                    src[1].append(sources_volume)
        
        # do not keep the empty hemisphere
        src = list(filter(lambda x: x, src))
        length = len(src)
        if length == 1:
            src = src[0]
        elif src == 0:
            src = None

        return src

    def get_index(self, index=None, hemi='all', lobe=None, name=None):
        """get index of interest structures in brain

        Parameters
        ----------
        index : None, list of int
            Positions in structure list
        hemi : 'all' | 'lh' | 'rh'
            Hemisphere of interest
        lobe : None | list of str
            Lobes of interest
        name : None | list of str
            Names of interest

        Returns
        -------
        select : dict of index list
            Indices in each hemisphere
        -------
        Author : Alexandre Fabre
        """

        select = dict(lh=None, rh=None)
        keys = list(self.obj.keys())

        # select hemipheres
        if hemi is None or hemi == 'all':
            obj = self.obj.copy()
        elif hemi in keys:
            obj = {hemi: self.obj[hemi].copy()}
        else:
            raise ValueError('hemi is invalid')

        # get indices to structures in valid hemispheres
        for hemi in obj:
            select[hemi] = np.arange(len(obj[hemi])).tolist()

        if isinstance(name, str):
            name = [name]
        if isinstance(lobe, str):
            name = [lobe]

        if index is not None:
            if isinstance(index, int):
                index = [index]
            for hemi in obj:
                # the intersection of old valid indices and new indices
                select[hemi] = list(set(select[hemi]).intersection(index))

        if name is not None or lobe is not None:
            for k, hemi in enumerate(obj):
                index = []
                for i, cour_obj in enumerate(obj[hemi]):

                    # an index is selected only if it's in the good lobe and have the good name
                    valid_test = 0
                    for attr, test in [[cour_obj.name, name],
                                       [cour_obj.lobe, lobe]]:
                        if test is not None:
                            if attr in test:
                                valid_test += 1
                        else:
                            valid_test += 1
                    if valid_test == 2:
                        index.append(i)

                # the intersection of old valid indices and new indices
                select[hemi] = list(set(select[hemi]).intersection(index))
                if not select[hemi]:
                    select[hemi] = None

        err = True
        for hemi in select:
            if select[hemi] is not None:
                err = False
        if err:
            raise Exception('no parcels have been selected')

        return select

    def get(self, index=None, hemi='all', lobe=None, name=None):
        """get structures of interest

        Parameters
        ----------
        index : None, list of int
            Positions in structure list
        hemi : 'all' | 'lh' | 'rh'
            Interest hemisphere
        lobe : None | list of str
            Interest lobes
        name : None | list of str
            Interest names

        Returns
        -------
        obj : dict of Surface list | dict of Volume list
            Selected structures by indices
        -------
        Author : Alexandre Fabre
        """

        # get indices of each hemisphere
        index = self.get_index(index=index, hemi=hemi,
                               lobe=lobe, name=name)
        obj = self.obj.copy()
        delete = []
        for hemi in obj:
            if index[hemi] is None:
                delete.append(hemi)
            else:
                obj[hemi] = np.take(obj[hemi], index[hemi])

        # delete empty hemisphere
        for hemi in delete:
            del obj[hemi]

        return obj

    def show(self, figure=None, index=None, hemi='all', lobe=None, name=None,
             color=None, opacity=1, bgcolor=(1, 1, 1), size=(600, 600),
             show_brain=False, brain=None, get_brain=False, fileformat='white'):
        """show structures

        Parameters
        ----------
        figure : None | Figure object
        index : None, list of int
            Positions in structure list
        hemi : 'all' | 'lh' | 'rh'
            Hemisphere of interest
        lobe : None | list of str
            Lobes of interest
        name : None | list of str
            Names of interest
        color : list(3) | None
            Color of the surface
        opacity : float | None
            Opacity of the surface. It's must be between 0 and 1
        bgcolor : list(3)
            Color of the background
        size : list(2)
            Size of the window that displays figures
        show_brain : Boolean
            If True, display the regions on brain hemispheres
        brain : Brain surfer instance
            The basic hemisphere
        get_brain : Boolean
            If True, return the last Brain surfer instance
        fileformat : str
            The format of file surface hemisphere. It's used only if show_brain is True

        Returns
        -------
        figure : Figure object
        -------
        Author : Alexandre Fabre
        """

        obj = self.get(index=index, hemi=hemi,
                       lobe=lobe, name=name)
        for hemi in obj:
            brain = None
            for cour_obj in obj[hemi]:
                if self.name_obj == 'volume':
                    cour_obj = cour_obj.get_surface()
                figure, brain = cour_obj.show(figure=figure, color=color, opacity=opacity,
                                              bgcolor=bgcolor, show_brain=show_brain,
                                              brain=brain, fileformat=fileformat,
                                              size=size, get_brain=True)
        return figure

    def show_sources(self, src, index=None, hemi='all', name=None, lobe=None, figure=None,
                     color=None, opacity=1, bgcolor=(1, 1, 1), size=(600, 600), show=True, show_brain=False,
                     fileformat='white', sphere_color=(0.7, 1, 0.7), scale_factor=1, resolution=8):
        """show structures

        Parameters
        ----------
        src : SourceSpaces object
        index : None | list of int
            Positions in structure list
        hemi : 'all' | 'lh' | 'rh'
            Interest hemisphere
        name : None | list of str
            Interest names
        lobe : None | list of str
            Interest lobes
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
        show_brain : bool
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
        -------
        Author : Alexandre Fabre
        """

        # get indices to each hemisphere
        index = self.get_index(index=index, hemi=hemi,
                               name=name, lobe=lobe)
        
        if isinstance(src, SourceSpaces) or\
           (len(src)==1 and isinstance(src[0], SourceSpaces)):
            if hemi not in ['lh', 'rh']:
                raise ValueError('hemi must be lh | rh')
            src = create_param_dict(src, hemi=hemi)
        else:
            src = create_param_dict(src)

        for hemi in src:
            if index[hemi] is not None:
                index_pack_src=[]
                if self.name_obj == 'volume':
                    surfaces=[]
                    for vol in self.obj[hemi]:
                        surfaces.append(vol.get_surface())
                        index_pack_src.append(vol.index_pack_src)
                else:
                    surfaces = self.obj[hemi]
                    for surf in surfaces:
                        index_pack_src.append(surf.index_pack_src)
                # call to the function that show sources
                figure = show_surface_sources(src[hemi], surfaces, index_pack_src, index[hemi], figure=figure, color=color,
                                              opacity=opacity,  bgcolor=bgcolor, show=show, brain=None,
                                              show_brain=show_brain, get_brain=False, fileformat=fileformat,
                                              size=size, sphere_color=sphere_color, scale_factor=scale_factor,
                                              resolution=resolution)
        return figure


def get_brain(subject, fname_surf_L=None, fname_surf_R=None, fname_tex_L=None,
              fname_tex_R=None, bad_areas=None, fname_vol=None, name_lobe_vol='Subcortical',
              trans=False, fname_atlas=None, fname_color=None):
    """show structures

    Parameters
    ----------
    subject : str
        The name of the subject
    fname_surf_L : None | str
        The filename of the surface of the left hemisphere
    fname_surf_R : None | str
        The filename of the surface of the right hemisphere
    fname_tex_L : None | str
        The filename of the texture surface of the right hemisphere
        The texture is used to select areas in the surface
    fname_tex_R : None | str
        The filename of the texture surface of the left hemisphere
    bad_areas : list of int
        Areas that will be excluded from selection
    fname_vol : None | str
        The filename of mri labelized
    name_lobe_vol : None | list of str | str
        Interest lobe names
    trans : str | None
        The filename that contains transformation matrix for surface
    fname_atlas : str | None
        The filename of the area atlas
    fname_color : Brain surfer instance
        The filename of color atlas
    Returns
    -------
    figure : Figure object
    -------
    Author : Alexandre Fabre
    """

    list_hemi = ['lh', 'rh']
    volumes = None
    surfaces = None

    fname_surf = [fname_surf_L, fname_surf_R]
    fname_tex = [fname_tex_L, fname_tex_R]

    print('build surface areas')

    for i, hemi in enumerate(list_hemi):

        if fname_surf[i] is not None and fname_tex[i] is not None:

            if surfaces is None:
                surfaces = []

            surface = get_surface(fname_surf[i], subject=subject, hemi=hemi, trans=trans)

            # save to project areas on the hemisphere
            surface.save_as_ref()

            areas_hemi = get_surface_areas(surface, texture=fname_tex[i], hemi=hemi,
                                           subject=subject, fname_atlas=fname_atlas,
                                           fname_color=fname_color)
            # Delete wrong areas
            if bad_areas is not None:
                areas_hemi = np.delete(areas_hemi, bad_areas, axis=0)

            surfaces.append(areas_hemi)

    print('[done]')

    print('build volume areas')
        
    if fname_vol is not None:
        volumes = []

        for hemi in list_hemi:
            cour_volume = get_volume(fname_vol, fname_atlas, name_lobe_vol, subject,
                                     hemi, True)

            volumes.append(cour_volume)

    print('[done]')

    brain = Brain(surfaces, volumes)

    return brain


def brain_test(subject):

    """ Display sources of the Occipital cortex in the left hemiphere
        and display sources of the Thalamus in the right hemisphere"""

    figure = None
    # Project 's directory
    subjects_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'
    # Surface files
    fname_surf_L = subjects_dir + '{0}/surf/{0}_Lwhite.gii'.format(subject)
    fname_surf_R = subjects_dir + '{0}/surf/{0}_Rwhite.gii'.format(subject)
    # MarsAtlas texture files
    fname_tex_L = subjects_dir + '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
    fname_tex_R = subjects_dir + '{0}/tex/{0}_Rwhite_parcels_marsAtlas.gii'.format(subject)
    # Transformatio file from BV to MNE
    trans = subjects_dir + '{0}/ref/{0}-trans.trm'.format(subject)
    # MarsAtas files
    fname_atlas = subjects_dir + 'label/MarsAtlas_BV_2015.txt'
    fname_color = subjects_dir + 'label/MarsAtlas.ima'
    # MarsAtlas volumetric parcellation file
    fname_vol = subjects_dir + '{0}/vol/{0}_parcellation.nii.gz'.format(subject)
    name_lobe_vol = ['Subcortical']

    # Create brian object
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, trans, fname_atlas, fname_color)

    # To show MarsAtlas parcels
    # brain.show()
    # To show the left frontal areas (problem in insula)
    # brain.show(hemi='lh', lobe=['Frontal'])
    # brain.show(hemi='lh', lobe=['Frontal'], name=['Insula'])
    # Create source space on surface and volume
    src = brain.get_sources(space=5, distance='euclidean')

    # Display sources in frontal lobe
    # brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], sphere_color=(0.7, 0.7, 0.7))
    # Display sources in occipital lobe
    # brain.show_sources(src[0], hemi='lh', lobe=['Occipital'])
    # The show_brain option = True does not work because it calls a FS mesh which is not correctly oriented
    # figure = brain.show_sources(src[0], hemi='lh', lobe=['Occipital'], figure=figure, opacity=1, show_brain=False)
    # Display sources in the motoro cortex
    # brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], name=['Mdl'], opacity=1)

    brain.set_index()

    # Does no work for hemi='all'
    brain.show_sources(src[1], hemi='all', lobe=['Subcortical'], name=['Thal'], opacity=0.1)

if __name__ == '__main__':
    brain_test('subject_04')
