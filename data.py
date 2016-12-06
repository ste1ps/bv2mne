# -*- coding: utf-8 -*-

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

__author__= "Alexandre Fabre"
__version__ = "1.0.1"
__date__="21/04/2016"

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import mne
from mne.transforms import (write_trans,
                            read_trans)
import numpy as np

from numpy.linalg import inv

from nibabel.affines import apply_affine

try:
   import cPickle as pickle
except ImportError:
   import pickle

class Master(object):

   def __init__(self, surf=None, areas=None, src=None):
      self.surf=surf
      self.areas= areas
      self.src= src

def serialize(obj, filename=None):   

    if not filename:
        filename='no_name.pickle'
     
    with open(filename,'wb') as output:
        pickle.dump(obj, output)

    output.close()
    
def read_serialize(filename):
    
    with open(filename,'rb') as file: 
        _obj= pickle.load(file)
    file.close()
    return _obj

def read_texture_label(filename):

    label = mne.read_label(filename, subject=None, color=None)

    return label

def create_trans(subject, fname, fname_out):
    """
       Get transformations of the surface from a file that containes filename matrix transformations
    """


    trans_list=[]
    
    with open(fname, 'r') as textfile:
        trans_name = textfile.read().strip().split("\n")

        for name in trans_name:
            split= name.split()
            inv_bool=('inv'== split[0])
            if inv_bool:
                name=split[1]
            else:
                name = split[0]

            name= name.format(subject)
                
            with open(name, 'r') as matfile:
                lines = matfile.read().strip().split("\n")
                lines_list= [l.split() for l in lines]
                translation = lines_list.pop(0)
                
                #transpose the rotations 
                transpose= list(zip(*lines_list))
                
                #append translations 
                transpose.append(translation)

                #create the matrix
                mat_str= np.array(list(zip(*transpose)))
                #mat_str= np.array(lines_list)
                mat= mat_str.astype(np.float)
                mat= np.vstack([mat,[0,0,0,1]])

                if inv_bool:
                   mat=inv(mat)

                #add line por computing translation
                trans_list.append(mat)
        res=None
        trans=None
        for trans_cour in trans_list:
            if trans is None:
                trans=trans_cour
            else:
                trans=np.dot(trans, trans_cour)

    if fname_out.endswith('fif'):
        write_trans(fname_out, trans)
        
    else:
        with open(fname_out, 'w') as matfile:
            for i in range(len(trans)):
                for j in range(len(trans[i])):
                    matfile.write(str(trans[i][j])+' ')
                matfile.write('\n')
                
    return trans

def compute_trans(pos, trans):

    pos = pos.copy()
    if isinstance(trans, str):
        if trans.endswith('fif'):
            trans = read_trans(trans)
        with open(trans, 'r') as matfile:
            lines = matfile.read().strip().split("\n")
            trans = [l.split() for l in lines]
            trans = np.array(trans).astype(np.float)

    pos= apply_affine(trans, pos)
        
    return pos

def read_texture_info(filename, hemi):

    """
        Read file with informations for each parcels
    """

    info_dict={}
    if filename is not None:

        list_hemi= ['lh','rh']
        if hemi not in list_hemi :
            raise ValueError('hemi must be lh or rh')
        hemi_exclude= list_hemi[list_hemi.index(hemi)-1][0].upper()
        fileformat=filename.split('.')[-1]

        def search(header):
            try:
                label = header.index('Label')
                hemi= header.index('Hemisphere')
                lobe= header.index('Lobe')
                name= header.index('Name')
            except ValueError:
                raise ('header Label, Hemisphere, Lobe, Name, must be label file')
            return label, hemi, name, lobe
        
        if fileformat=='xls':
            import xlrd
            wb = xlrd.open_workbook(filename)
            sh = wb.sheet_by_index(0)
            header = sh.row_values(0)
            index= search(header)
            info=[sh.col_values(ind)[1:] for ind in index]
            
            #reverse
            info= list(zip(*info))
            info= [line for line in info if line[1] != hemi_exclude]
            try:
                for line in info : info_dict[int(line[0])]=[line[2],line[3]]
            except ValueError:
                raise('every elements in first colonne must be integer')
        else:
            with open(filename,'r') as textfile:
                lines = textfile.read().strip().split("\n")               
                info= [l.split() for l in lines]
                header= info[0]
                label_ind,hemi_ind,name_ind,lobe_ind= search(header)
                info.pop(0)
                
                info= [line for line in info if line[hemi_ind] != hemi_exclude]
                try:
                    for line in info : info_dict[int(line[label_ind])]=[line[name_ind],line[lobe_ind]]
                except ValueError:
                    raise('every elements in first colonne must be integer')

                textfile.close()
        
    return info_dict



def create_param_dict(obj, hemi=None):
    
    param={}
    hemi_keys=['lh','rh']
    if len(obj) > 2:
        raise ValueError('obj size must less than 2')
    if hemi is not None:
       if hemi not in hemi_keys:
          raise ValueError('hemi must be None | lh | rh')
    
    if isinstance(obj, list):
        if hemi is not None:
            if len(obj) == 1:
                param[hemi]= obj[0]
            else:
                param[hemi]= obj[hemi_keys.index(hemi)]
        else:
            if len(obj)==2:
                param= dict(lh= obj[0], rh= obj[1])
            else:
                try:
                    key = obj[0][0].hemi
                except Exception:
                    key='lh'
                param[key]=obj[0]
    elif isinstance(obj, dict):
        for key in obj.keys():
            if key in hemi_keys:
                param[key]=obj[key]
    else:
        raise TypeError('obj must be list or dict')

    return param

if __name__ == '__main__':
    pass



