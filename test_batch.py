import sys
# Remove competing MNE versions
sys.path.remove('/home/brovelli.a/.local/lib/python2.7/site-packages/mne-0.11.0-py2.7.egg')
sys.path.remove('/hpc/soft/lib/python2.7/site-packages/mne-0.12.0-py2.7.egg')
sys.path.remove('/hpc/comco/anaconda2/lib/python2.7/site-packages/mne-0.14.dev0-py2.7.egg')
sys.path.append('/home/brovelli.a/PycharmProjects/mne-python')
# Change paths
sys.path.remove('/home/brovelli.a/.local/lib/python2.7/site-packages')
sys.path.remove('/hpc/soft/lib/python2.7/dist-packages')
sys.path.remove('/hpc/soft/lib/python2.7/site-packages')

print(sys.path)

Subject = sys.argv[1]
Session = sys.argv[2]


import mne

import numpy as np

from mne import Label
from mne.surface import complete_surface_info
from nibabel import gifti
from nibabel.gifti.gifti import GiftiImage, GiftiDataArray
from mne.source_space import SourceSpaces
#from surfer import Brain
from scipy.stats import rankdata
import source
from data import (read_texture_info,
                  read_serialize,
                  compute_trans)
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
import networkx as nx
import gdist

print(Subject)
print(Session)