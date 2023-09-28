import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.expanduser("~")+"/ivpy/src")
from ivpy import *
from ivpy.plot import overlay
from ivpy.glyph import radar
import pandas as pd

HOME = os.path.expanduser("~") + "/"
lmlvalsfile = HOME + "lmlproc/proc/genome/lml.pkl"

colors = {
    "lola_orange": "#c99277",
    "lola_orange_grey": "#403931",
    "manray_cream": "#fef7db",
    "manray_green": "#235e31"
}

def pkl(o,o_path):
    with open(o_path,'wb') as f:
        pickle.dump(o,f)

def unpkl(o_path):
    with open(o_path,'rb') as f:
        o = pickle.load(f)
    return o

def get_lmlvals():
    return unpkl(lmlvalsfile)

def get_collvals(collvalsfile):
    return unpkl(collvalsfile)

def get_min_range(l):
    lmin = min(l)
    lmax = max(l)
    lrange = lmax - lmin
    
    return lmin,lrange

def get_lmlbounds():
    lmlvals = get_lmlvals()
    lmlbounds = {}
    
    for k in lmlvals.keys():
        lmlbounds[k] = get_min_range(lmlvals[k])
    
    return lmlbounds

def get_collbounds(collvalsfile):
    collvals = get_collvals(collvalsfile)
    collbounds = {}
    
    for k in collvals.keys():
        collbounds[k] = get_min_range(collvals[k])
    
    return collbounds

class CollectionItem:
    def __init__(self):
        self.coll = ''
        self.acc = ''
        self.printpath = ''  
        self.artist = ''
        self.nationality = ''
        self.active = ''
        self.title = ''
        self.date = ''
        self.medium = ''
        self.dims = []
        self.dimvis = None
        self.credit = ''
        self.glyph = None
        self.thickness = [] # list of floats
        self.gloss = [] # list of floats
        self.color = [] # list of dicts with color data
        self.texture = [] # list of dicts with texture data
        self.fluorescence = [] # list of floats (AUC)
        self.goose = None

    def draw_glyph(self,
                   overwrite=False,
                   return_glyph=True,
                   universe='lml',
                   collvalsfile=None,
                   colorloc='base',
                   fill='gray',
                   side=1600,
                   outline='black',
                   outlinewidth=8,
                   gridlinefill='lightgrey',
                   gridlinewidth=4,
                   colloutline='dodgerblue'
                   ):
        
        if all([self.glyph is not None,not overwrite]):
            print("Glyph already exists. Set `overwrite=True` to overwrite.")        
        
        if universe=='lml':
            lmlnorms = get_glyph_norms(self,'lml',colorloc)
            if collvalsfile is not None:
                lmlradar = radar(lmlnorms,fill=fill,side=side,outline=outline,outlinewidth=outlinewidth,gridlinefill=gridlinefill,gridlinewidth=gridlinewidth)

                collnorms = get_glyph_norms(self,'coll',colorloc,collvalsfile)
                collradar = radar(collnorms,fill=None,side=side,outline=colloutline,outlinewidth=int(side/50),radii=False,gridlines=False)

                g = overlay(lmlradar,collradar,side=side,bg='transparent') # lml with coll overlay

            elif collvalsfile is None:
                g = radar(lmlnorms,fill=fill,side=side,outline=outline,outlinewidth=outlinewidth,gridlinefill=gridlinefill,gridlinewidth=gridlinewidth) # lml only

        elif universe=='coll':
            collnorms = get_glyph_norms(self,'coll',colorloc,collvalsfile)
            g = radar(collnorms,fill=fill,side=side,outline=outline,outlinewidth=outlinewidth,gridlinefill=gridlinefill,gridlinewidth=gridlinewidth) # coll only

        if overwrite:
            self.glyph = g

        if return_glyph:
            return g

def get_glyph_norm(i,dim,bounds):
    if dim=='roughness':
        val = np.median([item['roughness'] for item in i.texture])
    elif dim=='bstar_base':
        val = np.median([item['LAB_B'] for item in i.color if item['mloc']=='base'])
    elif dim=='bstar_image':
        val = np.median([item['LAB_B'] for item in i.color if item['mloc']=='image'])
    else:
        val = np.median(getattr(i,dim))    
    
    norm = (val - bounds[0]) / bounds[1]
    
    if dim=='gloss':
        norm = 1 - norm
        
    return norm

def get_glyph_norms(i,universe,colorloc,collvalsfile=None):
    
    if colorloc=='base':
        colordim = 'bstar_base'
    elif colorloc=='image':
        colordim = 'bstar_image'
    
    if universe=='lml':
        lmlbounds = get_lmlbounds()
        norms = [ # counterclockwise from top
            get_glyph_norm(i,colordim,lmlbounds[colordim]),
            get_glyph_norm(i,'thickness',lmlbounds['thickness']),
            get_glyph_norm(i,'roughness',lmlbounds['roughness']),
            get_glyph_norm(i,'gloss',lmlbounds['gloss']),
        ]
    elif universe=='coll':
        collbounds = get_collbounds(collvalsfile)
        norms = [
            get_glyph_norm(i,colordim,collbounds[colordim]),
            get_glyph_norm(i,'thickness',collbounds['thickness']),
            get_glyph_norm(i,'roughness',collbounds['roughness']),
            get_glyph_norm(i,'gloss',collbounds['gloss']),
        ]
        
    return norms

def class_list_to_dataframe(class_list):
    """
    Converts a list of class instances to a pandas dataframe.
    """

    data = [i.__dict__ for i in class_list]
    df = pd.DataFrame(data)

    return df