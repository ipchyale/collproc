import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.expanduser("~")+"/ivpy/src")
from ivpy import *
from ivpy.plot import overlay
from ivpy.glyph import _radar,_mat

HOME = os.path.expanduser("~") + "/"
lmlvalsfile = HOME + "genome_2021/processing/genome/lml.pkl"

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
        self.acc = ''
        self.print = None  
        self.artist = ''
        self.nationality = ''
        self.active = ''
        self.title = ''
        self.date = ''
        self.medium = ''
        self.dims = ''
        self.credit = ''
        self.glyph = None
        self.thickness = [] # list of floats
        self.gloss = [] # list of floats
        self.color = [] # list of dicts with color data
        self.texture = [] # list of dicts with texture data
        self.fluorescence = [] # list of floats (AUC)
        self.goose = None
        self.imagelight = None
        self.basesat = None
        self.kmap = None

    def draw_glyph(self,overwrite=False,return_glyph=True,universe='lml',collvalsfile=None,side=1600,c='#c99277'):
        if all([self.glyph is not None,not overwrite]):
            raise Exception("Glyph exists; if you wish to overwrite, pass `overwrite=True` to `draw_glyph()`")
        
        if universe=='lml':
            lmlnorms = get_glyph_norms(self,'lml')
            if collvalsfile is not None:
                lmlradar = _radar(lmlnorms,radii=True,gridlines=True,radarfill='dimgrey',outline='#403931',side=side)

                collnorms = get_glyph_norms(self,'coll',collvalsfile)
                collradar = _radar(collnorms,radii=False,gridlines=False,radarfill=None,outline=c,side=side,coll=True)

                radar = overlay(lmlradar,collradar,side=side,bg='transparent')

            elif collvalsfile is None:
                radar = _radar(lmlnorms,radii=True,gridlines=True,radarfill=c,outline='#403931',side=side)

        elif universe=='coll':
            collnorms = get_glyph_norms(self,'coll',collvalsfile)
            radar = _radar(collnorms,radii=True,gridlines=True,radarfill=c,outline='#403931',side=side)

        self.glyph = radar

        if return_glyph:
            return radar


def get_glyph_norm(i,dim,bounds):
    if dim=='roughness':
        val = np.median([item['roughness'] for item in i.texture])
    elif dim=='bstar_base':
        val = np.median([item['LAB_B'] for item in i.color if item['mloc']=='base'])
    else:
        val = np.median(getattr(i,dim))    
    
    norm = (val - bounds[0]) / bounds[1]
    
    if dim=='gloss':
        norm = 1 - norm
        
    return norm

def get_glyph_norms(i,universe,collvalsfile=None):
    if universe=='lml':
        lmlbounds = get_lmlbounds()
        norms = [
            get_glyph_norm(i,'thickness',lmlbounds['thickness']),
            get_glyph_norm(i,'gloss',lmlbounds['gloss']),
            get_glyph_norm(i,'roughness',lmlbounds['roughness']),
            get_glyph_norm(i,'bstar_base',lmlbounds['bstar_base'])
        ]
    elif universe=='coll':
        collbounds = get_collbounds(collvalsfile)
        norms = [
            get_glyph_norm(i,'thickness',collbounds['thickness']),
            get_glyph_norm(i,'gloss',collbounds['gloss']),
            get_glyph_norm(i,'roughness',collbounds['roughness']),
            get_glyph_norm(i,'bstar_base',collbounds['bstar_base'])
        ]
        
    return norms

