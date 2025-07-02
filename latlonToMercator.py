import numpy as np

def latlonToMercator(lat,lon,scale):
    er = 6378137;
    mx = scale * lon * np.pi * er / 180;
    my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )
    
    return mx,my