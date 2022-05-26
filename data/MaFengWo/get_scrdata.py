import numpy as np
import pandas as pd
import os

path = '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HAN-CDGR/data/MaFengWo/'

yelp = np.loadtxt(path + 'yelpRatingTrain.txt', delimiter='\t').astype(np.int)

np.savetxt(os.path.join(path, 'yelpUserRatingTrain.txt'), yelp[:,0:2], fmt='%d', delimiter=' ')

print('Done')