import os

os.environ["OMP_NUM_THREADS"] = "12"

import turicreate as tc


sg = tc.load_sgraph('/afs/andrew.cmu.edu/usr8/navyac/ece/pca/soc-LiveJournal1.txt',format='snap')


tri = tc.triangle_counting.create(sg)
print(tri.summary())
