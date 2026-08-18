
import numpy as np

def generate_linear_data(n=300):
    x=np.random.rand(n,2)
    y=(x[:,0]+ x[:,1] > 0).astype(int)
    return x,y

