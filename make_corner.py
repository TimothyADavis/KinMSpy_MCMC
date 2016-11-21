import numpy as np
import corner

def make_corner(filename="chain.dat",discard=0,labels=None,truths=None):
  f = np.loadtxt(filename,skiprows=discard) 
  samples=f[:,1:]
  fig = corner.corner(samples, labels=labels,truths=truths)
  fig.savefig("triangle.png")

  mymap = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
  bestfit=np.array(list(mymap))
  print("Best fit values, Upper Error, Lower Error")
  print(bestfit)
  print("Just the best fit")
  print(bestfit[:,0])
  
