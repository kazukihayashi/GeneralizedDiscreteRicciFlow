import numpy as np
import Draw
import RicciFlow
import Flatten

## Read vertex coordinates
# vert1 = np.loadtxt("vertex_small_srf.dat")
vert1 = np.loadtxt("vertex_large_srf.dat", delimiter=",")
# vert1 += np.random.rand(*vert1.shape)*0.5
# vert1 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float)

## Read face-node relation
# face1 = np.loadtxt("face_small_srf.dat",dtype=int)
face1 = np.loadtxt("face_large_srf.dat",dtype=int, delimiter=",")
# face1 = np.array([[0,1,2],[0,2,3],[0,1,5],[0,5,4],[1,2,5],[2,6,5],[2,3,6],[3,6,7],[0,3,4],[3,4,7],[4,5,6],[4,6,7]],dtype=int)

## Draw initial shape
Draw.Draw_Shape(vert1,face1,True)

## Euler characteristic
ec = RicciFlow.Euler_Characteristic(vert1,face1)

## Identify boundary nodes
is_boundary = RicciFlow.Is_Boundary_Node(vert1,face1)

## Assign target Gaussian curvature
'''
(NOTE): The sum of the target Gaussian curvatures must satisfy
(Euler characteristic)*2*pi.
The resulting surface can be flattened by assigning positive target Gaissuan curvatures
to some (or all) boundary nodes and 0 target Gaussian curvatures to the others.
'''
gauss_target = np.zeros(len(vert1))
# gauss_target[[0,1,2,3]] = np.pi
# gauss_target[[0,4,20,24]] = np.pi/2
# gauss_target[is_boundary] = 2*np.pi/sum(is_boundary) # small_srf
# gauss_target[is_boundary] = 4*np.pi/3/sum(is_boundary) # small_srf
# gauss_target[[12,7,13]] = 2*np.pi/3
gauss_target[[203,217,232,250]] = np.pi/2 # large_srf

## Optimize Ricci energy for discrete conformal deformation
gamma,Iij,edge_len = RicciFlow.Optimize_Ricci_Energy(vert1,face1,gauss_target,n_step=200,u_change_factor=0.5,boundary_free=True)
# np.savetxt("edge_length.dat",edge_len)

## Flatten the 3D triangular mesh isometrically on the plane
cut = []#[[13,14],[12,11,10],[7,2]] # The order of specification must be from inner to outer. Please specify [] if no cut.
face_cut, cut, is_boundary, _ = Flatten.Cut(face1,cut,is_boundary,None)
vert_2D_cut = Flatten.Flatten_Mesh(face_cut,edge_len,len(is_boundary)) # This has error so far. I have to fix this.

# Extract unique edges and output their connectivity and lengths
edge_unique, edge_len_unique = RicciFlow.Edge_Unique(face1,edge_len)
np.savetxt("edge.dat",edge_unique,fmt='%d')
np.savetxt("edge_length.dat",edge_len_unique)

## Draw flattened shape (edges with large error will be highlighted in red)
Draw.Draw_Shape(vert_2D_cut,face_cut,True,edge_len,edge=cut)
