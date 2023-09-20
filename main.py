import numpy as np
import Draw
import RicciFlow
import Flatten

## Read vertex coordinates
vert1 = np.loadtxt("vertex_developable.dat")#np.loadtxt("vertex.dat", delimiter=",")
# vert1 += np.random.rand(*vert1.shape)*0.5
# vert1 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float)

## Read face-node relation
face1 = np.loadtxt("face_developable.dat",dtype=int)-1#np.loadtxt("face.dat", delimiter=",",dtype=int)
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
gauss_target[is_boundary] = 2*np.pi/sum(is_boundary)

## Optimize Ricci energy for discrete conformal deformation
gamma,Iij,edge_len = RicciFlow.Optimize_Ricci_Energy(vert1,face1,gauss_target,n_step=500,u_change_factor=0.1,boundary_free=False)

## Flatten the 3D triangular mesh isometrically on the plane
vert_2D = Flatten.Flatten_Mesh(face1,edge_len,len(vert1))

## Draw flattened shape (edges with large error will be highlighted in red)
Draw.Draw_Shape(vert_2D,face1,True,edge_len)
