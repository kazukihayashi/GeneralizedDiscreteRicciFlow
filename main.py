import numpy as np
import Draw
import RicciFlow
import Flatten

## Read vertex coordinates
vert1 = np.loadtxt("vertex_srf.dat", delimiter=",")

## Read face-node relation
face1 = np.loadtxt("face_srf.dat",dtype=int, delimiter=",")

## Draw initial shape
# Draw.Draw_Shape(vert1,face1,True)

## Euler characteristic
# ec = RicciFlow.Euler_Characteristic(vert1,face1)

## Identify boundary nodes
is_boundary = RicciFlow.Is_Boundary_Node(len(vert1),face1)

## Assign target Gaussian curvature
'''
(NOTE): The sum of the target Gaussian curvatures must satisfy
(Euler characteristic)*2*pi if boundary_free option is False when using the RicciFlow.Optimize_Ricci_Energy method.
The resulting surface can be flattened by assigning positive target Gaissuan curvatures
to some (or all) boundary nodes and 0 target Gaussian curvatures to the others.
'''
gauss_target = np.zeros(len(vert1))

## Optimize Ricci energy for discrete conformal deformation
edge_len,u = RicciFlow.Optimize_Ricci_Energy(vert1,face1,gauss_target,n_step=200,u_change_factor=0.5,boundary_free=True)
conformal_factor = np.exp(2*u)

## Flatten the 3D triangular mesh isometrically on the plane
cut =[]
face_cut, cut, is_boundary, [conformal_factor,vert_cut] = Flatten.Cut(face1,cut,is_boundary,[conformal_factor,vert1])
vert_2D_cut = Flatten.Flatten_Mesh(face_cut,edge_len,len(is_boundary))

## Draw flattened shape (edges with large error will be highlighted in red)
Draw.Draw_Shape(vert_2D_cut,face_cut,False,edge_len,edge=cut,conformal_factor=conformal_factor)
Draw.Draw_Shape(vert_cut,face_cut,False,edge=cut,conformal_factor=conformal_factor)
