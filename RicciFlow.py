import numpy as np
import scipy as sp
import RadicalCircle

def Euler_Characteristic(vert,face):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity

    (output)
    <int>: Euler characteristic number (V+E-F)
    '''
    edge1 = np.vstack((face[:,0],face[:,1])).T
    edge2 = np.vstack((face[:,1],face[:,2])).T
    edge3 = np.vstack((face[:,2],face[:,0])).T
    edge = np.concatenate((edge1,edge2,edge3))
    edge.sort(axis=1)
    edge_unique = np.unique(edge,axis=0)
    
    return len(vert)-len(edge_unique)+len(face)

def Edge_Length(vert,face):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity

    (output)
    edge_len[nf,3]<float>: edge lengths of each face
    '''
    edge_vec = vert[face[:,[1,2,0]]]-vert[face]
    edge_len = np.linalg.norm(edge_vec,axis=2)

    return edge_len

def Gaussian_Curvature_from_Edge_Length(face,edge_len,is_boundary):
    '''
    (input)
    face[nf,3]<int>: face connectivity
    edge_len[nf,3]<float>: edge lengths of each face
    is_boundary[nv]<bool>: True if the vertex is on the boundary

    (output)
    gauss<float>[nv]: discrete Gaussian curvatures at the vertices
    '''
    gauss = np.ones_like(is_boundary)*2.0*np.pi # Interior nodes' Gaussian curvature is 2*pi-Sum(angles)
    gauss[is_boundary] -= np.pi # Boundary nodes' Gaussian curvature is pi-Sum(angles)

    cc = (edge_len**2+edge_len[:,[2,0,1]]**2-edge_len[:,[1,2,0]]**2)/(2*edge_len*edge_len[:,[2,0,1]])
    if np.any(cc>1.0) or np.any(cc<-1.0):
        raise Exception("u_change_factor might be too large. The triangle inequality condition has been violated.")

    np.subtract.at(gauss, face.flatten(), np.arccos(cc).flatten())

    return gauss

def Inversive_Distance(vert,face):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity

    (output)
    Iij[nf,3]<float>: inversive distance associated with each edge on the faces
    edge_len_init[nf,3]<float>: edge lengths of each face
    gamma_init[nv]<float>: radii of circles at the vertices
    '''

    ## Edge length
    edge_len_init = Edge_Length(vert,face)

    ## gamma_ijk
    gamma_ijk = (edge_len_init + edge_len_init[:,[2,0,1]] - edge_len_init[:,[1,2,0]])/2.0

    ## gamma_init = min_i (gamma_ijk), which is the initial radii of circles
    gamma_init = np.full(len(vert), np.inf)  # Initialize gamma_init with np.inf
    np.minimum.at(gamma_init, face.flatten(), gamma_ijk.flatten()) # Update gamma_init using element-wise minimum operation

    ## Inversive distance
    ## NOTE: discrete conformal deformation is to change gamma only, while PRESERVING inversive distances
    g1 = gamma_init[face]
    g2 = gamma_init[face[:,[1,2,0]]]
    Iij = (edge_len_init**2-g1**2-g2**2)/(2*g1*g2)

    return Iij, edge_len_init, gamma_init

def Is_Boundary_Node(vert,face):

    edge_face = np.zeros((len(vert),len(vert)),dtype=int)
    for i in range(len(face)):
        for j in range(3):
            n1 = min(face[i,j],face[i,(j+1)%3])
            n2 = max(face[i,j],face[i,(j+1)%3])
            edge_face[n1,n2] += 1

    is_boundary = np.zeros(len(vert),dtype=bool)
    for i in range(len(vert)):
        for j in range(len(vert)):
            if edge_face[i,j] == 1:
                is_boundary[[i,j]] = True
    
    return is_boundary

def Optimize_Ricci_Energy(vert,face,gauss_target,is_boundary=None,n_step=20,tol=1.0e-8,u_change_factor=0.1,boundary_free=False):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity
    gauss_target[nv]: target Gaussian curvatures
    is_boundary[nv]<bool>: True if the vertex is on the boundary
    n_step<int>: number of updating the inversive circle packing metric
    u_change_factor<float>: stepsize to change the values of u
    boundary_free<bool>: boundary metric is unchanged if True

    (output)
    gamma[nv]<float>: radii of circles at the vertices
    Iij[nf,3]<float>: inversive distance associated with each edge on the faces
    edge_len[nf,3]<float>: edge lengths of each face
    
    (NOTE)
    An appropriate value should be assigned to u_change_factor because of the following tradeoff:
    if large: faster convergence, larger risk of divergence
    if small: slower convergence, smaller risk of divergence
    '''

    '''
    Step 1: Identify the boundary nodes from mesh connectivity
    '''
    if is_boundary is None:
        is_boundary = Is_Boundary_Node(vert,face) 

    '''
    Step 2: Compute the initial circle packing metric, i.e., inversive distances
    '''
    Iij, edge_len, gamma = Inversive_Distance(vert,face)

    '''
    Step 3: Initial scalar functions (u) and Gaussian curvatures (gauss)
    '''
    u = np.log(gamma)
    gauss = Gaussian_Curvature_from_Edge_Length(face,edge_len,is_boundary)

    '''
    Step 4: Optimize Ricci Energy
    '''
    for iter in range(n_step):

        ## Edge lengths
        for i in range(len(face)):
            for j in range(3):
                g1 = gamma[face[i,j]]
                g2 = gamma[face[i,(j+1)%3]]
                edge_len[i,j] = np.sqrt(g1**2+g2**2+2*Iij[i,j]*g1*g2)

        ## Discrete Gaussian curvatures at the vertices
        gauss = Gaussian_Curvature_from_Edge_Length(face,edge_len,is_boundary)
        gauss_error = gauss_target - gauss

        ## Display error and check convergence criterion
        print(f'Iter {iter+1}: Error max: {np.max(abs(gauss_error))}')
        # print(f'Gaussian curvatures: {gauss}')
        if np.max(abs(gauss_error)) < tol:
            print(f"Converged. The maximum error of Gaussian curvature has been decreased below tol:{tol}).")
            break

        ## Radical centers at each face
        radical_center = RadicalCircle.RadicalCenter3D(vert[face],gamma[face])

        # Draw.Draw_Shape(vert,face,True,pt=None)

        ## Distance from the radical center to the edge
        h = RadicalCircle.Distance_Radical_Center_To_Edge(vert,face,radical_center)

        ## Hessian
        Hessian = np.zeros((len(vert),len(vert)))
        for j in range(3): # (nondiagonal elements)
            Hessian[face[:,j],face[:,(j+1)%3]] -= h[:,j]/edge_len[:,j]
        Hessian += Hessian.T
        np.fill_diagonal(Hessian,-np.sum(Hessian,axis=1)) # (diagonal elements)   

        ## Update u
        ## (NOTE): np.linalg.solve and sp.linalg.solve do not work due to ill-conditioned (i.e., singular) Hessian matrix
        if boundary_free:
            mu = sp.sparse.linalg.lsqr(Hessian[~is_boundary][:,~is_boundary],gauss_error[~is_boundary])[0]
            u[~is_boundary] += u_change_factor*mu
            gauss_error[is_boundary] = 0.0 # Ignore Gaussian curvatures at the boundary
        else:
            mu = sp.sparse.linalg.lsqr(Hessian,gauss_error)[0] # Compute a least-square solution of the linear system of equations
            u += u_change_factor*mu
            
        # Update gamma
        gamma = np.exp(u)
    if iter == n_step-1:
        print(f"Maximum iteration limit (n_step:{n_step}) reached")
    return gamma, Iij, edge_len
