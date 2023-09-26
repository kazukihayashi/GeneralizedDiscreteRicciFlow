import numpy as np
import Draw

def Construct_Basic_Triangle_with_3_Edge_Lengths(l3):
    '''
    (input)
    l3[3]<float>: 3 edge lengths

    (output)
    v[3,2]<float>: 2D positions of triangle corners
    '''

    v = np.zeros((3,2)) # v[0] = (0,0) 
    v[1] = (l3[0],0)
    cos = (l3[1]**2-l3[0]**2-l3[2]**2)/(2*l3[0]*l3[2])
    sin = np.sqrt(np.clip(1-cos**2,0,1))
    v[2] = (l3[2]*cos,l3[2]*sin)

    if not np.isclose(np.linalg.norm(v[2]-v[1]),l3[1]):
        v[1] *= -1
    
    return v

def Mirror(v,axis_v2):
    '''
    (input)
    v[2]<float>: 2D position of point to be mirrored
    axis_v2[2,2]<float>: symmetric axis defined by two points to mirror the point
    (output)
    v_mirror[2]<float> 2D position of mirrored point
    '''
    normalized_axis_vec = (axis_v2[1]-axis_v2[0])/np.linalg.norm(axis_v2[1]-axis_v2[0])
    v_vec = v-axis_v2[0]
    dot_product = np.dot(v_vec,normalized_axis_vec)
    v_mirror = 2*dot_product*normalized_axis_vec -v + 2*axis_v2[0]

    return v_mirror

def Construct_Triangle_with_2_Known_Vertices(l3,v_pos2,v_known3,v_pos_adj):
    """
    (input)
    l3[3]<float> 3 edge lengths
    v_pos2[2,2]<float> 2 known vertex positions
    v_known3[3]<bool> True if the vertex positions are known

    (output)
    v_out[2]<float> revealed vertex position
    """
    v_basic = Construct_Basic_Triangle_with_3_Edge_Lengths(l3)
    v_basic2 = v_basic[v_known3]
    vec_basic = v_basic2[1]-v_basic2[0]
    vec_true = v_pos2[1]-v_pos2[0]
    
    cos = np.dot(vec_basic,vec_true)/(np.linalg.norm(vec_basic)*np.linalg.norm(vec_true))
    sin = np.dot(vec_basic/np.linalg.norm(vec_basic),np.array([vec_true[1],-vec_true[0]])/np.linalg.norm(vec_true))
    if v_known3[0]:
        v = np.dot(np.array([[cos,-sin],[sin,cos]]),v_basic.T).T+v_pos2[0]
    else: # This implies v_known3[1] == True
        v = np.dot(np.array([[cos,-sin],[sin,cos]]),(v_basic-v_basic2[0]).T).T+v_pos2[0]

    v_out1 = v[~v_known3]
    v_out2 = Mirror(v_out1,v_pos2)
    if np.linalg.norm(v_out1-v_pos_adj) > np.linalg.norm(v_out2-v_pos_adj):
        v_out = v_out1
    else:
        v_out = v_out2

    return v_out
    
def Flatten_Mesh(face, edge_len, nv):
    '''
    (input)
    face[nf,3]<int>: triangle element connectivity
    edge_len[nf,3]<float>: edge lengths of triangular elements
    nv<int>: number of vertices

    (output)
    vert2D[nv,2]<float>: vertex positions
    '''

    vert2D = np.zeros((nv,2))
    is_vert_constructed = np.zeros(nv,dtype=bool)

    ## Construct the first triangle
    vert2D[face[0]] = Construct_Basic_Triangle_with_3_Edge_Lengths(edge_len[0]) 
    is_vert_constructed[face[0]] = True
    face_unconstructed = np.copy(face[1:])
    face_constructed = np.array([face[0]])
    edge_len_unconstructed = np.copy(edge_len[1:])

    while not np.all(is_vert_constructed):
        c = is_vert_constructed[face_unconstructed.flatten()].reshape(face_unconstructed.shape)
        n_constructed_edge = c.sum(axis=1)

        ii = 0
        is_adj = False
        while np.all(is_adj==False):
            face_i = np.where(n_constructed_edge==2)[0][ii]
            vert_i = face_unconstructed[face_i][~c[face_i]]
            face_constructed = np.vstack([face_constructed, face_unconstructed[n_constructed_edge==3]])
            is_adj = np.logical_and(np.isin(face_constructed,face_unconstructed[face_i][c[face_i]][0]).any(axis=1),np.isin(face_constructed,face_unconstructed[face_i][c[face_i]][1]).any(axis=1))
            ii += 1

        f_adj = face_constructed[is_adj][0]
        v_pos_adj = vert2D[np.setdiff1d(f_adj,face_unconstructed[face_i])][0] # already known vertex position in the triangle sharing the same edge
        
        lll1 = edge_len_unconstructed[face_i]
        vert2D[vert_i] = Construct_Triangle_with_2_Known_Vertices(edge_len_unconstructed[face_i],vert2D[face_unconstructed[face_i][c[face_i]]],c[face_i],v_pos_adj)

        vvv = vert2D[face_unconstructed[face_i,[1,2,0]]] - vert2D[face_unconstructed[face_i]]
        lll2 = np.linalg.norm(vvv,axis=1)

        face_constructed = np.vstack([face_constructed, face_unconstructed[face_i]])
        constructed_face_i = np.append(np.where(n_constructed_edge==3)[0],face_i)
        edge_len_unconstructed = np.delete(edge_len_unconstructed,constructed_face_i,axis=0)
        face_unconstructed = np.delete(face_unconstructed,constructed_face_i,axis=0)
        is_vert_constructed[vert_i] = True

        # Draw.Draw_Shape(vert2D,face_constructed)

        if not np.all(np.isclose(lll1,lll2)):
            raise Exception("Inconsistent length")

    return vert2D

def Cut(face,cut,is_boundary_vertex,vert_attributes=None):

    if np.any(~is_boundary_vertex[[cut[i][-1] for i in cut]]):
        raise Exception("The last index of each cut must be on the boundary.")

    nv = len(is_boundary_vertex)
    face_cut = np.copy(face)
    cut_out = cut[:]
    if vert_attributes is None:
        vert_attributes_cut = None
    else:
        vert_attributes_cut = vert_attributes[:]

    count = 0
    for i in range(len(cut)):
        cut_out.append(np.copy(cut[i]))
        for j in range(1,len(cut[i])):
            cut_out[-1][j] = nv+count
            count += 1

    n_add = 0
    for single_cut_line in cut:

        f = np.where(np.logical_and(np.any(face==single_cut_line[0],axis=1),np.any(face==single_cut_line[1],axis=1)))[0][1]
        
        for i in range(len(single_cut_line)-1):
            
            neighbor_i = single_cut_line[i]
            
            ## add vert_attributes
            is_boundary_vertex = np.append(is_boundary_vertex,is_boundary_vertex[single_cut_line[i+1]])
            if vert_attributes is not None:
                for k in range(len(vert_attributes)):
                    vert_attributes_cut[k] = np.concatenate([vert_attributes_cut[k],np.expand_dims(vert_attributes[k][single_cut_line[i+1]],0)],axis=0)
                    
            while True:
                j = np.where(face[f]==single_cut_line[i+1])[0][0]
                neighbor_i2 = np.setdiff1d(face[f],[face[f,j],neighbor_i])[0]
                face_cut[f,j] = nv+n_add+i # Please modify face indexing after modifying the vertex attributes.

                if np.all(is_boundary_vertex[[face[f,j],neighbor_i2]]):
                    break # stop if next
                if i != len(single_cut_line)-2 and np.any(face[f]==single_cut_line[i+2]):
                    break # stop if next

                adj_fs_next_edge = np.isin(face,np.delete(face[f],np.where(face[f]==neighbor_i)[0][0])).sum(axis=1)
                next_f = int(np.setdiff1d(np.where(adj_fs_next_edge==2)[0],f))

                neighbor_i = int(np.setdiff1d(face_cut[f],[neighbor_i,nv+n_add+i]))
                f = next_f
            
        n_add += len(single_cut_line)-1
                
    return face_cut, cut_out, is_boundary_vertex, vert_attributes_cut