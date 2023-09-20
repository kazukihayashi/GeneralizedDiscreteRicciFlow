import numpy as np

def RadicalCenter2D(center_in,radius_in):
    '''
    (input)
    center_in[nf,3,2]<float>: circle centers
    radius_in[nf,3]<float>: circle radii

    (output)
    center_out[nf,2]<float>: centers of the radical circle, which is orthogonal to the 3 circles
    '''
    det = 0.5/((center_in[:,1,0]-center_in[:,0,0])*(center_in[:,2,1]-center_in[:,1,1])-(center_in[:,2,0]-center_in[:,1,0])*(center_in[:,1,1]-center_in[:,0,1]))
    A_inv = np.array([[center_in[:,2,1]-center_in[:,1,1],center_in[:,0,1]-center_in[:,1,1]],[center_in[:,1,0]-center_in[:,2,0],center_in[:,1,0]-center_in[:,0,0]]]).transpose((2,0,1))
    b = np.array([center_in[:,1,0]**2-center_in[:,0,0]**2+center_in[:,1,1]**2-center_in[:,0,1]**2+radius_in[:,0]**2-radius_in[:,1]**2,
                  center_in[:,2,0]**2-center_in[:,1,0]**2+center_in[:,2,1]**2-center_in[:,1,1]**2+radius_in[:,1]**2-radius_in[:,2]**2]).transpose()[:,:,np.newaxis]
    
    radical_center_2D = det[:,np.newaxis]*np.squeeze(np.matmul(A_inv,b),axis=2)

    return radical_center_2D

def RadicalCenter3D(center_in,radius_in):
    '''
    (input)
    center_in[nf,3,3]<float>: circle centers
    radius_in[nf,3]<float>: circle radii

    (output)
    center_out[nf,3]<float>: centers of the radical circle, which is orthogonal to the 3 circles
    '''
    ## Two vectors from one center to another
    v1 = center_in[:,1]-center_in[:,0]
    v2 = center_in[:,2]-center_in[:,0]
    n1 = np.linalg.norm(v1,axis=1)
    n2 = np.linalg.norm(v2,axis=1)
    cos = np.sum(v1*v2,axis=1)/n1/n2
    sin = np.sqrt(np.clip(1-cos**2,0,1))

    ## Compute a radical center projected onto a 2D space
    center_in_2D = np.zeros((len(center_in),3,2))
    center_in_2D[:,1,0] = n1
    center_in_2D[:,2,0] = cos*n2
    center_in_2D[:,2,1] = sin*n2
    center_2D = RadicalCenter2D(center_in_2D,radius_in)

    ## Obtain linear combination coefficients for the radical center analytically
    a2 = center_2D[:,1]/(sin*n2)
    a1 = (center_2D[:,0]-a2*cos*n2)/n1
    center_3D = center_in[:,0]+a1[:,np.newaxis]*v1+a2[:,np.newaxis]*v2

    return center_3D

def Distance_Radical_Center_To_Edge(vert,face,radical_center):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity
    radical_center[nf,3]<float>: centers of the radical circle, which is orthogonal to the 3 circles

    (output)
    dist[nf,3]<float>: the Euclidian shortest distance between the point and the line
    '''

    ## Distance from the radical center to the edge
    line_start = vert[face]
    line_vec = vert[face[:,[1,2,0]]]-vert[face]

    d_vecs = np.zeros([len(face),3,3])
    for j in range(3):
        d_vecs[:,j] = (radical_center-line_start[:,j])-np.sum(line_vec[:,j]*(radical_center-line_start[:,j]),axis=1,keepdims=True)*line_vec[:,j]/np.sum(line_vec[:,j]**2,axis=1,keepdims=True)

    dist = np.linalg.norm(d_vecs,axis=2)

    return dist

# ## Example of Drawing a figure of a radical center

# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

# def Shape2D(center,radius,radical_center):

#     ax = plt.figure(figsize=(10,10)).add_subplot()

#     plt.autoscale(True)

#     for i in range(len(radius)):
#         ax.add_patch(Circle(center[i],radius[i],facecolor=(0,0,0,0.2),edgecolor='black'))
#     ax.plot(np.append(center[:,0],center[0,0]),np.append(center[:,1],center[0,1]),color='black',label="triangle")
#     ax.plot(radical_center[0],radical_center[1],color='red',marker='*',label="radical center")

#     plt.legend()
#     plt.axis("equal")
#     plt.show()
#     plt.close()

# cc = np.array([[0,0,0],[5,0,0],[3,3,0]])
# rr = np.array([2,2,1])
# rcc3 = RadicalCenter3D(cc,rr)
# rcc2 = RadicalCenter2D(cc[:,0:2],rr)
# Shape2D(cc,rr,rcc2)
