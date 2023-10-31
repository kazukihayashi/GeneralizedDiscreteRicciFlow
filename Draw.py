import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def Log_Gauss_Error_History(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_yscale('log')
    fig.show()

def Draw_Shape(vert,face,annotate=False,target_edge_length=None,pt=None,edge=None,conformal_factor=None):
    '''
    (input)
    vert[nv,2 or 3]<float>: vertex positions
    face[nf,3]<int>: face connectivity
    annotate<bool>: enable node annotation if True
    target_edge_length[nf,3]: compute the edge lengths and highlight edges in red if their lengths deviate from the target value
    pt[n_point,2 or 3]<float>: additional points to show

    (output)
    None
    '''

    if vert.shape[1] == 3: # 3D
        ax = plt.subplot(projection='3d')
        vert_facewise = vert[face.flatten()].reshape((face.shape[0],3,3)) # (number of faces) x (3: triangle corners) x (3: xyz)
        ax.add_collection(Poly3DCollection([tri for tri in vert_facewise],facecolor='gray',edgecolor='black',linewidths=0.3,alpha=0.3))
        if conformal_factor is not None:
            tcf = ax.scatter3D(vert[:,0], vert[:,1], vert[:,2], c=conformal_factor, cmap="bwr_r", vmin=0.0, vmax=2.0, marker='o', edgecolors='black',linewidths=0.3)
            plt.colorbar(tcf,ticks=[0.0,0.5,1.0,1.5,2.0],extend='max',label="conformal factor")

    elif vert.shape[1] == 2: # 2D
        fig, ax = plt.subplots()
        tcf = ax.tricontourf(vert[:,0], vert[:,1], conformal_factor, cmap="bwr_r", vmin=0.0, vmax=None, levels = np.linspace(0.0,2.0,50), extend="max")
        vert_facewise = vert[face.flatten()].reshape((face.shape[0],3,2)) # (number of faces) x (3: triangle corners) x (2: xy)
        ax.add_collection(PolyCollection([tri for tri in vert_facewise],facecolor=(0,0,0,0),edgecolor=(0.5,0.5,0.5),linewidths=0.5))
        ax.text(*vert[np.argmax(conformal_factor)],f"max:{np.max(conformal_factor):.2f}",color="blue",bbox=dict(facecolor='white', alpha=0.4, edgecolor='None'))
        ax.plot(*vert[np.argmax(conformal_factor)],'bo')
        ax.text(*vert[np.argmin(conformal_factor)],f"min:{np.min(conformal_factor):.2f}",color="red",bbox=dict(facecolor='white', alpha=0.4, edgecolor='None'))
        ax.plot(*vert[np.argmin(conformal_factor)],'ro')
        fig.colorbar(tcf,ticks=[0.0,0.5,1.0,1.5,2.0],label="conformal factor")
        
    if annotate:
        for i in range(len(vert)):
            ax.text(*vert[i],str(i))
        for i in range(len(face)):
            ax.text(*np.mean(vert[face[i]],axis=0),str(i),color="green")
    
    if target_edge_length is not None: # the edges are highlighted in red if the edge length does not match with the target value
        edge_length = Edge_Length(vert,face)
        for i in range(len(face)):
            for j in range(3):
                if not np.isclose(edge_length[i,j],target_edge_length[i,j]):
                    ax.add_line(plt.Line2D([vert[face[i,j]][0],vert[face[i,(j+1)%3]][0]],[vert[face[i,j]][1],vert[face[i,(j+1)%3]][1]], color="r"))

    if pt is not None:
        if pt.shape[1] == 2:
            ax.scatter(pt[:,0],pt[:,1],c="blue")
        else:
            ax.scatter3D(pt[:,0],pt[:,1],pt[:,2],c="blue")
    if edge is not None:
        if vert.shape[1] == 3: # 3D
            for e in edge:
                ax.plot3D(vert[e,0],vert[e,1],vert[e,2],color='orange',linestyle='--',linewidth=2)
        elif vert.shape[1] == 2: # 2D
            for e in edge:
                ax.plot(vert[e,0],vert[e,1],color='orange',linestyle='--',linewidth=2)

    ax.axis("equal")
    plt.axis('off')
    plt.autoscale(False)
    plt.show()
    plt.close()

def Edge_Length(vert,face):
    '''
    (input)
    vert[nv,3]<float>: vertex positions
    face[nf,3]<int>: face connectivity

    (output)
    edge_len[nf,3]<float>: edge lengths of each face
    '''

    if vert.shape[1] == 2:
        vert = np.hstack([vert,np.zeros([len(vert),1])])

    edge_vec = np.zeros((len(face),3,3))
    for i in range(len(face)):
        for j in range(3):
            edge_vec[i,j] = vert[face[i,(j+1)%3]] - vert[face[i,j]]
    edge_len = np.linalg.norm(edge_vec,axis=2)

    return edge_len
