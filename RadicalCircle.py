import numpy as np
from sympy import solve
from sympy.abc import x,y

def RadicalCenter2D(center_in,radius_in):
    '''
    (input)
    center_in[3,2]<float>: circle centers
    radius_in[3]<float>: circle radii

    (output)
    center_out[2]<float>: center of the radical circle, which is orthogonal to the 3 circles
    '''
    circle_equations = []
    for i in range(3):
        circle_equations.append(-2*center_in[i,0]*x-2*center_in[i,1]*y+center_in[i,0]**2+center_in[i,1]**2-radius_in[i]**2) # omitted x**2 and y**2 because they are not necessary

    res = solve([circle_equations[1]-circle_equations[0],circle_equations[2]-circle_equations[1]],x,y) # radical center is the intersection of 2 radical axes

    return np.array((res[x],res[y]),dtype=float)

def RadicalCenter3D(center_in,radius_in):
    '''
    (input)
    center_in[3,3]<float>: circle centers
    radius_in[3]<float>: circle radii

    (output)
    center_out[3]<float>: center of the radical circle, which is orthogonal to the 3 circles
    '''
    ## Two vectors from one center to another
    v1 = center_in[1]-center_in[0]
    v2 = center_in[2]-center_in[0]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    cos = np.dot(v1,v2)/n1/n2
    sin = np.sqrt(np.clip(1-cos**2,0,1))

    ## Compute a radical center projected onto a 2D space
    center_in_2D = np.zeros((3,2))
    center_in_2D[1,0] = np.linalg.norm(v1)
    center_in_2D[2] = cos*n2, sin*n2
    center_2D = RadicalCenter2D(center_in_2D,radius_in)

    ## Obtain linear combination coefficients for the radical center analytically
    a2 = center_2D[1]/(sin*n2)
    a1 = (center_2D[0]-a2*cos*n2)/n1
    center_3D = center_in[0]+a1*v1+a2*v2

    return center_3D

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
