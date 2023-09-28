import numpy as np

# User inputs
boundaries = [(0, 0),(10, 0),(10, 10),(0, 10)]
points = np.array([[-1,5,2,11],[-1,-4,5,11]])
_nturbs = len(points[0])
gradient=True

_boundaries = np.array(boundaries).T
_nbounds = len(_boundaries[0])

# Compute edge information
_boundary_edge = np.roll(_boundaries,-1,axis=1) - _boundaries
_boundary_len = np.sqrt(_boundary_edge[0]**2 + _boundary_edge[1]**2)
_boundary_norm = np.array([_boundary_edge[1],-_boundary_edge[0]]) / _boundary_len
_boundary_int = (np.roll(_boundary_norm,1,axis=1) + _boundary_norm) / 2


# Compute distances from turbines to boundary points
a = np.zeros((_nturbs,2,_nbounds))
for i in range(_nturbs):
    a[i] = np.expand_dims(points[:,i].T,axis=-1) - _boundaries

# Compute projections
a_edge = np.sum(a*_boundary_edge, axis=1) / _boundary_len
a_int = np.sum(a*_boundary_norm, axis=1)
sigma = np.sign(np.sum(a*_boundary_int, axis=1))

# Initialize signed distance containers
C = np.zeros(_nturbs)
D = np.zeros(_nbounds)
if gradient:
    Cx = np.zeros(_nturbs)
    Cy = np.zeros(_nturbs)

# Compute signed distance
for i in range(_nturbs):
    for k in range(_nbounds):
        if a_edge[i,k] < 0:
            D[k] = np.sqrt(a[i,0,k]**2 + a[i,1,k]**2)*sigma[i,k]
        elif a_edge[i,k] > _boundary_len[k]:
            D[k] = np.sqrt(a[i,0,(k+1)%_nbounds]**2 + a[i,1,(k+1)%_nbounds]**2)*sigma[i,(k+1)%_nbounds]
        else:
            D[k] = a_int[i,k]
    
    # Select minimum distance
    idx = np.argmin(np.abs(D))
    C[i] = D[idx]

    if gradient:
        if a_edge[i,idx] < 0:
            Cx[i] = (points[0,i] - _boundaries[0,idx]) / np.sqrt((_boundaries[0,idx]-points[0,i])**2 + (_boundaries[1,idx]-points[1,i])**2)
            Cy[i] = (points[1,i] - _boundaries[1,idx]) / np.sqrt((_boundaries[0,idx]-points[0,i])**2 + (_boundaries[1,idx]-points[1,i])**2)
        elif a_edge[i,idx] > _boundary_len[idx]:
            Cx[i] = (points[0,i] - _boundaries[0,(idx+1)%_nbounds]) / np.sqrt((_boundaries[0,(idx+1)%_nbounds]-points[0,i])**2 + (_boundaries[1,(idx+1)%_nbounds]-points[1,i])**2)
            Cy[i] = (points[1,i] - _boundaries[1,(idx+1)%_nbounds]) / np.sqrt((_boundaries[0,(idx+1)%_nbounds]-points[0,i])**2 + (_boundaries[1,(idx+1)%_nbounds]-points[1,i])**2)
        else:
            Cx[i] = -(_boundaries[1,idx] - _boundaries[1,(idx+1)%_nbounds]) / _boundary_len[idx]
            Cy[i] = -(_boundaries[0,(idx+1)%_nbounds] - _boundaries[0,idx]) / _boundary_len[idx]

# # Compute distances from turbines to boundary points
# a = np.zeros((_nturbs,2,_nbounds))
# for i in range(_nturbs):
#     a[i] = np.expand_dims(points[:,i].T,axis=-1) - _boundaries
# # Compute projections
# a_edge = np.sum(a*_boundary_edge, axis=1) / _boundary_len
# a_int = np.sum(a*_boundary_norm, axis=1)
# sigma = np.sign(np.sum(a*_boundary_int, axis=1))

# # Initialize signed distance containers
# C = np.zeros(_nturbs)
# D = np.zeros(_nbounds)
# if gradient:
#     Cx = np.zeros(_nturbs)
#     Cy = np.zeros(_nturbs)

# # Compute signed distance
# for i in range(_nturbs):
#     for k in range(_nbounds):
#         if a_edge[i,k] < 0:
#             D[k] = np.sqrt(a[i,0,k]**2 + a[i,1,k]**2)*sigma[i,k]
#         elif a_edge[i,k] > _boundary_len[k]:
#             D[k] = np.sqrt(a[i,0,(k+1)%_nbounds]**2 + a[i,1,(k+1)%_nbounds]**2)*sigma[i,k]
#         else:
#             D[k] = a_int[i,k]

# # Select minimum distance
#     idx = np.argmin(np.abs(D))
#     C[i] = D[idx]

#     if gradient:
#         if a_edge[i,idx] < 0:
#             Cx[i] = (points[0,i] - _boundaries[0,idx]) / np.sqrt((_boundaries[0,idx]-points[0,i])**2 + (_boundaries[1,idx]-points[1,i])**2)
#             Cy[i] = (points[1,i] - _boundaries[1,idx]) / np.sqrt((_boundaries[0,idx]-points[0,i])**2 + (_boundaries[1,idx]-points[1,i])**2)
#         elif a_edge[i,idx] > _boundary_len[idx]:
#             Cx[i] = (points[0,i] - _boundaries[0,(idx+1)%_nbounds]) / np.sqrt((_boundaries[0,(idx+1)%_nbounds]-points[0,i])**2 + (_boundaries[1,(idx+1)%_nbounds]-points[1,i])**2)
#             Cy[i] = (points[1,i] - _boundaries[1,(idx+1)%_nbounds]) / np.sqrt((_boundaries[0,(idx+1)%_nbounds]-points[0,i])**2 + (_boundaries[1,(idx+1)%_nbounds]-points[1,i])**2)
#         else:
#             Cx[i] = (_boundaries[1,idx] - _boundaries[1,(idx+1)%_nbounds]) / _boundary_len[idx]
#             Cy[i] = (_boundaries[0,(idx+1)%_nbounds] - _boundaries[0,idx]) / _boundary_len[idx]


# ne = len(boundaries)
# nx = len(point[0])

# # Compute independent of points
# boundaries = np.array(boundaries).T
# e = np.roll(boundaries, -1) - boundaries
# e_mag = (np.sqrt(e[0]**2 + e[1]**2))
# e = e / e_mag
# n = np.array([-e[1],e[0]])
# q = np.roll(n, -1) + n

# # Compute distances for each point
# a = np.zeros((nx,2,ne))
# for i in range(nx):
#     a[i] = np.expand_dims(point[:,i].T,axis=-1) - boundaries

# a_tilde = np.sum(a*e, axis=1)
# a_hat = np.sum(a*n, axis=1)
# sigma = np.sign(np.sum(a*q, axis=1))

# C = np.zeros(nx)
# Cx = np.zeros(nx)
# Cy = np.zeros(nx)
# D = np.zeros(ne)
# for i in range(nx):
#     for k in range(ne):
#         if a_tilde[i,k] < 0:
#             D[k] = np.sqrt(a[i,0,k]**2 + a[i,1,k]**2)*sigma[i,k]
#         elif a_tilde[i,k] > e_mag[k]:
#             D[k] = np.sqrt(a[i,0,(k+1)%ne]**2 + a[i,1,(k+1)%ne]**2)*sigma[i,k]
#         else:
#             D[k] = a_hat[i,k]
#     idx = np.argmin(np.abs(D))
#     C[i] = D[idx]

#     if a_tilde[i,idx] < 0:
#         Cx[i] = (point[0,i] - boundaries[0,idx]) / np.sqrt((boundaries[0,idx]-point[0,i])**2 + (boundaries[1,idx]-point[1,i])**2)
#         Cy[i] = (point[1,i] - boundaries[1,idx]) / np.sqrt((boundaries[0,idx]-point[0,i])**2 + (boundaries[1,idx]-point[1,i])**2)
#     elif a_tilde[i,idx] > e_mag[idx]:
#         Cx[i] = (point[0,i] - boundaries[0,(idx+1)%ne]) / np.sqrt((boundaries[0,(idx+1)%ne]-point[0,i])**2 + (boundaries[1,(idx+1)%ne]-point[1,i])**2)
#         Cy[i] = (point[1,i] - boundaries[1,(idx+1)%ne]) / np.sqrt((boundaries[0,(idx+1)%ne]-point[0,i])**2 + (boundaries[1,(idx+1)%ne]-point[1,i])**2)
#     else:
#         Cx[i] = (boundaries[1,idx] - boundaries[1,(idx+1)%ne]) / e_mag[idx]
#         Cy[i] = (boundaries[0,(idx+1)%ne] - boundaries[0,idx]) / e_mag[idx]


print(C)
print(Cx)
print(Cy)

