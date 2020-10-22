import numpy as np

'''
Implementation of a 3D-cube.
@param size: 3-tuple with (x,y,z) lengths
@param center: a point (x_0, y_0, z_0) that the cuboid is centered about
@return: vertices and faces of the cuboid (V, F)
@param scale: Scale of the solid
'''
def make_cuboid(size, center=(0.,0.,0.), scale=1.0, refining=1):
    # For easier xml parsing
    if center is None:
        center = (0., 0., 0.)
    if scale is None:
        scale = 1.0

    if len(size) != 3:
        raise Exception("Size must have 3 components")

    if len(center) != 3:
        raise Exception("Center must be a 3-dimensional vector")

    x, y, z = size
    # The vertices of the cuboid
    V = np.array([[-x, y, z], [-x, y, -z], [x, y, -z], [x, y, z],
                  [x, -y, z], [x, -y, -z], [-x ,-y, -z], [-x, -y, z]])
    # Making sure the scale is correct and that the cube is centered around 'center'
    V = (V * scale / 2.0 + center)
    # Faces of the cuboid (ensures outward normals)
    F = np.array([[3,1,0],[3,2,1],
                  [4,2,3],[4,5,2],
                  [4,0,7],[4,3,0],
                  [7,5,4],[7,6,5],
                  [0,6,7],[0,1,6],
                  [6,2,5],[6,1,2]], dtype=np.int32)


    for _ in range(refining):
        FRefined = []
        VRefined = []
        for i,f in enumerate(F):
            v0 = V[f[0]]
            v1 = V[f[1]]
            v2 = V[f[2]]
            v3 = [(v0[0] + v1[0]) / 2.0, (v0[1] + v1[1]) / 2.0, (v0[2] + v1[2]) / 2.0]
            v4 = [(v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0, (v1[2] + v2[2]) / 2.0]
            v5 = [(v2[0] + v0[0]) / 2.0, (v2[1] + v0[1]) / 2.0, (v2[2] + v0[2]) / 2.0]
            # Add the new vertices to VRefined
            VRefined.append(v0)
            VRefined.append(v1)
            VRefined.append(v2)
            VRefined.append(v3)
            VRefined.append(v4)
            VRefined.append(v5)
            # Add the new faces to FRefined
            idx = i * 6
            FRefined.append([idx, idx+3, idx+5])
            FRefined.append([idx+1, idx+4, idx+3])
            FRefined.append([idx+2, idx+5, idx+4])
            FRefined.append([idx+3, idx+4, idx+5])
        V = np.array(VRefined)
        F = np.array(FRefined, dtype=np.int32)
    return V, F

def spherify_cube(V, F, radius):
    V_ = []
    center = np.mean(V, axis=0)
    for vi, v in enumerate(V):
        v_ = (v - center)
        v_ = v_ / np.linalg.norm(v_) * radius
        V_.append(v_)
    return np.array(V_), F

def coolify_sphere(V, F, tval):
    V_ = []
    center = np.mean(V, axis=0)
    for vi, v in enumerate(V):
        height = 1. + np.sin(v[1] * tval) * 0.05
        v_ = (v - center) / np.linalg.norm(v - center)
        V_.append(height * v_)
    return np.array(V_), F
