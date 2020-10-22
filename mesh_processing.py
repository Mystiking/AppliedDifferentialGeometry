import jax.numpy as jnp
import numpy as np

def objloader(folder, fname):
    V = []
    UV = []
    F = []
    with open(folder + fname, 'r') as f:
        lines = f.readlines()
        for l in lines:
            token = l.split(' ')[0]
            if token == 'v':
                V.append(jnp.array([float(v) for v in l.split(' ')[1:]]))
            if token == 'vt':
                UV.append(jnp.array([float(v) for v in l.split(' ')[1:]]))
            if token == 'f':
                F.append(jnp.array([int(f.split('/')[0]) - 1 for f in l.split(' ')[1:]]))
    V = jnp.array(V)
    UV = jnp.array(UV)
    if UV.shape[0] != 0:
        # Reorder F
        pass
    F = jnp.array(F, dtype=jnp.int32)
    return V, F

def compute_vertex_neighbourhood_matrix(V, F):
    N = np.zeros((V.shape[0], V.shape[0]), dtype=np.int)
    for f in F:
        N[f[0], [f[1], f[2]]] = 1
        N[f[1], [f[0], f[2]]] = 1
        N[f[2], [f[0], f[1]]] = 1
    return jnp.array(N)

def compute_face_neighbourhood_matrix(V, F, E):
    N = np.zeros((V.shape[0], F.shape[0]), dtype=np.int)
    for fi, f in enumerate(F):
        N[f, fi] = 1
    return jnp.array(N)

def compute_edge_connectivity(V, F):
    E = np.zeros((V.shape[0], V.shape[0]), dtype=np.int)
    E[F[:, 0], F[:, 1]] = 1
    E[F[:, 1], F[:, 2]] = 1
    E[F[:, 2], F[:, 0]] = 1
    # Length of edges
    E_lens = np.zeros((V.shape[0], V.shape[0]))
    E_lens[F[:, 0], F[:, 1]] = jnp.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
    E_lens[F[:, 1], F[:, 2]] = jnp.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1)
    E_lens[F[:, 2], F[:, 0]] = jnp.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)
    return jnp.array(E), jnp.array(E_lens)

def compute_normals_and_areas(V, F):
    Ns = jnp.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:,2]] - V[F[:, 0]])
    A = jnp.linalg.norm(Ns, axis=1) * 0.5
    Ns = Ns / jnp.linalg.norm(Ns, axis=1).reshape(Ns.shape[0], 1)
    return jnp.array(Ns), jnp.array(A)

def compute_voronoi_cell_areas(V, F, NF):
    VA = np.zeros((V.shape[0]))
    for vi, v in enumerate(V):
        for f in F[NF[vi] == 1]:
            vs = V[f]
            # Get mid-point
            mid_point = jnp.mean(vs, axis=0)
            # Get edge mid-points
            vj, vk = [fi for fi in f if fi != vi]
            meij = (V[vj] - v) / 2.
            meik = (V[vk] - v) / 2.
            meim = (mid_point - v)
            # Compute the two triangle areas
            VA[vi] += jnp.linalg.norm(jnp.cross(meij, meim))
            VA[vi] += jnp.linalg.norm(jnp.cross(meik, meim))
    return VA

def compute_mesh_dual_points(V, F, E):
    return jnp.mean(V[F], axis=1)

def compute_vertex_face_participation_matrix(V, F):
    VF = np.zeros((V.shape[0], F.shape[0]), dtype=np.int)
    for vi, _ in enumerate(V):
        VF[vi, np.where(F == vi)[0]] = 1
    return VF

def compute_dual_edges(V, F, D, VF, E):
    DE = np.zeros((E.shape[0], E.shape[1],3 ))
    DEl = np.zeros((E.shape[0], E.shape[1]))
    DA = np.zeros((V.shape[0]))

    Eindices = np.where(E == 1)

    E0 = Eindices[0] # "From" vertices
    E1 = Eindices[1] # "To" vertices

    F_range = jnp.array([jnp.arange(F.shape[0])])
    F_pairs = F_range[(VF[E0] & VF[E1]) == 1] # Get the face pairs sharing each edge
    F_pairs = F_pairs.reshape((F_pairs.shape[0] // 2, 2))

    DE[E0, E1] = D[F_pairs[:, 1]] - D[F_pairs[:, 0]]
    for i in range(DA.shape[0]):
        DA[i] = 0.5 * jnp.sum(jnp.linalg.norm(jnp.cross(V[i, :] - (-1)*DE[i, np.where(E[i, :] == 1)], DE[i, np.where(E[i, :] == 1)]), axis=2))
    DEl = jnp.linalg.norm(DE, axis=2)
    return DEl, DE, DA

def precompute_mesh_attributes(folder, fname):
    import time
    now = time.time()
    V, F = objloader(folder, fname)
    print("Loading in mesh:", time.time() - now, "s")
    now = time.time()
    E, El = compute_edge_connectivity(V, F)
    print("edge connectivity + lens:", time.time() - now, "s")
    now = time.time()
    NV = compute_vertex_neighbourhood_matrix(V, F)
    print("vertex neighbourhood connectivity:", time.time() - now, "s")
    now = time.time()
    NF = compute_face_neighbourhood_matrix(V, F, E)
    print("face neighbourhood connectivity:", time.time() - now, "s")
    now = time.time()
    N, A = compute_normals_and_areas(V, F)
    print("normals + area:", time.time() - now, "s")
    now = time.time()
    D = compute_mesh_dual_points(V, F, E)
    print("dual points:", time.time() - now, "s")
    now = time.time()
    VF = compute_vertex_face_participation_matrix(V, F)
    print("vertex - face adjacency:", time.time() - now, "s")
    now = time.time()
    DEl, DE, DA = compute_dual_edges(V, F, D, VF, E)
    print("dual edge computation:", time.time() - now, "s")
    return V, F, NF, E, NV, N, A, VF, D, El, DEl, DE, DA

def compute_mesh_attributes(V, F):
    E, El = compute_edge_connectivity(V, F)
    NV = compute_vertex_neighbourhood_matrix(V, F)
    NF = compute_face_neighbourhood_matrix(V, F, E)
    N, A = compute_normals_and_areas(V, F)
    D = compute_mesh_dual_points(V, F, E)
    VF = compute_vertex_face_participation_matrix(V, F)
    DEl, DE, DA = compute_dual_edges(V, F, D, VF, E)
    return NF, E, NV, N, A, VF, D, El, DEl, DE, DA

def compute_laplacian_crane(vertices, edges, edge_lengths, dual_edge_lengths):
    # Initialize the Laplacian
    L = np.zeros((vertices.shape[0], vertices.shape[0]))
    # Initialize the weights (i.e. the length fo the dual edge divided by the length of the primal edge)
    L[np.where(edges==1)] = jnp.multiply(dual_edge_lengths[dual_edge_lengths != 0], 1./edge_lengths[edge_lengths != 0])
    # Initialize the diagonal as the sum of all these weights
    L[np.diag_indices_from(L)] = -np.sum(L, axis=1)
    return L

def compute_laplacian_sorkine(edges):
    D = jnp.diag(jnp.sum(edges, axis=1))
    L = jnp.eye(edges.shape[0], dtype=jnp.float32) - jnp.dot(jnp.linalg.inv(D), edges)
    return L

def reconstruct(V, L, deltas, m=2, omega=1.0):
    n = L.shape[1] # Number of columns in L
    m = 2 # Num constraints
    Imxm = jnp.eye(m, dtype=jnp.float32)
    # Choose the m first vertices as being fixed
    c = V[:m]
    L_tilde = np.zeros((L.shape[0] + m, L.shape[1]))
    L_tilde[:L.shape[0], :L.shape[1]] = L
    L_tilde[L.shape[0]:, :m] = Imxm
    L_tilde = jnp.array(L_tilde)

    rhs = np.zeros((deltas.shape[0] + m, 3))
    rhs[:deltas.shape[0], :] = deltas
    rhs[deltas.shape[0]:, :] = omega * c
    rhs = jnp.array(rhs)

    Vreconstructed = jnp.dot(jnp.dot(jnp.linalg.inv(jnp.dot(L_tilde.T, L_tilde)), L_tilde.T), rhs)
    return Vreconstructed
