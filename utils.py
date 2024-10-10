import numpy as np


def nsga_ref_points(x0, y0, z0, n_u=100, n_v=100, u_scale=1.0, v_scale=1.0):

    # normal vector to the plane containing ref points
    a, b, c = 1 / x0, 1 / y0, 1 / z0
    n = np.array([a, b, c])

    # normalized plane basis vectors
    u = np.array([1.0, 0.0, -a / c])
    v = np.cross(u, n)
    u = u / np.sqrt(np.dot(u, u))
    v = v / np.sqrt(np.dot(v, v))

    # get a point
    o = np.array([x0, y0, z0]) / 3.0

    # build the point grid
    point_grid = []
    for i_u in np.arange(-n_u,n_u):
        for i_v in np.arange(-n_v,n_v):

            # sweep points
            p = o + i_u*u*u_scale + i_v*v*v_scale

            # append poin if in first quadrant
            is_first_quadrant = (p[0]>=0) and (p[1]>=0) and (p[2]>=0)
            if is_first_quadrant:
                point_grid.append(p)

    return point_grid
