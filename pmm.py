# pmm: parameterized mesh mapping

import numpy as onp
import jax.numpy as jnp
from jax import random

def scale(fun, V, F, values, sigma=2):
    vfun = jnp.vectorize(lambda v: fun(V, F, V*v, F, sigma))
    return vfun(values)

def scale_x(fun, V, F, values, sigma=2):
    axis_array = onp.zeros(V.shape)
    axis_array[:,0] = 1
    axis_array = jnp.array(axis_array)
    vfun = jnp.vectorize(lambda v: fun(V, F, V*axis_array*v, F, sigma))
    return vfun(values)

def translate_x(fun, V, F, values, sigma=2):
    vfun = jnp.vectorize(lambda v: fun(V, F, V + v, F, sigma))
    return vfun(values)

def add_noise(fun, V, F, values, sigma=2):
    key = random.PRNGKey(42)

    def add_noise(v):
        N = random.normal(key, V.shape)
        return fun(V, F, V + N*v, F, sigma)

    vfun = jnp.vectorize(add_noise)

    return vfun(values)

def rotate_x(fun, V, F, values, sigma=2):

    def rotate(v):
        ct = jnp.cos(v)
        st = jnp.sin(v)

        R = jnp.array([[1,  0,   0],
                       [0, ct, -st],
                       [0, st,  ct]])

        return fun(V, F, jnp.dot(R, V.T).T, F, sigma)

    vfun = jnp.vectorize(rotate)

    return vfun(values)

def rotate_y(fun, V, F, values, sigma=2):

    def rotate(v):
        ct = jnp.cos(v)
        st = jnp.sin(v)

        R = jnp.array([[ ct, 0, st],
                       [  0, 1,  0],
                       [-st, 0, ct]])

        return fun(V, F, jnp.dot(R, V.T).T, F, sigma)

    vfun = jnp.vectorize(rotate)

    return vfun(values)

def rotate_z(fun, V, F, values, sigma=2):

    def rotate(v):
        ct = jnp.cos(v)
        st = jnp.sin(v)

        R = jnp.array([[ct, -st, 0],
                       [st,  ct, 0],
                       [ 0,   0, 1]])

        return fun(V, F, jnp.dot(R, V.T).T, F, sigma)

    vfun = jnp.vectorize(rotate)

    return vfun(values)
