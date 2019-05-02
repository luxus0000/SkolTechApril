from __future__ import division
from networkx.utils import py_random_state
from networkx import path_graph, random_layout
import itertools
import random
import itertools
import math
import numpy
import scipy
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
#import matrix-decomposition as md
from networkx import Graph
from numpy.core import (array, asarray)
from matplotlib import pyplot
from scipy.sparse import csr_matrix
from numpy import linalg as LA
"""from .classic import empty_graph, path_graph, complete_graph
from .degree_seq import degree_sequence_tree
from collections import defaultdict"""
"""
__all__ = ['fast_gnp_random_graph',
           'gnp_random_graph',
           'dense_gnm_random_graph',
           'gnm_random_graph',
           'erdos_renyi_graph',
           'binomial_graph',
           'newman_watts_strogatz_graph',
           'watts_strogatz_graph',
           'connected_watts_strogatz_graph',
           'random_regular_graph',
           'barabasi_albert_graph',
           'dual_barabasi_albert_graph',
           'extended_barabasi_albert_graph',
           'powerlaw_cluster_graph',
           'random_lobster',
           'random_shell_graph',
           'random_powerlaw_tree',
           'random_powerlaw_tree_sequence',
           'random_kernel_graph']
"""
class PrintGraph(Graph):
    """
    Example subclass of the Graph class.

    Prints activity log to file or standard output.
    """

    def __init__(self, data=None, name='', file=None, **attr):
        Graph.__init__(self, data=data, name=name, **attr)
        if file is None:
            import sys
            self.fh = sys.stdout
        else:
            self.fh = open(file, 'w')

    def add_node(self, n, attr_dict=None, **attr):
        Graph.add_node(self, n, attr_dict=attr_dict, **attr)
        self.fh.write("Add node: %s\n" % n)

    def add_nodes_from(self, nodes, **attr):
        for n in nodes:
            self.add_node(n, **attr)

    def remove_node(self, n):
        Graph.remove_node(self, n)
        self.fh.write("Remove node: %s\n" % n)

    def remove_nodes_from(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def add_edge(self, u, v, attr_dict=None, **attr):
        Graph.add_edge(self, u, v, attr_dict=attr_dict, **attr)
        self.fh.write("Add edge: %s-%s\n" % (u, v))

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        for e in ebunch:
            u, v = e[0:2]
            self.add_edge(u, v, attr_dict=attr_dict, **attr)

    def remove_edge(self, u, v):
        Graph.remove_edge(self, u, v)
        self.fh.write("Remove edge: %s-%s\n" % (u, v))

    def remove_edges_from(self, ebunch):
        for e in ebunch:
            u, v = e[0:2]
            self.remove_edge(u, v)

    def clear(self):
        Graph.clear(self)
        self.fh.write("Clear graph\n")
def empty_graph(n=0, create_using=None):
    """Return the empty graph with n nodes and zero edges.

    Node labels are the integers 0 to n-1

    For example:
    >>> G=nx.empty_graph(10)
    >>> G.number_of_nodes()
    10
    >>> G.number_of_edges()
    0

    The variable create_using should point to a "graph"-like object that
    will be cleaned (nodes and edges will be removed) and refitted as
    an empty "graph" with n nodes with integer labels. This capability
    is useful for specifying the class-nature of the resulting empty
    "graph" (i.e. Graph, DiGraph, MyWeirdGraphClass, etc.).

    The variable create_using has two main uses:
    Firstly, the variable create_using can be used to create an
    empty digraph, network,etc.  For example,

    >>> n=10
    >>> G=nx.empty_graph(n,create_using=nx.DiGraph())

    will create an empty digraph on n nodes.

    Secondly, one can pass an existing graph (digraph, pseudograph,
    etc.) via create_using. For example, if G is an existing graph
    (resp. digraph, pseudograph, etc.), then empty_graph(n,create_using=G)
    will empty G (i.e. delete all nodes and edges using G.clear() in
    base) and then add n nodes and zero edges, and return the modified
    graph (resp. digraph, pseudograph, etc.).

    See also create_empty_copy(G).

    """
    if create_using is None:
        # default empty graph is a simple graph
        G = nx.Graph()
    else:
        G = create_using
        G.clear()

    G.add_nodes_from(range(n))
    G.name = "empty_graph(%d)" % n
    return G
def complete_graph(n,create_using=None):
    """ Return the complete graph K_n with n nodes.

    Node labels are the integers 0 to n-1.
    """
    G=empty_graph(n,create_using)
    G.name="complete_graph(%d)"%(n)
    if n>1:
        if G.is_directed():
            edges=itertools.permutations(range(n),2)
        else:
            edges=itertools.combinations(range(n),2)
        G.add_edges_from(edges)
    return G
def laplacian_matrix(G, nodelist=None, weight='weight'):
    """Return the Laplacian matrix of G.

    The graph Laplacian is the matrix L = D - A, where
    A is the adjacency matrix and D is the diagonal matrix of node degrees.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    L : SciPy sparse matrix
      The Laplacian matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.

    See Also
    --------
    to_numpy_matrix
    normalized_laplacian_matrix
    """
    import scipy.sparse
    if nodelist is None:
        nodelist = G.nodes()
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n,m = A.shape
    diags = A.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
    return  D - A
    # -*- coding: utf-8 -*-
    #    Copyright (C) 2004-2019 by
    #    Aric Hagberg <hagberg@lanl.gov>
    #    Dan Schult <dschult@colgate.edu>
    #    Pieter Swart <swart@lanl.gov>
    #    All rights reserved.
    #    BSD license.
def fast_gnp_random_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
    a binomial graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    Notes
    -----
    The $G_{n,p}$ graph algorithm chooses each of the $[n (n - 1)] / 2$
    (undirected) or $n (n - 1)$ (directed) possible edges with probability $p$.

    This algorithm [1]_ runs in $O(n + m)$ time, where `m` is the expected number of
    edges, which equals $p n (n - 1) / 2$. This should be faster than
    :func:`gnp_random_graph` when $p$ is small and the expected number of edges
    is small (that is, the graph is sparse).

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    G = empty_graph(n)

    if p <= 0 or p >= 1:
        return nx.gnp_random_graph(n, p, seed=seed, directed=directed)

    w = -1
    lp = math.log(1.0 - p)

    if directed:
        G = nx.DiGraph(G)
        # Nodes in graph are from 0,n-1 (start with v as the first node index).
        v = 0
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            if v == w:  # avoid self loops
                w = w + 1
            while v < n <= w:
                w = w - n
                v = v + 1
                if v == w:  # avoid self loops
                    w = w + 1
            if v < n:
                G.add_edge(v, w)
    else:
        # Nodes in graph are from 0,n-1 (start with v as the second node index).
        v = 1
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            while w >= v and v < n:
                w = w - v
                v = v + 1
            if v < n:
                G.add_edge(v, w)
    return G
def erdos_renyi_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
    or a binomial graph.

    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

    The functions :func:`binomial_graph` and :func:`erdos_renyi_graph` are
    aliases of this function.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    See Also
    --------
    fast_gnp_random_graph

    Notes
    -----
    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        edges = itertools.permutations(range(n), 2)
        G = nx.DiGraph()
    else:
        edges = itertools.combinations(range(n), 2)
        G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return complete_graph(n, create_using=G)

    for e in edges:
        if seed.random() < p:
            G.add_edge(*e)
    return G
def watts_strogatz_graph(n, k, p, seed=None):
    """Returns a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    newman_watts_strogatz_graph()
    connected_watts_strogatz_graph()

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.

    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    if k >= n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.Graph()
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G
def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap
def _assertRankAtLeast2(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)
class LinAlgError(Exception):
    """
    Generic Python-exception-derived object raised by linalg functions.
    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.
    Parameters
    ----------
    None
    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...linalg.py", line 350,
        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
      File "...linalg.py", line 249,
        in solve
        raise LinAlgError('Singular matrix')
    numpy.linalg.LinAlgError: Singular matrix
    """
def eig(a):
    """
    Compute the eigenvalues and right eigenvectors of a square array.
    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed
    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be of complex type, unless the imaginary part is
        zero in which case it will be cast to a real type. When `a`
        is real the resulting eigenvalues will be real (0 imaginary
        part) or occur in conjugate pairs
    v : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.
    See Also
    --------
    eigvals : eigenvalues of a non-symmetric array.
    eigh : eigenvalues and eigenvectors of a real symmetric or complex
           Hermitian (conjugate symmetric) array.
    eigvalsh : eigenvalues of a real symmetric or complex Hermitian
               (conjugate symmetric) array.
    Notes
    -----
    .. versionadded:: 1.8.0
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    This is implemented using the _geev LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.
    The number `w` is an eigenvalue of `a` if there exists a vector
    `v` such that ``dot(a,v) = w * v``. Thus, the arrays `a`, `w`, and
    `v` satisfy the equations ``dot(a[:,:], v[:,i]) = w[i] * v[:,i]``
    for :math:`i \\in \\{0,...,M-1\\}`.
    The array `v` of eigenvectors may not be of maximum rank, that is, some
    of the columns may be linearly dependent, although round-off error may
    obscure that fact. If the eigenvalues are all different, then theoretically
    the eigenvectors are linearly independent. Likewise, the (complex-valued)
    matrix of eigenvectors `v` is unitary if the matrix `a` is normal, i.e.,
    if ``dot(a, a.H) = dot(a.H, a)``, where `a.H` denotes the conjugate
    transpose of `a`.
    Finally, it is emphasized that `v` consists of the *right* (as in
    right-hand side) eigenvectors of `a`.  A vector `y` satisfying
    ``dot(y.T, a) = z * y.T`` for some number `z` is called a *left*
    eigenvector of `a`, and, in general, the left and right eigenvectors
    of a matrix are not necessarily the (perhaps conjugate) transposes
    of each other.
    References
    ----------
    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,
    Academic Press, Inc., 1980, Various pp.
    Examples
    --------
    >>> from numpy import linalg as LA
    (Almost) trivial example with real e-values and e-vectors.
    >>> w, v = LA.eig(np.diag((1, 2, 3)))
    >>> w; v
    array([ 1.,  2.,  3.])
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.
    >>> w, v = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> w; v
    array([ 1. + 1.j,  1. - 1.j])
    array([[ 0.70710678+0.j        ,  0.70710678+0.j        ],
           [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]])
    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that a.conj().T = a, i.e., a is Hermitian.
    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = LA.eig(a)
    >>> w; v
    array([  2.00000000e+00+0.j,   5.98651912e-36+0.j]) # i.e., {2, 0}
    array([[ 0.00000000+0.70710678j,  0.70710678+0.j        ],
           [ 0.70710678+0.j        ,  0.00000000+0.70710678j]])
    Be careful about round-off error!
    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. e-values are 1 +/- 1e-9
    >>> w, v = LA.eig(a)
    >>> w; v
    array([ 1.,  1.])
    array([[ 1.,  0.],
           [ 0.,  1.]])
    """
    a, wrap = _makearray(a)
    _assertRankAtLeast2(a)
    _assertNdSquareness(a)
    _assertFinite(a)
    t, result_t = _commonType(a)

    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    w, vt = _umath_linalg.eig(a, signature=signature, extobj=extobj)

    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:
        result_t = _complexType(result_t)

    vt = vt.astype(result_t, copy=False)
    return w.astype(result_t, copy=False), wrap(vt)
"""
if __name__ == '__main__':
    G = PrintGraph()
    G.add_node('foo')
    G.add_nodes_from('bar', weight=8)
    G.remove_node('b')
    G.remove_nodes_from('ar')
    print("Nodes in G: ", G.nodes(data=True))
    G.add_edge(0, 1, weight=10)
    print("Edges in G: ", G.edges(data=True))
    G.remove_edge(0, 1)
    G.add_edges_from(zip(range(0, 3), range(1, 4)), weight=10)
    print("Edges in G: ", G.edges(data=True))
    G.remove_edges_from(zip(range(0, 3), range(1, 4)))
    print("Edges in G: ", G.edges(data=True))

    G = PrintGraph()
    nx.add_path(G, range(10))
    nx.add_star(G, range(9, 13))
    nx.draw(G)
    plt.show()
"""
seed=numpy.random
'''G_e_r  = erdos_renyi_graph(20,0.1, seed, False)
nx.draw(G_e_r)
plt.show()
L1 = laplacian_matrix(G_e_r)
L1NM = L1.todense()'''
watts_strog_s = 30
G_watts_strog = watts_strogatz_graph(watts_strog_s, 5, 0.02, seed)
nx.draw(G_watts_strog)
plt.show()
L2 = laplacian_matrix(G_watts_strog)
L2NM = L2.todense()
N = 200
betas, Entropy = [0]*N,[0]*N
A = [0]*watts_strog_s
for beta in numpy.arange(N+1):
    '''L1NM_exp = scipy.linalg.expm(-beta * L1NM)
    Z = numpy.trace(L1NM_exp)
    DM_L1NM_exp = L1NM_exp / Z
    Lambda_DM_L1NM_exp, Vectors_DM_L1NM_exp = numpy.linalg.eig(DM_L1NM_exp)'''
    beta_exp = numpy.exp((N*0.5 - beta)/20)
    L2NM_exp = scipy.linalg.expm(-beta_exp*L2NM/10)
    Z = numpy.trace(L2NM_exp)
    DM_L2NM_exp = L2NM_exp / Z
    Lambda_DM_L2NM_exp, Vectors_DM_L2NM_exp = numpy.linalg.eig(DM_L2NM_exp)
    Sum_Lambda = 0
    for i in range(len(A)):
        #real_lambda = numpy.real(Lambda_DM_L1NM_exp[i])
        real_lambda = numpy.real(Lambda_DM_L2NM_exp[i])
        if real_lambda < 0 or real_lambda ==0:
            A[i] = 0
        else:
            A[i] = real_lambda * numpy.log(real_lambda) * (1/numpy.log(2))
        Sum_Lambda-=A[i]
        Sum_Lambda = numpy.real(Sum_Lambda)
    betas[beta-1], Entropy[beta-1] = beta_exp, Sum_Lambda
'''fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.set_xscale('log')
pyplot.subplot(1,1,1)'''
betas_inverse = [0]*200
for i in range(len(betas)):
    if betas[i] >0:
        betas_inverse[i] = 1/betas[i]
pyplot.plot(betas_inverse, Entropy, color='blue', lw=1)
pyplot.xscale('log')
pyplot.show()
'''
plt.plot(betas, Entropy)
plt.show()'''