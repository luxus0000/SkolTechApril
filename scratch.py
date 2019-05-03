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
("\n"
 "__all__ = ['fast_gnp_random_graph',\n"
 "           'gnp_random_graph',\n"
 "           'dense_gnm_random_graph',\n"
 "           'gnm_random_graph',\n"
 "           'erdos_renyi_graph',\n"
 "           'binomial_graph',\n"
 "           'newman_watts_strogatz_graph',\n"
 "           'watts_strogatz_graph',\n"
 "           'connected_watts_strogatz_graph',\n"
 "           'random_regular_graph',\n"
 "           'barabasi_albert_graph',\n"
 "           'dual_barabasi_albert_graph',\n"
 "           'extended_barabasi_albert_graph',\n"
 "           'powerlaw_cluster_graph',\n"
 "           'random_lobster',\n"
 "           'random_shell_graph',\n"
 "           'random_powerlaw_tree',\n"
 "           'random_powerlaw_tree_sequence',\n"
 "           'random_kernel_graph']\n"
 "def eig(a):\n"
 "    \"\"\"\n"
 "    Compute the eigenvalues and right eigenvectors of a square array.\n"
 "    Parameters\n"
 "    ----------\n"
 "    a : (..., M, M) array\n"
 "        Matrices for which the eigenvalues and right eigenvectors will\n"
 "        be computed\n"
 "    Returns\n"
 "    -------\n"
 "    w : (..., M) array\n"
 "        The eigenvalues, each repeated according to its multiplicity.\n"
 "        The eigenvalues are not necessarily ordered. The resulting\n"
 "        array will be of complex type, unless the imaginary part is\n"
 "        zero in which case it will be cast to a real type. When `a`\n"
 "        is real the resulting eigenvalues will be real (0 imaginary\n"
 "        part) or occur in conjugate pairs\n"
 "    v : (..., M, M) array\n"
 "        The normalized (unit \"length\") eigenvectors, such that the\n"
 "        column ``v[:,i]`` is the eigenvector corresponding to the\n"
 "        eigenvalue ``w[i]``.\n"
 "    Raises\n"
 "    ------\n"
 "    LinAlgError\n"
 "        If the eigenvalue computation does not converge.\n"
 "    See Also\n"
 "    --------\n"
 "    eigvals : eigenvalues of a non-symmetric array.\n"
 "    eigh : eigenvalues and eigenvectors of a real symmetric or complex\n"
 "           Hermitian (conjugate symmetric) array.\n"
 "    eigvalsh : eigenvalues of a real symmetric or complex Hermitian\n"
 "               (conjugate symmetric) array.\n"
 "    Notes\n"
 "    -----\n"
 "    .. versionadded:: 1.8.0\n"
 "    Broadcasting rules apply, see the `numpy.linalg` documentation for\n"
 "    details.\n"
 "    This is implemented using the _geev LAPACK routines which compute\n"
 "    the eigenvalues and eigenvectors of general square arrays.\n"
 "    The number `w` is an eigenvalue of `a` if there exists a vector\n"
 "    `v` such that ``dot(a,v) = w * v``. Thus, the arrays `a`, `w`, and\n"
 "    `v` satisfy the equations ``dot(a[:,:], v[:,i]) = w[i] * v[:,i]``\n"
 "    for :math:`i \\in \\{0,...,M-1\\}`.\n"
 "    The array `v` of eigenvectors may not be of maximum rank, that is, some\n"
 "    of the columns may be linearly dependent, although round-off error may\n"
 "    obscure that fact. If the eigenvalues are all different, then theoretically\n"
 "    the eigenvectors are linearly independent. Likewise, the (complex-valued)\n"
 "    matrix of eigenvectors `v` is unitary if the matrix `a` is normal, i.e.,\n"
 "    if ``dot(a, a.H) = dot(a.H, a)``, where `a.H` denotes the conjugate\n"
 "    transpose of `a`.\n"
 "    Finally, it is emphasized that `v` consists of the *right* (as in\n"
 "    right-hand side) eigenvectors of `a`.  A vector `y` satisfying\n"
 "    ``dot(y.T, a) = z * y.T`` for some number `z` is called a *left*\n"
 "    eigenvector of `a`, and, in general, the left and right eigenvectors\n"
 "    of a matrix are not necessarily the (perhaps conjugate) transposes\n"
 "    of each other.\n"
 "    References\n"
 "    ----------\n"
 "    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,\n"
 "    Academic Press, Inc., 1980, Various pp.\n"
 "    Examples\n"
 "    --------\n"
 "    >>> from numpy import linalg as LA\n"
 "    (Almost) trivial example with real e-values and e-vectors.\n"
 "    >>> w, v = LA.eig(np.diag((1, 2, 3)))\n"
 "    >>> w; v\n"
 "    array([ 1.,  2.,  3.])\n"
 "    array([[ 1.,  0.,  0.],\n"
 "           [ 0.,  1.,  0.],\n"
 "           [ 0.,  0.,  1.]])\n"
 "    Real matrix possessing complex e-values and e-vectors; note that the\n"
 "    e-values are complex conjugates of each other.\n"
 "    >>> w, v = LA.eig(np.array([[1, -1], [1, 1]]))\n"
 "    >>> w; v\n"
 "    array([ 1. + 1.j,  1. - 1.j])\n"
 "    array([[ 0.70710678+0.j        ,  0.70710678+0.j        ],\n"
 "           [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]])\n"
 "    Complex-valued matrix with real e-values (but complex-valued e-vectors);\n"
 "    note that a.conj().T = a, i.e., a is Hermitian.\n"
 "    >>> a = np.array([[1, 1j], [-1j, 1]])\n"
 "    >>> w, v = LA.eig(a)\n"
 "    >>> w; v\n"
 "    array([  2.00000000e+00+0.j,   5.98651912e-36+0.j]) # i.e., {2, 0}\n"
 "    array([[ 0.00000000+0.70710678j,  0.70710678+0.j        ],\n"
 "           [ 0.70710678+0.j        ,  0.00000000+0.70710678j]])\n"
 "    Be careful about round-off error!\n"
 "    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])\n"
 "    >>> # Theor. e-values are 1 +/- 1e-9\n"
 "    >>> w, v = LA.eig(a)\n"
 "    >>> w; v\n"
 "    array([ 1.,  1.])\n"
 "    array([[ 1.,  0.],\n"
 "           [ 0.,  1.]])\n"
 "    \"\"\"\n"
 "    a, wrap = _makearray(a)\n"
 "    _assertRankAtLeast2(a)\n"
 "    _assertNdSquareness(a)\n"
 "    _assertFinite(a)\n"
 "    t, result_t = _commonType(a)\n"
 "\n"
 "    extobj = get_linalg_error_extobj(_raise_linalgerror_eigenvalues_nonconvergence)\n"
 "    signature = 'D->DD' if isComplexType(t) else 'd->DD'\n"
 "    w, vt = _umath_linalg.eig(a, signature=signature, extobj=extobj)\n"
 "\n"
 "    if not isComplexType(t) and all(w.imag == 0.0):\n"
 "        w = w.real\n"
 "        vt = vt.real\n"
 "        result_t = _realType(result_t)\n"
 "    else:\n"
 "        result_t = _complexType(result_t)\n"
 "\n"
 "    vt = vt.astype(result_t, copy=False)\n"
 "    return w.astype(result_t, copy=False), wrap(vt)\n"
 "if __name__ == '__main__':\n"
 "    G = PrintGraph()\n"
 "    G.add_node('foo')\n"
 "    G.add_nodes_from('bar', weight=8)\n"
 "    G.remove_node('b')\n"
 "    G.remove_nodes_from('ar')\n"
 "    print(\"Nodes in G: \", G.nodes(data=True))\n"
 "    G.add_edge(0, 1, weight=10)\n"
 "    print(\"Edges in G: \", G.edges(data=True))\n"
 "    G.remove_edge(0, 1)\n"
 "    G.add_edges_from(zip(range(0, 3), range(1, 4)), weight=10)\n"
 "    print(\"Edges in G: \", G.edges(data=True))\n"
 "    G.remove_edges_from(zip(range(0, 3), range(1, 4)))\n"
 "    print(\"Edges in G: \", G.edges(data=True))\n"
 "\n"
 "    G = PrintGraph()\n"
 "    nx.add_path(G, range(10))\n"
 "    nx.add_star(G, range(9, 13))\n"
 "    nx.draw(G)\n"
 "    plt.show()\n")
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
seed=numpy.random
'''G_e_r  = erdos_renyi_graph(20,0.1, seed, False)
nx.draw(G_e_r)
plt.show()
L1 = laplacian_matrix(G_e_r)
L1NM = L1.todense()'''
watts_strog_s = 100 #doesn't work for watts_strog_s >199; watts_strog_s - number of nodes
#nx.draw(G_watts_strog)
#plt.show()
#L2NM = laplacian_matrix(watts_strogatz_graph(watts_strog_s, 6, 0, seed)).todense()
N = 200 #N - number of betas
betas, Entropy, A = [0]*N, [0]*N, [0]*watts_strog_s
#E_M = [[0 for x in range(N)] for y in range(6)]
log_watts_strog_s = numpy.log(watts_strog_s)/numpy.log(2) #This is needed for normalization
N1 = N/15
N2 = N*0.5
ns = [0, .25, .5, .75, 1]
colors = ['blue', 'red', 'green', 'yellow', 'purple']
for i in range(len(ns)):
    na = ns[i]
    L2NM = laplacian_matrix(watts_strogatz_graph(watts_strog_s, 6, na, seed)).todense()
    for beta in numpy.arange(N+1):
        '''L1NM_exp = scipy.linalg.expm(-beta * L1NM)
        Z = numpy.trace(L1NM_exp)
        DM_L1NM_exp = L1NM_exp / Z
        Lambda_DM_L1NM_exp, Vectors_DM_L1NM_exp = numpy.linalg.eig(DM_L1NM_exp)'''
        beta_exp = numpy.exp((N2 - beta)/N1)
        L2NM_exp = scipy.linalg.expm(-beta_exp*L2NM/10)
        Z = numpy.trace(L2NM_exp)
        DM_L2NM_exp = L2NM_exp / Z
        Lambda_DM_L2NM_exp, Vectors_DM_L2NM_exp = numpy.linalg.eig(DM_L2NM_exp)
        Sum_Lambda = 0
        for j in range(len(A)):
            #real_lambda = numpy.real(Lambda_DM_L1NM_exp[i])
            real_lambda = numpy.real(Lambda_DM_L2NM_exp[j])
            if real_lambda < 0 or real_lambda ==0:
                A[j] = 0
            else:
                A[j] = real_lambda * numpy.log(real_lambda) * (1/numpy.log(2))
            Sum_Lambda-=A[j]
            Sum_Lambda = numpy.real(Sum_Lambda)
        betas[beta-1], Entropy[beta-1] = beta_exp, Sum_Lambda/log_watts_strog_s
        #betas[beta - 1], E_M[i][beta - 1] = beta_exp, Sum_Lambda / log_watts_strog_s
    '''fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xscale('log')
    pyplot.subplot(1,1,1)'''
    #E_M[i] = Entropy
    betas_inverse = [0] * 200
    for n in range(len(betas)):
        if betas[n] > 0:
            betas_inverse[n] = 1 / betas[n]
    pyplot.plot(betas_inverse, Entropy, colors[i], lw=1)
    #pyplot.plot(betas_inverse, Entropy, color='blue', lw=1)
pyplot.xscale('log')
plt.show()
#pyplot.plot(betas_inverse, E_M[0], color='blue', lw=1)
#pyplot.plot(betas_inverse, E_M[1], color='red', lw=1)
#pyplot.plot(betas_inverse, E_M[2], color='blue', lw=1)
#pyplot.plot(betas_inverse, E_M[3], color='blue', lw=1)
#pyplot.plot(betas_inverse, E_M[4], color='blue', lw=1)
'''
plt.plot(betas, Entropy)
plt.show()'''