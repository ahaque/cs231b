# distutils: language = c++
# distutils: sources = maxflow.cpp graph.cpp

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "graph.h":
    cdef cppclass Block[T]:
        pass
    cdef enum termtype:
        SOURCE, SINK
    cdef cppclass Graph[capT, tcapT, flowT]:
        Graph(int node_num_max, int edge_num_max) except +
        int add_node(int)
        void add_edge(int i, int j, capT cap, capT rev_cap)
        void add_tweights(int i, tcapT cap_source, tcapT cap_sink)
        flowT maxflow()
        termtype what_segment(int i)
        int get_node_num()

cdef class PyGraph:
    cdef Graph[float,float,float] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int node_num_max, int edge_num_max):
        self.thisptr = new Graph[float,float,float](node_num_max, edge_num_max)
    def __dealloc__(self):
        del self.thisptr
    def add_node(self, int num=1):
        return self.thisptr.add_node(num)
    def add_edge(self, int i, int j, float cap, float rev_cap):
        self.thisptr.add_edge(i, j, cap, rev_cap)
    def add_tweights(self, int i, float cap_source, float cap_sink):
        self.thisptr.add_tweights(i, cap_source, cap_sink)
    def maxflow(self):
        return self.thisptr.maxflow()
    def what_segment(self, int i):
        return self.thisptr.what_segment(i)

    @cython.boundscheck(False)
    def add_edge_vectorized(self,
                            np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] i,
                            np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] j,
                            np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False] cap,
                            np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False] rev_cap):
        assert i.size == j.size
        assert i.size == cap.size
        assert i.size == rev_cap.size
        cdef int l
        for l in range(i.size):
            self.thisptr.add_edge(i[l], j[l], cap[l], rev_cap[l])

    @cython.boundscheck(False)
    def add_tweights_vectorized(self,
                            np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] i,
                            np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False] cap_source,
                            np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False] cap_sink):
        assert i.size == cap_source.size
        assert i.size == cap_sink.size
        cdef int l
        for l in range(i.size):
            self.thisptr.add_tweights(i[l], cap_source[l], cap_sink[l])

    @cython.boundscheck(False)
    def what_segment_vectorized(self):
        cpdef np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] out_segments = np.empty(self.thisptr.get_node_num(), np.int32)
        cdef int l
        for l in range(self.thisptr.get_node_num()):
            out_segments[l] = self.thisptr.what_segment(l)
        return out_segments
