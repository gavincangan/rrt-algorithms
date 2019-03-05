from rtree import index


class Tree(object):
    def __init__(self, X, dimensions=None ):
        """
        Tree representation
        :param X: Search Space
        """
        p = index.Property()
        p.dimension = dimensions if dimensions else X.dimensions
        self.V = index.Index(interleaved=True, properties=p)  # vertices in an rtree
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent
