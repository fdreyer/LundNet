# This file is part of LundNet by F. Dreyer and H. Qu

import fastjet as fj
import numpy as np
import math

# ======================================================================


class LundCoordinates:
    """
    LundCoordinates takes two subjets associated with a declustering,
    and store the corresponding Lund coordinates.
    """

    # components of the LundCoordinates
    components = ['lnz', 'lnDelta', 'psi', 'lnm', 'lnKt']

    # number of dimensions for the state() method
    dimension = 5

    # ----------------------------------------------------------------------
    def __init__(self, j1, j2):
        """Define a number of variables associated with the declustering."""
        delta = np.float32(max(1e-6, j1.delta_R(j2)))
        z = np.float32(j2.pt() / (j1.pt() + j2.pt()))
        self.lnm = np.float32(0.5 * math.log(abs((j1 + j2).m2())))
        self.lnKt = np.float32(math.log(j2.pt() * delta))
        self.lnz = np.float32(math.log(z))
        self.lnDelta = np.float32(math.log(delta))
        self.lnKappa = np.float32(math.log(z * delta))
        try:
            self.psi = np.float32(math.atan((j1.rap() - j2.rap()) / (j1.phi() - j2.phi())))
        except ZeroDivisionError:
            self.psi = 0

    # ----------------------------------------------------------------------
    @staticmethod
    def change_dimension(n, order=['lnz', 'lnDelta', 'psi', 'lnm', 'lnKt']):
        LundCoordinates.components = order[:n]
        print(LundCoordinates.components)
        LundCoordinates.dimension = len(LundCoordinates.components)

    # ----------------------------------------------------------------------
    def state(self):
        # WARNING: For consistency with other parts of the code,
        #          lnz and lnDelta need to be the first two components
        return np.array([getattr(self, v) for v in LundCoordinates.components], dtype='float32')


# ======================================================================
class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    ktmin = 0.0
    deltamin = 0.0

    # ----------------------------------------------------------------------
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        self.delta2 = 0.0
        self.lundCoord = None
        # if it has a direct child (i.e. one level further up in the
        # tree), give a link to the corresponding tree object here
        self.child = child

        while True:
            j1 = fj.PseudoJet()
            j2 = fj.PseudoJet()
            if pseudojet and pseudojet.has_parents(j1, j2):
                # order the parents in pt
                if (j2.pt() > j1.pt()):
                    j1, j2 = j2, j1
                # check if we satisfy cuts
                delta = j1.delta_R(j2)
                kt = j2.pt() * delta
                if (delta < JetTree.deltamin):
                    break
                # then create two new tree nodes with j1 and j2
                if kt >= JetTree.ktmin:
                    self.harder = JetTree(j1, child=self)
                    self.softer = JetTree(j2, child=self)
                    self.delta2 = np.float32(delta * delta)
                    self.lundCoord = LundCoordinates(j1, j2)
                    break
                else:
                    pseudojet = j1
            else:
                break

        # finally define the current node
        self.node = np.array([pseudojet.px(), pseudojet.py(), pseudojet.pz(), pseudojet.E()], dtype='float32')

    # ----------------------------------------------------------------------
    @staticmethod
    def change_cuts(ktmin=0.0, deltamin=0.0):
        JetTree.ktmin = ktmin
        JetTree.deltamin = deltamin

    # -------------------------------------------------------------------------------
    def remove_soft(self):
        """Remove the softer branch of the JetTree node."""
        # start by removing softer parent momentum from the rest of the tree
        child = self.child
        while(child):
            child.node -= self.softer.node
            child = child.child
        del self.softer
        # then move the harder branch to the current node,
        # effectively deleting the soft branch
        newTree = self.harder
        self.node = newTree.node
        self.softer = newTree.softer
        self.harder = newTree.harder
        self.delta2 = newTree.delta2
        self.lundCoord = newTree.lundCoord
        # finally set the child pointer in the two parents to
        # the current node
        if self.harder:
            self.harder.child = self
        if self.softer:
            self.softer.child = self
        # NB: self.child doesn't change, we are just moving up the part
        #     of the tree below it

    # ----------------------------------------------------------------------
    def state(self):
        """Return state of lund coordinates."""
        if not self.lundCoord:
            return np.zeros(LundCoordinates.dimension)
        return self.lundCoord.state()

    # ----------------------------------------------------------------------
    def jet(self, pseudojet=False):
        """Return the kinematics of the JetTree."""
        # TODO: implement pseudojet option which returns a pseudojet
        #      with the reclustered constituents (after grooming)
        if not pseudojet:
            return self.node
        else:
            raise ValueError("JetTree: jet() with pseudojet return value not implemented.")

    # ----------------------------------------------------------------------
    def __lt__(self, other_tree):
        """Comparison operator needed for priority queue."""
        return self.delta2 > other_tree.delta2

    # ----------------------------------------------------------------------
    def __del__(self):
        """Delete the node."""
        if self.softer:
            del self.softer
        if self.harder:
            del self.harder
        del self.node
        del self

# ======================================================================


class LundImage:
    """Class to create Lund images from a jet tree."""

    # ----------------------------------------------------------------------
    def __init__(self, xval=[0.0, 7.0], yval=[-3.0, 7.0],
                 npxlx=50, npxly=None):
        """Set up the LundImage instance."""
        # set up the pixel numbers
        self.npxlx = npxlx
        if not npxly:
            self.npxly = npxlx
        else:
            self.npxly = npxly
        # set up the bin edge and width
        self.xmin = xval[0]
        self.ymin = yval[0]
        self.x_pxl_wdth = (xval[1] - xval[0]) / self.npxlx
        self.y_pxl_wdth = (yval[1] - yval[0]) / self.npxly

    # ----------------------------------------------------------------------
    def __call__(self, tree):
        """Process a jet tree and return an image of the primary Lund plane."""
        res = np.zeros((self.npxlx, self.npxly))

        self.fill(tree, res)
        return res

    # ----------------------------------------------------------------------
    def fill(self, tree, res):
        """Fill the res array recursively with the tree declusterings of the hard branch."""
        if(tree and tree.lundCoord):
            x = -tree.lundCoord.lnDelta
            y = tree.lundCoord.lnKt
            xind = math.ceil((x - self.xmin) / self.x_pxl_wdth - 1.0)
            yind = math.ceil((y - self.ymin) / self.y_pxl_wdth - 1.0)
            if (xind < self.npxlx and yind < self.npxly and min(xind, yind) >= 0):
                res[xind, yind] += 1
            self.fill(tree.harder, res)
            #self.fill(tree.softer, res)


# ======================================================================
class RSD:
    """RSD applies Recursive Soft Drop grooming to a JetTree."""

    # ----------------------------------------------------------------------
    def __init__(self, zcut=0.05, beta=1.0, R0=1.0):
        """Initialize RSD with its parameters."""
        self.lnzcut = math.log(zcut)
        self.beta = beta
        self.lnR0 = math.log(R0)

    # ----------------------------------------------------------------------
    def __call__(self, jet, returnTree=False):
        """Apply the groomer after casting the jet to a JetTree, and return groomed momenta."""
        # TODO: replace result by reclustered jet of all remaining constituents.
        if type(jet) == JetTree:
            tree = jet
        else:
            tree = JetTree(jet)
        self._groom(tree)
        return tree

    # ----------------------------------------------------------------------
    def _groom(self, tree):
        """Apply RSD grooming to a jet."""
        if not tree.lundCoord:
            # current node has no subjets => no grooming
            return
        state = tree.state()
        if not state.size > 0:
            # current node has no subjets => no grooming
            return
        # check the SD condition
        lnz, lndelta = state[:1]
        remove_soft = (lnz < self.lnzcut + self.beta * (lndelta - self.lnR0))
        if remove_soft:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            tree.remove_soft()
            # now we groom the new tree, since both nodes have been changed
            self._groom(tree)
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self._groom(tree.harder)
            if tree.softer:
                self._groom(tree.softer)
