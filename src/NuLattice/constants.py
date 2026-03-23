"""
Provides useful constants to be used in the rest of the code
"""

HBARC = 197.3269804  # MeV fm
MASS_P = 938.27208943 # MeV
MASS_N = 939.56542194 # MeV
MASS  = 0.5*(MASS_P+MASS_N)



# NOTE(vivek): double check reference generation with Matthias/Thomas
def generate_reference(coords):
    """Generates 4 states (all spin/isospin combos) for each (x, y, z) coordinate."""
    reference = []
    for x, y, z in coords:
        for tz in [0, 1]:
            for sz in [0, 1]:
                reference.append([x, y, z, sz, tz]) # currently sz, tz but should be tz, sz
    return reference

class ReferenceState:
    H2_GS  = [[0,0,0,0,0], [0,0,0,1,0]]
    H3_GS  = [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,1,1]]
    HE3_GS = [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1]]
    HE4_GS = generate_reference([(0, 0, 0)])

    LI6_GS     = [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,1,1], [1,0,0,0,0], [1,0,0,1,0]]
    LI6_3HE3H  = [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,1,1], [1,0,0,0,0], [1,0,0,1,0], [1,0,0,0,1]]

    BE8_GS  = generate_reference([(0, 0, 0), (1, 0, 0)])
    C12_GS  = generate_reference([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    
    C12_HOYLE  = generate_reference([(0, 0, 0), (1, 0, 0), (2, 1, 0)])
    C12_LINEAR = generate_reference([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    C12_LOOSE  = generate_reference([(0, 0, 0), (2, 1, 2), (1, 2, 1)])

    O16_GS = generate_reference([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    O16_EX = generate_reference([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])

    @staticmethod
    def holes(ref, basis):
        """
        given a reference state, and a lattice basis, this function returns the corresponding holes
        as a tuple

        :param ref:   reference state as list of states [lx, ly, lz, tz, sz] where the first three integers
                      lx, ly, lzx are the lattice site, and the last two integers are the
                      isospin and spin (with values 0, 1 for -1/2, 1/2)
        :type ref:    list[list[int,int,int,int,int]]
        :param basis: list of basis states in the lattice
        :type basis:  list[list[int,int,int,int,int]]
        :return:      tuple of A integers that are the indices of the hole states
        :rtype:       tuple(int,int,...)
        """
        holes = []
        for state in ref:
            i = basis.index(state)
            holes.append(i)
        return tuple(holes)
