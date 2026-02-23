from typing import TypeAlias, List, Tuple
import numpy as np

LatticeState : TypeAlias = List[int]
"""
Represents a single-particle state on the 3D lattice.
Format: [i, j, k, tz, sz] where i,j,k are spatial and tz,sz are isospin/spin.
"""

SingleParticleBasis: TypeAlias = List[LatticeState]
"""
A list containing all single-particle states in the basis.
"""

LatticeSite: TypeAlias = List[Tuple[int, int, int]]
"""
Represents a spatial coordinate on the 3D lattice.
Format: (i, j, k)
"""

LatticeSites: TypeAlias = List[LatticeSite]

OneBodyElement: TypeAlias = Tuple[int, int, float]
"""
A single-particle matrix element in sparse format.
Format: (p, q, value) representing <p|O|q>.
"""

TwoBodyElement: TypeAlias = Tuple[int, int, int, int, float]
"""
A two-body matrix element in sparse format.
Format: (p, q, r, s, value) representing <pq|V|rs>.
Typically stored with p < q and r < s.
"""

ThreeBodyElement: TypeAlias = Tuple[int, int, int, int, int, int, float]
"""
A three-body matrix element in sparse format.
Format: (p, q, r, s, t, u, value) representing <pqr|W|stu>.
Typically stored with p < q < r and s < t < u.
"""

class Operator:
    """
    Base class for many-body operators using Structure of Arrays (SoA) storage.
    Decouples indices (integers) from values (floats) for type safety and performance.
    """
    def __init__(self, indices, values, nstat: int):
        """
        Args:
            indices (np.ndarray): Integer array of shape (N_elements, Rank).
            values (np.ndarray): Float array of shape (N_elements,).
            nstat (int): Dimension of the single-particle basis.
        """
        # Enforce contiguous memory layout for C-speed access
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        if np.iscomplexobj(values):
            self.values = np.ascontiguousarray(values, dtype=np.complex128)
        else:
            self.values = np.ascontiguousarray(values, dtype=np.float64)
        self.nstat = nstat
        
        # Validation
        if self.values.ndim != 1:
            raise ValueError(f"Values must be 1D array, got shape {self.values.shape}")
        
        if self.indices.ndim == 1:
            self.indices = self.indices.reshape(-1, 1)

        if len(self.indices) != len(self.values):
            raise ValueError(f"Length mismatch: {len(self.indices)} indices vs {len(self.values)} values.")
        
    def __len__(self):
        return len(self.values)

    @classmethod
    def from_list(cls, operator, nstat: int):
        """
        Operator from a legacy list of lists.
        
        Args:
            data_list: List of tuples/lists, e.g. [[p, q, val], ...]
            nstat: Dimension of the basis.
        """
        if not operator:
            rank = cls._get_expected_rank()
            return cls(np.empty((0, rank), dtype=np.int64), np.array([], dtype=np.float64), nstat)

        arr = np.array(operator, dtype=np.float64)
        
        values = arr[:, -1]
        indices = np.round(arr[:, :-1]).astype(np.int64)
        
        return cls(indices, values, nstat)

    def to_list(self):
        """
        Converts to a list of lists.
        For backward compatibility or serialization.
        """
        if len(self) == 0:
            return []
            
        out_list = []
        for i in range(len(self.values)):
            row = self.indices[i].tolist()
            row.append(self.values[i])
            out_list.append(row)
        return out_list

    @classmethod
    def _get_expected_rank(cls):
        return 0


class OneBodyOperator(Operator):
    """
    Represents a 1-body operator h_pq.
    Indices shape: (N, 2) -> [p, q]
    """
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 2:
            raise ValueError(f"OneBodyOperator indices must have shape (N, 2), got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 2

    def to_dense(self):
        """Returns the standard N x N matrix representation."""
        mat = np.zeros((self.nstat, self.nstat))
        if len(self) > 0:
            p, q = self.indices[:, 0], self.indices[:, 1]
            mat[p, q] = self.values
        return mat


class TwoBodyOperator(Operator):
    """
    Represents a 2-body operator V_pqrs.
    Indices shape: (N, 4) -> [p, q, r, s]
    """
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"TwoBodyOperator indices must have shape (N, 4), got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 4


class ThreeBodyOperator(Operator):
    """
    Represents a 3-body operator W_pqrstu.
    Indices shape: (N, 6) -> [p, q, r, s, t, u]
    """
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"ThreeBodyOperator indices must have shape (N, 6), got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 6
