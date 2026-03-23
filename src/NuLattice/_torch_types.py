from typing import TypeAlias, List, Tuple
import torch

LatticeState: TypeAlias = List[int]
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
    Supports both NumPy and PyTorch backends.
    """

    __slots__ = ('indices', 'values', 'nstat', 'device')
    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        nstat: int,
    ):
        """
        Args:
            indices: Integer array of shape (N_elements, Rank).
            values: Float array of shape (N_elements,).
            nstat (int): Dimension of the single-particle basis.
        """
        self.nstat = nstat

        self.indices = indices.to(dtype=torch.int64).contiguous()
        self.values = values.contiguous()
        self.device = self.values.device

        if self.indices.ndim == 1:
            self.indices = self.indices.unsqueeze(1)

        if self.values.ndim != 1:
            raise ValueError(f"Values must be 1D, got shape {self.values.shape}")

        if len(self.indices) != len(self.values):
            raise ValueError(
                f"Length mismatch: {len(self.indices)} indices vs {len(self.values)} values."
            )

    def __len__(self):
        return len(self.values)

    def to_torch(self, device=None):
        """Converts operator to PyTorch backend."""
        if device is None:
            return self
        return self.__class__(
                self.indices.to(device, dtype=torch.int32),
                self.values.to(device),
                self.nstat
            )

    def to_numpy(self):
        """Converts operator to NumPy backend."""
        return self.__class__(
            self.indices.cpu().numpy(), self.values.cpu().numpy(), self.nstat
        )

    def to_list(self):
        """
        Converts to a list of lists.
        For backward compatability or serialization
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
    def from_list(
        cls, operator_list, nstat: int, use_torch: bool = False, device: str = None
    ):
        """Operator from a legacy list of lists [[p, q, ..., val], ...]"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not operator_list:
            rank = cls._get_expected_rank()
            return cls(
                torch.empty((0, rank), dtype=torch.long, device=device),
                torch.empty((0,), device=device),
                nstat,
            )

        data = torch.tensor(operator_list, dtype=torch.float64, device=device)
        indices = torch.round(data[:, :-1]).to(dtype=torch.int64, device=device)
        values = data[:, -1]

        return cls(indices, values, nstat)

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
            raise ValueError(f"Expected (N, 2) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 2

    def to_dense(self, n_states: int = None):
        """Returns N x N dense matrix representation. If n_states not provided, defaults to self.n_states"""
        n = n_states or self.nstat
        mat = torch.zeros((n, n), dtype=self.values.dtype, device=self.device)
        if len(self) > 0:
            mat.index_put_(
                (self.indices[:, 0], self.indices[:, 1]),
                self.values,
                accumulate=True,
            )
        return mat


class TwoBodyOperator(Operator):
    """
    Represents a 2-body operator V_pqrs.
    Indices shape: (N, 4) -> [p, q, r, s]
    """

    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 4

    def to_dense(self, n_states: int = None):
        """Returns N x N x N x N dense tensor/array."""
        n = n_states or self.nstat
        shape = (n, n, n, n)

        mat = torch.zeros(shape, dtype=self.values.dtype, device=self.device)
        if len(self) > 0:
            # Project (N, 4) indices into coordinate tuples for index_put_
            idx_tuple = (
                self.indices[:, 0],
                self.indices[:, 1],
                self.indices[:, 2],
                self.indices[:, 3],
            )
            mat.index_put_(idx_tuple, self.values, accumulate=True)
        return mat


class ThreeBodyOperator(Operator):
    """
    Represents a 3-body operator W_pqrstu
    Indices shape: (N, 6) -> [p, q, r, s, t, u]
    """

    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 6

