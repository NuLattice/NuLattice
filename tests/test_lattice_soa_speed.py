import pytest
from NuLattice import lattice as aos
from NuLattice import lattice_soa as soa

@pytest.fixture
def lattice_cfg():
    """Default lattice configuration for benchmarking."""
    return {
        "myL": 12,
        "spin": 2,
        "isospin": 2
    }

@pytest.mark.benchmark(group="SP Basis Generation")
def test_basis_aos(benchmark, lattice_cfg):
    """Benchmarks AoS basis generation (returns list of lists)."""
    benchmark(aos._get_sp_basis_np, **lattice_cfg)

@pytest.mark.benchmark(group="SP Basis Generation")
def test_basis_soa(benchmark, lattice_cfg):
    """Benchmarks SoA basis generation (returns contiguous numpy array)."""
    benchmark(soa._get_sp_basis, **lattice_cfg)

@pytest.mark.benchmark(group="Kinetic Energy (1-Body)")
def test_tkin_aos(benchmark, lattice_cfg):
    """Benchmarks AoS Tkin generation."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(aos._Tkin_np, lat, **lattice_cfg)

@pytest.mark.benchmark(group="Kinetic Energy (1-Body)")
def test_tkin_soa(benchmark, lattice_cfg):
    """Benchmarks SoA _Tkin generation returning OneBodyOperator."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(soa._Tkin, lat, **lattice_cfg)

@pytest.mark.benchmark(group="Contact Interaction (2-Body)")
def test_contacts_aos(benchmark, lattice_cfg):
    """Benchmarks AoS contacts generation."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(aos._contacts_np, 1.0, 1.0, lat, **lattice_cfg)

@pytest.mark.benchmark(group="Contact Interaction (2-Body)")
def test_contacts_soa(benchmark, lattice_cfg):
    """Benchmarks SoA _contacts generation returning TwoBodyOperator."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(soa._contacts, 1.0, 1.0, lat, **lattice_cfg)

@pytest.mark.benchmark(group="NNN Interaction (3-Body)")
def test_nnn_aos(benchmark, lattice_cfg):
    """Benchmarks AoS NNNcontact generation."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(aos._NNNcontact_np, 1.0, lat, **lattice_cfg)

@pytest.mark.benchmark(group="NNN Interaction (3-Body)")
def test_nnn_soa(benchmark, lattice_cfg):
    """Benchmarks SoA _NNNcontact generation returning ThreeBodyOperator."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(soa._NNNcontact, 1.0, lat, **lattice_cfg)

@pytest.mark.benchmark(group="Momentum (1-Body)")
def test_momentum_aos(benchmark, lattice_cfg):
    """Benchmarks AoS p_x generation."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(aos._p_np, lat, **lattice_cfg, dim=0)

@pytest.mark.benchmark(group="Momentum (1-Body)")
def test_momentum_soa(benchmark, lattice_cfg):
    """Benchmarks SoA _p generation returning OneBodyOperator."""
    lat = aos.get_lattice(lattice_cfg["myL"])
    benchmark(soa._p, lat, **lattice_cfg, dim=0)
