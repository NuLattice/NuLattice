import pytest
import numpy as np
from NuLattice import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator
class TestOperators:
    
    def test_one_body_from_list(self):
        """Test conversion of 1-body list [p, q, val]."""
        nstat = 5
        # Input data with mixed types (floats for indices) to test sanitization
        data = [
            [0, 0, -10.5],
            [1.0, 1.0, 5.0],   # Float indices should become ints
            [0, 1, 0.25]
        ]
        
        op = OneBodyOperator.from_list(data, nstat)
        
        # Checks
        assert isinstance(op, OneBodyOperator)
        assert len(op) == 3
        assert op.nstat == 5
        
        # Verify indices are pure integers
        assert op.indices.dtype == np.int64
        assert np.array_equal(op.indices[1], [1, 1])
        
        # Verify values
        assert op.values.dtype == np.float64
        assert op.values[0] == -10.5
        assert op.values[2] == 0.25
        
        # Test Round Trip
        out_list = op.to_list()
        # Verify structure matches input (converting input tuple types to match output)
        assert out_list[0] == [0, 0, -10.5]
        assert out_list[1] == [1, 1, 5.0]

    def test_two_body_from_list(self):
        """Test conversion of 2-body list [p, q, r, s, val]."""
        nstat = 10
        data = [
            [0, 1, 2, 3, -0.5],
            [4, 5, 6, 7, 1.234e-5]
        ]
        
        op = TwoBodyOperator.from_list(data, nstat)
        
        assert isinstance(op, TwoBodyOperator)
        assert len(op) == 2
        assert op.indices.shape == (2, 4) # Rank 4
        
        # Check specific values
        assert op.indices[0, 3] == 3
        assert op.values[1] == 1.234e-5

    def test_three_body_from_list(self):
        """Test conversion of 3-body list [p, q, r, s, t, u, val]."""
        nstat = 6
        data = [
            [0, 1, 2, 3, 4, 5, -8.0]
        ]
        
        op = ThreeBodyOperator.from_list(data, nstat)
        
        assert isinstance(op, ThreeBodyOperator)
        assert len(op) == 1
        assert op.indices.shape == (1, 6) # Rank 6
        assert op.values[0] == -8.0

    def test_empty_initialization(self):
        """Test edge case for empty input list."""
        nstat = 4
        op = TwoBodyOperator.from_list([], nstat)
        
        assert len(op) == 0
        assert op.indices.shape == (0, 4)
        assert op.values.shape == (0,)
        assert op.to_list() == []

    def test_invalid_rank_raises_error(self):
        """Test that passing wrong dimensions raises ValueError."""
        nstat = 5
        # 1-body data passed to 2-body operator
        data_1b = [[0, 0, 1.0]] 
        
        # Should succeed for OneBody
        OneBodyOperator.from_list(data_1b, nstat)
        
        # Should fail for TwoBody (expects 4 indices + 1 val = 5 cols)
        with pytest.raises(ValueError):
            # from_list logic sees 2 indices + 1 val -> 3 cols. 
            # TwoBodyOperator __init__ checks shape[1] == 4.
            TwoBodyOperator.from_list(data_1b, nstat)

    def test_one_body_to_dense(self):
        """Test the dense matrix conversion helper."""
        nstat = 3
        data = [
            [0, 0, 1.0],
            [1, 2, 0.5],
            [2, 1, 0.5] # Hermitian conjugate part
        ]
        op = OneBodyOperator.from_list(data, nstat)
        dense = op.to_dense()
        
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0]
        ])
        
        assert np.allclose(dense, expected)
