using Test
using NDTensorsRS

@testset "NDTensorsRS" begin
    @testset "Tensor creation" begin
        # Create zeros tensor
        t = TensorF64(2, 3)
        @test ndims(t) == 2
        @test length(t) == 6
        @test size(t) == (2, 3)

        # All elements should be zero
        for i in 1:6
            @test t[i] == 0.0
        end
    end

    @testset "Tensor from data" begin
        data = [1.0 3.0 5.0; 2.0 4.0 6.0]  # 2x3 matrix
        t = TensorF64(data, (2, 3))

        @test size(t) == (2, 3)

        # Check column-major order
        @test t[1] == 1.0  # (1,1)
        @test t[2] == 2.0  # (2,1)
        @test t[3] == 3.0  # (1,2)
        @test t[4] == 4.0  # (2,2)
        @test t[5] == 5.0  # (1,3)
        @test t[6] == 6.0  # (2,3)
    end

    @testset "Element access" begin
        t = TensorF64(2, 3)
        t[1] = 1.0
        t[6] = 6.0

        @test t[1] == 1.0
        @test t[6] == 6.0

        # Bounds checking
        @test_throws BoundsError t[0]
        @test_throws BoundsError t[7]
    end

    @testset "Fill" begin
        t = TensorF64(2, 3)
        fill!(t, 42.0)

        for i in 1:6
            @test t[i] == 42.0
        end
    end

    @testset "Copy" begin
        t1 = TensorF64(2, 3)
        fill!(t1, 5.0)

        t2 = copy(t1)
        @test size(t2) == size(t1)
        @test t2[1] == 5.0

        # Modify t2, t1 should be unchanged
        t2[1] = 100.0
        @test t1[1] == 5.0
        @test t2[1] == 100.0
    end

    @testset "Array conversion" begin
        data = [1.0 3.0; 2.0 4.0]  # 2x2 matrix
        t = TensorF64(data, (2, 2))

        arr = Array(t)
        @test arr == data
    end

    @testset "Permutedims" begin
        # 2x3 matrix transpose
        data = [1.0 3.0 5.0; 2.0 4.0 6.0]  # 2x3 matrix
        t = TensorF64(data, (2, 3))

        t2 = permutedims(t, (2, 1))
        @test size(t2) == (3, 2)

        # Check t[i,j] == t2[j,i]
        arr = Array(t)
        arr2 = Array(t2)
        for i in 1:2, j in 1:3
            @test arr[i, j] == arr2[j, i]
        end

        # 3D permutation
        t3d = TensorF64(2, 3, 4)
        for i in 1:24
            t3d[i] = Float64(i)
        end

        t3d_perm = permutedims(t3d, (3, 1, 2))
        @test size(t3d_perm) == (4, 2, 3)

        # Invalid permutation
        @test_throws ArgumentError permutedims(t, (1,))
        @test_throws ArgumentError permutedims(t, (1, 1))
    end

    @testset "Contract" begin
        # Matrix multiplication: A[2x3] * B[3x4] = C[2x4]
        a = TensorF64(2, 3)
        fill!(a, 1.0)
        b = TensorF64(3, 4)
        fill!(b, 1.0)

        # A[1,-1] * B[-1,2] -> C[1,2]
        c = contract(a, (1, -1), b, (-1, 2))
        @test size(c) == (2, 4)

        # Each element should be 3 (sum over contracted dim of size 3)
        for i in 1:8
            @test c[i] == 3.0
        end

        # Inner product
        v1 = TensorF64([1.0, 2.0, 3.0], (3,))
        v2 = TensorF64([4.0, 5.0, 6.0], (3,))
        result = contract(v1, (-1,), v2, (-1,))
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        @test result[1] == 32.0

        # Outer product
        u = TensorF64([1.0, 2.0], (2,))
        v = TensorF64([3.0, 4.0, 5.0], (3,))
        outer = contract(u, (1,), v, (2,))
        @test size(outer) == (2, 3)
        # outer[i,j] = u[i] * v[j]
        arr = Array(outer)
        @test arr[1, 1] == 3.0   # 1*3
        @test arr[2, 1] == 6.0   # 2*3
        @test arr[1, 2] == 4.0   # 1*4
        @test arr[2, 3] == 10.0  # 2*5

        # Dimension mismatch
        a2 = TensorF64(2, 3)
        b2 = TensorF64(4, 5)
        @test_throws DimensionMismatch contract(a2, (1, -1), b2, (-1, 2))
    end
end
