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

    @testset "Tensor ones" begin
        t = ones(TensorF64, 2, 3)
        @test ndims(t) == 2
        @test length(t) == 6
        @test size(t) == (2, 3)

        # All elements should be one
        for i in 1:6
            @test t[i] == 1.0
        end

        # Test tuple form
        t2 = ones(TensorF64, (3, 4))
        @test size(t2) == (3, 4)
        @test t2[1] == 1.0
    end

    @testset "Tensor rand" begin
        t = rand(TensorF64, 2, 3)
        @test ndims(t) == 2
        @test length(t) == 6
        @test size(t) == (2, 3)

        # All elements should be in [0, 1)
        for i in 1:6
            @test 0.0 <= t[i] < 1.0
        end

        # Test tuple form
        t2 = rand(TensorF64, (3, 4))
        @test size(t2) == (3, 4)
        @test 0.0 <= t2[1] < 1.0
    end

    @testset "Tensor randn" begin
        # Use larger size for statistical test
        t = randn(TensorF64, 100)
        @test ndims(t) == 1
        @test length(t) == 100

        # Verify values are roughly normal (mean near 0)
        arr = Array(t)
        mean_val = sum(arr) / length(arr)
        @test abs(mean_val) < 0.5  # Should be near 0

        # Test 2D form
        t2 = randn(TensorF64, 10, 10)
        @test size(t2) == (10, 10)
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

    @testset "Contract VJP" begin
        # Matrix multiplication VJP: A[2x3] * B[3x4] = C[2x4]
        a = TensorF64(2, 3)
        fill!(a, 1.0)
        b = TensorF64(3, 4)
        fill!(b, 1.0)

        # grad_c is 2x4 of ones
        grad_c = TensorF64(2, 4)
        fill!(grad_c, 1.0)

        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        @test size(grad_a) == (2, 3)
        @test size(grad_b) == (3, 4)

        # grad_a[i,j] = sum_k grad_c[i,k] * B[j,k] = sum_k B[j,k] = 4
        arr_ga = Array(grad_a)
        for i in 1:2, j in 1:3
            @test arr_ga[i, j] == 4.0
        end

        # grad_b[j,k] = sum_i A[i,j] * grad_c[i,k] = sum_i A[i,j] = 2
        arr_gb = Array(grad_b)
        for j in 1:3, k in 1:4
            @test arr_gb[j, k] == 2.0
        end

        # Inner product VJP
        v1 = TensorF64([1.0, 2.0, 3.0], (3,))
        v2 = TensorF64([4.0, 5.0, 6.0], (3,))
        grad_out = TensorF64([1.0], (1,))

        gv1, gv2 = contract_vjp(v1, (-1,), v2, (-1,), grad_out)

        @test size(gv1) == (3,)
        @test size(gv2) == (3,)

        # grad_v1 = grad_out * v2
        arr_gv1 = Array(gv1)
        @test arr_gv1 == [4.0, 5.0, 6.0]

        # grad_v2 = grad_out * v1
        arr_gv2 = Array(gv2)
        @test arr_gv2 == [1.0, 2.0, 3.0]

        # Outer product VJP
        u = TensorF64([1.0, 2.0], (2,))
        v = TensorF64([3.0, 4.0, 5.0], (3,))
        grad_outer = TensorF64(2, 3)
        fill!(grad_outer, 1.0)

        gu, gv = contract_vjp(u, (1,), v, (2,), grad_outer)

        @test size(gu) == (2,)
        @test size(gv) == (3,)

        # grad_u[i] = sum_j grad_outer[i,j] * v[j] = sum_j v[j] = 3+4+5 = 12
        arr_gu = Array(gu)
        @test arr_gu == [12.0, 12.0]

        # grad_v[j] = sum_i grad_outer[i,j] * u[i] = sum_i u[i] = 1+2 = 3
        arr_gv = Array(gv)
        @test arr_gv == [3.0, 3.0, 3.0]
    end

    @testset "ChainRules rrule" begin
        using ChainRulesCore

        # Test that rrule is defined and returns correct structure
        a = TensorF64(2, 3)
        fill!(a, 1.0)
        b = TensorF64(3, 4)
        fill!(b, 1.0)

        c, pullback = ChainRulesCore.rrule(contract, a, (1, -1), b, (-1, 2))

        @test size(c) == (2, 4)

        # Test pullback with TensorF64 input
        grad_c = TensorF64(2, 4)
        fill!(grad_c, 1.0)

        tangents = pullback(grad_c)

        # tangents = (NoTangent(), grad_a, NoTangent(), grad_b, NoTangent())
        @test tangents[1] isa NoTangent
        @test tangents[3] isa NoTangent
        @test tangents[5] isa NoTangent

        @test size(tangents[2]) == (2, 3)  # grad_a
        @test size(tangents[4]) == (3, 4)  # grad_b
    end

    @testset "Zygote AD" begin
        using Zygote

        # Matrix multiplication with Zygote
        a = TensorF64(2, 3)
        fill!(a, 1.0)
        b = TensorF64(3, 4)
        fill!(b, 1.0)

        # Define loss function
        function loss(a, b)
            c = contract(a, (1, -1), b, (-1, 2))
            return sum(Array(c))
        end

        # Compute gradients using Zygote
        grad_a, grad_b = Zygote.gradient(loss, a, b)

        # loss = sum(C) where C[i,k] = sum_j A[i,j] * B[j,k]
        # d(loss)/d(A[i,j]) = sum_k B[j,k] = 4 (since B is all ones with 4 columns)
        # d(loss)/d(B[j,k]) = sum_i A[i,j] = 2 (since A is all ones with 2 rows)
        @test size(grad_a) == (2, 3)
        @test size(grad_b) == (3, 4)

        arr_ga = Array(grad_a)
        arr_gb = Array(grad_b)

        for i in 1:2, j in 1:3
            @test arr_ga[i, j] ≈ 4.0
        end

        for j in 1:3, k in 1:4
            @test arr_gb[j, k] ≈ 2.0
        end

        # Inner product with Zygote
        v1 = TensorF64([1.0, 2.0, 3.0], (3,))
        v2 = TensorF64([4.0, 5.0, 6.0], (3,))

        function inner_loss(v1, v2)
            c = contract(v1, (-1,), v2, (-1,))
            return c[1]
        end

        gv1, gv2 = Zygote.gradient(inner_loss, v1, v2)

        # d(v1·v2)/d(v1) = v2, d(v1·v2)/d(v2) = v1
        @test Array(gv1) ≈ [4.0, 5.0, 6.0]
        @test Array(gv2) ≈ [1.0, 2.0, 3.0]
    end
end
