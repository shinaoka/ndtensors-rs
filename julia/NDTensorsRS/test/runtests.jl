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
end
