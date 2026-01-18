module NDTensorsRS

using Libdl

export TensorF64, contract

# Load the shared library
const libpath = joinpath(dirname(dirname(@__FILE__)), "deps", "libndtensors_capi.$(dlext)")

function __init__()
    if !isfile(libpath)
        error("Library not found at $libpath. Please run `using Pkg; Pkg.build(\"NDTensorsRS\")`")
    end
end

# Status codes
const NDT_SUCCESS = Cint(0)
const NDT_INVALID_ARGUMENT = Cint(-1)
const NDT_SHAPE_MISMATCH = Cint(-2)
const NDT_INDEX_OUT_OF_BOUNDS = Cint(-3)
const NDT_INTERNAL_ERROR = Cint(-4)
const NDT_INVALID_PERMUTATION = Cint(-5)

"""
    TensorF64

A dense tensor of Float64 values backed by Rust.
"""
mutable struct TensorF64
    ptr::Ptr{Cvoid}

    function TensorF64(ptr::Ptr{Cvoid})
        t = new(ptr)
        finalizer(release!, t)
        return t
    end
end

function release!(t::TensorF64)
    if t.ptr != C_NULL
        ccall((:ndt_tensor_f64_release, libpath), Cvoid, (Ptr{Cvoid},), t.ptr)
        t.ptr = C_NULL
    end
    nothing
end

"""
    TensorF64(shape::NTuple{N, Int}) where N

Create a tensor filled with zeros.
"""
function TensorF64(shape::NTuple{N, Int}) where N
    shape_arr = collect(Csize_t, shape)
    status = Ref{Cint}(-999)
    ptr = ccall(
        (:ndt_tensor_f64_zeros, libpath),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t, Ptr{Cint}),
        shape_arr, N, status
    )
    if status[] != NDT_SUCCESS
        error("Failed to create tensor: status = $(status[])")
    end
    TensorF64(ptr)
end

TensorF64(shape::Vararg{Int, N}) where N = TensorF64(shape)

"""
    TensorF64(data::Array{Float64}, shape::NTuple{N, Int}) where N

Create a tensor from existing data.
"""
function TensorF64(data::Array{Float64}, shape::NTuple{N, Int}) where N
    # Julia arrays are column-major, same as our Rust tensor
    flat_data = vec(data)
    shape_arr = collect(Csize_t, shape)
    status = Ref{Cint}(-999)
    ptr = ccall(
        (:ndt_tensor_f64_from_data, libpath),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Cint}),
        flat_data, length(flat_data), shape_arr, N, status
    )
    if status[] != NDT_SUCCESS
        error("Failed to create tensor from data: status = $(status[])")
    end
    TensorF64(ptr)
end

"""
    ndims(t::TensorF64)

Get the number of dimensions.
"""
function Base.ndims(t::TensorF64)
    Int(ccall((:ndt_tensor_f64_ndim, libpath), Csize_t, (Ptr{Cvoid},), t.ptr))
end

"""
    length(t::TensorF64)

Get the total number of elements.
"""
function Base.length(t::TensorF64)
    Int(ccall((:ndt_tensor_f64_len, libpath), Csize_t, (Ptr{Cvoid},), t.ptr))
end

"""
    size(t::TensorF64)

Get the shape of the tensor.
"""
function Base.size(t::TensorF64)
    nd = ndims(t)
    shape = Vector{Csize_t}(undef, nd)
    status = ccall(
        (:ndt_tensor_f64_shape, libpath),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        t.ptr, shape
    )
    if status != NDT_SUCCESS
        error("Failed to get shape: status = $status")
    end
    Tuple(Int.(shape))
end

"""
    unsafe_data(t::TensorF64)

Get a pointer to the underlying data. The pointer is only valid while the tensor exists.
"""
function unsafe_data(t::TensorF64)
    ccall((:ndt_tensor_f64_data, libpath), Ptr{Cdouble}, (Ptr{Cvoid},), t.ptr)
end

"""
    Array(t::TensorF64)

Convert tensor to a Julia Array. This copies the data.
"""
function Base.Array(t::TensorF64)
    data_ptr = unsafe_data(t)
    len = length(t)
    shape = size(t)
    # Copy data from Rust to Julia
    data = unsafe_wrap(Array, data_ptr, len)
    reshape(copy(data), shape)
end

"""
    getindex(t::TensorF64, i::Int)

Get element by linear index (1-based).
"""
function Base.getindex(t::TensorF64, i::Int)
    @boundscheck if i < 1 || i > length(t)
        throw(BoundsError(t, i))
    end
    out = Ref{Cdouble}(0.0)
    status = ccall(
        (:ndt_tensor_f64_get_linear, libpath),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cdouble}),
        t.ptr, i - 1, out  # Convert to 0-based
    )
    if status == NDT_INDEX_OUT_OF_BOUNDS
        throw(BoundsError(t, i))
    elseif status != NDT_SUCCESS
        error("Failed to get element: status = $status")
    end
    out[]
end

"""
    setindex!(t::TensorF64, v::Float64, i::Int)

Set element by linear index (1-based).
"""
function Base.setindex!(t::TensorF64, v::Float64, i::Int)
    @boundscheck if i < 1 || i > length(t)
        throw(BoundsError(t, i))
    end
    status = ccall(
        (:ndt_tensor_f64_set_linear, libpath),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cdouble),
        t.ptr, i - 1, v  # Convert to 0-based
    )
    if status == NDT_INDEX_OUT_OF_BOUNDS
        throw(BoundsError(t, i))
    elseif status != NDT_SUCCESS
        error("Failed to set element: status = $status")
    end
    v
end

"""
    fill!(t::TensorF64, v::Float64)

Fill tensor with a value.
"""
function Base.fill!(t::TensorF64, v::Float64)
    status = ccall(
        (:ndt_tensor_f64_fill, libpath),
        Cint,
        (Ptr{Cvoid}, Cdouble),
        t.ptr, v
    )
    if status != NDT_SUCCESS
        error("Failed to fill tensor: status = $status")
    end
    t
end

"""
    copy(t::TensorF64)

Create a copy of the tensor.
"""
function Base.copy(t::TensorF64)
    ptr = ccall((:ndt_tensor_f64_clone, libpath), Ptr{Cvoid}, (Ptr{Cvoid},), t.ptr)
    if ptr == C_NULL
        error("Failed to clone tensor")
    end
    TensorF64(ptr)
end

function Base.show(io::IO, t::TensorF64)
    print(io, "TensorF64($(size(t)))")
end

function Base.show(io::IO, ::MIME"text/plain", t::TensorF64)
    println(io, "TensorF64 with shape $(size(t)):")
    show(io, MIME"text/plain"(), Array(t))
end

"""
    permutedims(t::TensorF64, perm)

Permute the dimensions of the tensor.

# Arguments
- `perm`: A tuple or vector specifying the permutation. `perm[i]` gives the source
  dimension for the i-th dimension of the result. Uses 1-based indexing.

# Examples
```julia
t = TensorF64([1.0 3.0; 2.0 4.0], (2, 2))
t2 = permutedims(t, (2, 1))  # Transpose
```
"""
function Base.permutedims(t::TensorF64, perm)
    nd = ndims(t)
    if length(perm) != nd
        throw(ArgumentError("permutation must have length $(nd), got $(length(perm))"))
    end
    # Convert 1-based Julia indexing to 0-based Rust indexing
    perm_arr = collect(Csize_t, p - 1 for p in perm)
    status = Ref{Cint}(-999)
    ptr = ccall(
        (:ndt_tensor_f64_permutedims, libpath),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Cint}),
        t.ptr, perm_arr, nd, status
    )
    if status[] == NDT_INVALID_PERMUTATION
        throw(ArgumentError("invalid permutation: $perm"))
    elseif status[] != NDT_SUCCESS
        error("Failed to permute tensor: status = $(status[])")
    end
    TensorF64(ptr)
end

"""
    contract(a::TensorF64, labels_a, b::TensorF64, labels_b)

Contract two tensors using label-based contraction.

# Arguments
- `a`: First tensor
- `labels_a`: Labels for each dimension of `a`
- `b`: Second tensor
- `labels_b`: Labels for each dimension of `b`

Labels are integers where:
- Negative values indicate contracted indices (matched between tensors)
- Positive values indicate uncontracted indices (appear in output)

# Examples
```julia
# Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
a = TensorF64(2, 3)
fill!(a, 1.0)
b = TensorF64(3, 4)
fill!(b, 1.0)

# A[1,-1] * B[-1,2] -> C[1,2]
c = contract(a, (1, -1), b, (-1, 2))
# c is 2x4, each element is 3.0

# Inner product: sum over all indices
v1 = TensorF64([1.0, 2.0, 3.0], (3,))
v2 = TensorF64([4.0, 5.0, 6.0], (3,))
result = contract(v1, (-1,), v2, (-1,))
# result[1] = 1*4 + 2*5 + 3*6 = 32
```
"""
function contract(a::TensorF64, labels_a, b::TensorF64, labels_b)
    nd_a = ndims(a)
    nd_b = ndims(b)

    if length(labels_a) != nd_a
        throw(ArgumentError("labels_a must have length $(nd_a), got $(length(labels_a))"))
    end
    if length(labels_b) != nd_b
        throw(ArgumentError("labels_b must have length $(nd_b), got $(length(labels_b))"))
    end

    labels_a_arr = collect(Clong, labels_a)
    labels_b_arr = collect(Clong, labels_b)
    status = Ref{Cint}(-999)

    ptr = ccall(
        (:ndt_tensor_f64_contract, libpath),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Clong}, Csize_t, Ptr{Cvoid}, Ptr{Clong}, Csize_t, Ptr{Cint}),
        a.ptr, labels_a_arr, nd_a, b.ptr, labels_b_arr, nd_b, status
    )

    if status[] == NDT_SHAPE_MISMATCH
        throw(DimensionMismatch("contracted dimensions must have matching sizes"))
    elseif status[] != NDT_SUCCESS
        error("Failed to contract tensors: status = $(status[])")
    end

    TensorF64(ptr)
end

end # module
