using Libdl: dlext

const PROJECT_ROOT = dirname(dirname(dirname(@__DIR__)))

# Build Rust project and copy the shared library to deps/
cd(PROJECT_ROOT) do
    run(`cargo build --release -p ndtensors-capi`)
end

libpath = joinpath(PROJECT_ROOT, "target", "release", "libndtensors_capi.$(dlext)")
destpath = joinpath(@__DIR__, "libndtensors_capi.$(dlext)")

if isfile(libpath)
    cp(libpath, destpath; force=true)
    @info "Copied library to $destpath"
else
    error("Library not found at $libpath. Did cargo build succeed?")
end
