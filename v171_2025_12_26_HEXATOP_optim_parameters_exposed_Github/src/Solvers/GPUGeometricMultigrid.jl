# FILE: .\src\Solvers\GPUGeometricMultigrid.jl
module GPUGeometricMultigrid

using CUDA
using LinearAlgebra
using SparseArrays
using Printf
using ..Element
using ..Mesh 

export MGWorkspace, setup_multigrid, apply_mg_vcycle!, clear_multigrid_cache!

mutable struct MGWorkspace
    is_initialized::Bool
    levels::Int  

    # Fine Level Dimensions
    nf_x::Int; nf_y::Int; nf_z::Int
    
    # Medium Level (Level 2)
    nc_x::Int; nc_y::Int; nc_z::Int
    r_med::Any        
    x_med::Any        
    diag_med::Any     
    density_med::Any
    
    dx_m::Float32; dy_m::Float32; dz_m::Float32

    # Coarse Level (Level 3)
    n_cst_x::Int; n_cst_y::Int; n_cst_z::Int
    r_cst::Any
    x_cst::Any
    diag_cst::Any
    density_cst::Any
    
    dx_cst::Float32; dy_cst::Float32; dz_cst::Float32

    # Very Coarse Level (Level 4)
    n_l4_x::Int; n_l4_y::Int; n_l4_z::Int
    r_l4::Any
    x_l4::Any
    diag_l4::Any
    density_l4::Any
    
    dx_l4::Float32; dy_l4::Float32; dz_l4::Float32

    # Expansion Buffers
    r_fine_full::Any
    z_fine_full::Any
    
    MGWorkspace() = new(false, 2, 
                        # Fine
                        0,0,0, 
                        # Med
                        0,0,0, nothing, nothing, nothing, nothing, 1.0, 1.0, 1.0, 
                        # Coarse
                        0,0,0, nothing, nothing, nothing, nothing, 1.0, 1.0, 1.0,
                        # L4
                        0,0,0, nothing, nothing, nothing, nothing, 1.0, 1.0, 1.0,
                        # Buffers
                        nothing, nothing)
end

const GLOBAL_MG_CACHE = MGWorkspace()

function compute_diagonal_structured_kernel!(diag_vec, Ke_diag, factors, nElem, nDof, nx, ny, nz)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        
        # Factor is essentially local density * E
        factor = factors[e]
        if abs(factor) < 1.0e-9
            return
        end
        
        # Decompose element index to (ex, ey, ez)
        tmp = e - 1
        nx_long = Int64(nx)
        ny_long = Int64(ny)
        slice = nx_long * ny_long
        
        ez = div(tmp, slice)
        rem = tmp - ez * slice
        ey = div(rem, nx_long)
        ex = rem - ey * nx_long

        # Calculate node indices
        nnx = nx_long + 1
        nny = ny_long + 1
        node_slice = nnx * nny
        base_n = (ex + 1) + ey * nnx + ez * node_slice
        
        # 8 corner nodes of the hexahedron
        n1 = base_n
        n2 = base_n + 1
        n3 = base_n + 1 + nnx
        n4 = base_n + nnx
        n5 = base_n + node_slice
        n6 = base_n + 1 + node_slice
        n7 = base_n + 1 + nnx + node_slice
        n8 = base_n + nnx + node_slice
        
        nodes = (n1, n2, n3, n4, n5, n6, n7, n8)

        # Accumulate diagonal entries
        for i in 1:8
            node = nodes[i]
            
            idx1 = (node - 1) * 3 + 1
            idx3 = (node - 1) * 3 + 3
            
            # Check bounds
            if idx3 <= nDof
                k_val = Ke_diag[(i-1)*3 + 1] * factor
                CUDA.atomic_add!(pointer(diag_vec, idx1), k_val)
                CUDA.atomic_add!(pointer(diag_vec, idx1 + 1), k_val)
                CUDA.atomic_add!(pointer(diag_vec, idx1 + 2), k_val)
            end
        end
    end
    return nothing
end

function restrict_residual_structured_kernel!(r_coarse, r_fine, nx_f, ny_f, nz_f, nx_c, ny_c, nz_c)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nNodes_fine = (nx_f + 1) * (ny_f + 1) * (nz_f + 1)
    
    if idx <= nNodes_fine
        
        # Decompose fine node index
        slice_f = (nx_f + 1) * (ny_f + 1)
        k_f = div(idx - 1, slice_f)
        rem_k = (idx - 1) - k_f * slice_f
        j_f = div(rem_k, nx_f + 1)
        i_f = rem_k - j_f * (nx_f + 1)
        
        # Load fine residual
        f_base = (idx - 1) * 3
        rf_x = r_fine[f_base + 1]
        rf_y = r_fine[f_base + 2]
        rf_z = r_fine[f_base + 3]
        
        # Skip empty
        if abs(rf_x) + abs(rf_y) + abs(rf_z) < 1.0f-12
            return nothing
        end

        # Calculate position in coarse grid coordinates (Implicit geometry)
        x_c_pos = Float32(i_f) * 0.5f0
        y_c_pos = Float32(j_f) * 0.5f0
        z_c_pos = Float32(k_f) * 0.5f0
        
        i0 = floor(Int, x_c_pos)
        j0 = floor(Int, y_c_pos)
        k0 = floor(Int, z_c_pos)
        
        tx = x_c_pos - Float32(i0)
        ty = y_c_pos - Float32(j0)
        tz = z_c_pos - Float32(k0)
        
        nx_nodes_c = nx_c + 1
        ny_nodes_c = ny_c + 1
        slice_c = nx_nodes_c * ny_nodes_c
        
        # Trilinear Interpolation (Restriction via atomic add)
        
        # Node 000
        ii, jj, kk = i0, j0, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (1.0f0 - ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 100
        ii, jj, kk = i0 + 1, j0, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (1.0f0 - ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 010
        ii, jj, kk = i0, j0 + 1, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 110
        ii, jj, kk = i0 + 1, j0 + 1, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 001
        ii, jj, kk = i0, j0, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (1.0f0 - ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 101
        ii, jj, kk = i0 + 1, j0, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (1.0f0 - ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 011
        ii, jj, kk = i0, j0 + 1, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end

        # Node 111
        ii, jj, kk = i0 + 1, j0 + 1, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                CUDA.atomic_add!(pointer(r_coarse, c_base + 1), rf_x * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 2), rf_y * w)
                CUDA.atomic_add!(pointer(r_coarse, c_base + 3), rf_z * w)
            end
        end
    end
    return nothing
end

function prolongate_correction_structured_kernel!(x_fine, x_coarse, nx_f, ny_f, nz_f, nx_c, ny_c, nz_c)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nNodes_fine = (nx_f + 1) * (ny_f + 1) * (nz_f + 1)
    
    if idx <= nNodes_fine
        
        slice_f = (nx_f + 1) * (ny_f + 1)
        k_f = div(idx - 1, slice_f)
        rem_k = (idx - 1) - k_f * slice_f
        j_f = div(rem_k, nx_f + 1)
        i_f = rem_k - j_f * (nx_f + 1)
        
        # Calculate position in coarse grid coordinates
        x_c_pos = Float32(i_f) * 0.5f0
        y_c_pos = Float32(j_f) * 0.5f0
        z_c_pos = Float32(k_f) * 0.5f0
        
        i0 = floor(Int, x_c_pos)
        j0 = floor(Int, y_c_pos)
        k0 = floor(Int, z_c_pos)
        
        tx = x_c_pos - Float32(i0)
        ty = y_c_pos - Float32(j0)
        tz = z_c_pos - Float32(k0)
        
        val_x = 0.0f0
        val_y = 0.0f0
        val_z = 0.0f0
        
        nx_nodes_c = nx_c + 1
        ny_nodes_c = ny_c + 1
        slice_c = nx_nodes_c * ny_nodes_c
        
        # Trilinear Interpolation (Gather)
        
        # Node 000
        ii, jj, kk = i0, j0, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (1.0f0 - ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 100
        ii, jj, kk = i0 + 1, j0, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (1.0f0 - ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 010
        ii, jj, kk = i0, j0 + 1, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 110
        ii, jj, kk = i0 + 1, j0 + 1, k0
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (ty) * (1.0f0 - tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 001
        ii, jj, kk = i0, j0, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (1.0f0 - ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 101
        ii, jj, kk = i0 + 1, j0, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (1.0f0 - ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 011
        ii, jj, kk = i0, j0 + 1, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (1.0f0 - tx) * (ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end

        # Node 111
        ii, jj, kk = i0 + 1, j0 + 1, k0 + 1
        if ii >= 0 && ii <= nx_c && jj >= 0 && jj <= ny_c && kk >= 0 && kk <= nz_c
            w = (tx) * (ty) * (tz)
            if w > 1.0f-6
                c_idx = (ii + 1) + jj * nx_nodes_c + kk * slice_c
                c_base = (c_idx - 1) * 3
                val_x += x_coarse[c_base + 1] * w
                val_y += x_coarse[c_base + 2] * w
                val_z += x_coarse[c_base + 3] * w
            end
        end
        
        f_base = (idx - 1) * 3
        x_fine[f_base + 1] += val_x
        x_fine[f_base + 2] += val_y
        x_fine[f_base + 3] += val_z
    end
    return nothing
end

function expand_kernel!(x_full, x_free, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_full[map[idx]] = x_free[idx]
    end
    return nothing
end

function contract_add_kernel!(y_free, y_full, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds y_free[idx] += y_full[map[idx]]
    end
    return nothing
end

function setup_multigrid(fine_nodes, fine_density, config)
    ws = GLOBAL_MG_CACHE
    
    if !isdefined(ws, :levels)
        error("MGWorkspace struct definition mismatch. Restart Julia.")
    end

    geom = config["geometry"]
    dx_f = Float32(get(geom, "dx_computed", 1.0)); dy_f = Float32(get(geom, "dy_computed", 1.0)); dz_f = Float32(get(geom, "dz_computed", 1.0))
    nx_f = Int(get(geom, "nElem_x_computed", 0)); ny_f = Int(get(geom, "nElem_y_computed", 0)); nz_f = Int(get(geom, "nElem_z_computed", 0))
    
    use_3_levels = (nx_f >= 16 && ny_f >= 16 && nz_f >= 16)
    
    nx_m_temp = max(1, div(nx_f, 2))
    nx_c_temp = max(1, div(nx_m_temp, 2))
    ny_m_temp = max(1, div(ny_f, 2))
    ny_c_temp = max(1, div(ny_m_temp, 2))
    nz_m_temp = max(1, div(nz_f, 2))
    nz_c_temp = max(1, div(nz_m_temp, 2))

    use_4_levels = use_3_levels && (nx_c_temp >= 16 && ny_c_temp >= 16 && nz_c_temp >= 16)

    if use_4_levels
        ws.levels = 4
    elseif use_3_levels
        ws.levels = 3
    else
        ws.levels = 2
    end
    
    # Store fine grid info
    ws.nf_x = nx_f; ws.nf_y = ny_f; ws.nf_z = nz_f

    # Calculate Medium (Level 2) specs
    nx_m = max(1, div(nx_f, 2)); ny_m = max(1, div(ny_f, 2)); nz_m = max(1, div(nz_f, 2))
    ws.dx_m = dx_f * 2.0; ws.dy_m = dy_f * 2.0; ws.dz_m = dz_f * 2.0
    ws.nc_x = nx_m; ws.nc_y = ny_m; ws.nc_z = nz_m
    
    nElem_m = nx_m * ny_m * nz_m
    nNodes_m = (nx_m + 1) * (ny_m + 1) * (nz_m + 1)
    
    # Initialization Check
    if !ws.is_initialized || length(ws.density_med) != nElem_m
        clear_multigrid_cache!() 
        
        ws.r_med = CUDA.zeros(Float32, nNodes_m * 3)
        ws.x_med = CUDA.zeros(Float32, nNodes_m * 3)
        ws.diag_med = CUDA.zeros(Float32, nNodes_m * 3)
        ws.density_med = CUDA.zeros(Float32, nElem_m)
        
        nNodes_f = (nx_f + 1) * (ny_f + 1) * (nz_f + 1)
        ws.r_fine_full = CUDA.zeros(Float32, nNodes_f * 3)
        ws.z_fine_full = CUDA.zeros(Float32, nNodes_f * 3)
        
        if ws.levels >= 3
            nx_c = max(1, div(nx_m, 2)); ny_c = max(1, div(ny_m, 2)); nz_c = max(1, div(nz_m, 2))
            ws.n_cst_x = nx_c; ws.n_cst_y = ny_c; ws.n_cst_z = nz_c
            ws.dx_cst = ws.dx_m * 2.0; ws.dy_cst = ws.dy_m * 2.0; ws.dz_cst = ws.dz_m * 2.0
            
            nElem_c = nx_c * ny_c * nz_c
            nNodes_c = (nx_c + 1) * (ny_c + 1) * (nz_c + 1)
            
            ws.r_cst = CUDA.zeros(Float32, nNodes_c * 3)
            ws.x_cst = CUDA.zeros(Float32, nNodes_c * 3)
            ws.diag_cst = CUDA.zeros(Float32, nNodes_c * 3)
            ws.density_cst = CUDA.zeros(Float32, nElem_c)
        end

        if ws.levels == 4
            nx_l4 = max(1, div(ws.n_cst_x, 2)); ny_l4 = max(1, div(ws.n_cst_y, 2)); nz_l4 = max(1, div(ws.n_cst_z, 2))
            ws.n_l4_x = nx_l4; ws.n_l4_y = ny_l4; ws.n_l4_z = nz_l4
            ws.dx_l4 = ws.dx_cst * 2.0; ws.dy_l4 = ws.dy_cst * 2.0; ws.dz_l4 = ws.dz_cst * 2.0

            nElem_l4 = nx_l4 * ny_l4 * nz_l4
            nNodes_l4 = (nx_l4 + 1) * (ny_l4 + 1) * (nz_l4 + 1)

            ws.r_l4 = CUDA.zeros(Float32, nNodes_l4 * 3)
            ws.x_l4 = CUDA.zeros(Float32, nNodes_l4 * 3)
            ws.diag_l4 = CUDA.zeros(Float32, nNodes_l4 * 3)
            ws.density_l4 = CUDA.zeros(Float32, nElem_l4)
        end
        
        ws.is_initialized = true
    end
    
    # --- SETUP LEVEL 2 (MEDIUM) ---
    # Initialize uniform density for coarse level
    fill!(ws.density_med, 1.0f0) 
    
    fill!(ws.diag_med, 1.0f0)
    Ke_base_m = Element.get_canonical_stiffness(ws.dx_m, ws.dy_m, ws.dz_m, 0.3f0)
    Ke_diag_m = CuArray(diag(Ke_base_m))
    
    @cuda threads=512 blocks=cld(nElem_m, 512) compute_diagonal_structured_kernel!(
        ws.diag_med, Ke_diag_m, ws.density_med, nElem_m, nNodes_m*3, ws.nc_x, ws.nc_y, ws.nc_z
    )
    CUDA.unsafe_free!(Ke_diag_m)
    
    # --- SETUP LEVEL 3 (COARSE) ---
    if ws.levels >= 3
        # Initialize uniform density for coarse level
        fill!(ws.density_cst, 1.0f0)
        
        fill!(ws.diag_cst, 1.0f0)
        Ke_base_c = Element.get_canonical_stiffness(ws.dx_cst, ws.dy_cst, ws.dz_cst, 0.3f0)
        Ke_diag_c = CuArray(diag(Ke_base_c))
        nElem_c = ws.n_cst_x * ws.n_cst_y * ws.n_cst_z
        nNodes_c = (ws.n_cst_x + 1) * (ws.n_cst_y + 1) * (ws.n_cst_z + 1)
        
        @cuda threads=512 blocks=cld(nElem_c, 512) compute_diagonal_structured_kernel!(
            ws.diag_cst, Ke_diag_c, ws.density_cst, nElem_c, nNodes_c*3, ws.n_cst_x, ws.n_cst_y, ws.n_cst_z
        )
        CUDA.unsafe_free!(Ke_diag_c)
    end

    # --- SETUP LEVEL 4 (VERY COARSE) ---
    if ws.levels == 4
        # Initialize uniform density for coarse level
        fill!(ws.density_l4, 1.0f0)

        fill!(ws.diag_l4, 1.0f0)
        Ke_base_l4 = Element.get_canonical_stiffness(ws.dx_l4, ws.dy_l4, ws.dz_l4, 0.3f0)
        Ke_diag_l4 = CuArray(diag(Ke_base_l4))
        nElem_l4 = ws.n_l4_x * ws.n_l4_y * ws.n_l4_z
        nNodes_l4 = (ws.n_l4_x + 1) * (ws.n_l4_y + 1) * (ws.n_l4_z + 1)
        
        @cuda threads=512 blocks=cld(nElem_l4, 512) compute_diagonal_structured_kernel!(
            ws.diag_l4, Ke_diag_l4, ws.density_l4, nElem_l4, nNodes_l4*3, ws.n_l4_x, ws.n_l4_y, ws.n_l4_z
        )
        CUDA.unsafe_free!(Ke_diag_l4)
    end
    
    return ws
end

function apply_mg_vcycle!(z_fine_free, r_fine_free, ws::MGWorkspace, fine_diag_inv_free, map_gpu, n_free)
    
    # Pre-smoothing (Jacobi)
    @. z_fine_free = r_fine_free * fine_diag_inv_free
    
    # Expand residual to full fine grid
    fill!(ws.r_fine_full, 0.0f0)
    @cuda threads=512 blocks=cld(n_free, 512) expand_kernel!(ws.r_fine_full, r_fine_free, map_gpu, n_free)
    
    # Restriction to Medium
    fill!(ws.r_med, 0.0f0)
    nNodes_f = (ws.nf_x + 1) * (ws.nf_y + 1) * (ws.nf_z + 1)
    
    @cuda threads=512 blocks=cld(nNodes_f, 512) restrict_residual_structured_kernel!(
        ws.r_med, ws.r_fine_full, ws.nf_x, ws.nf_y, ws.nf_z, ws.nc_x, ws.nc_y, ws.nc_z
    )
    
    # Solve on Medium (Jacobi)
    @. ws.x_med = ws.r_med / (ws.diag_med + 1.0f-9)
    
    if ws.levels >= 3
        # Restriction to Coarse
        fill!(ws.r_cst, 0.0f0)
        nNodes_m = (ws.nc_x + 1) * (ws.nc_y + 1) * (ws.nc_z + 1)
        
        @cuda threads=512 blocks=cld(nNodes_m, 512) restrict_residual_structured_kernel!(
            ws.r_cst, ws.r_med, ws.nc_x, ws.nc_y, ws.nc_z, ws.n_cst_x, ws.n_cst_y, ws.n_cst_z
        )
        
        # Solve on Coarse (Jacobi)
        @. ws.x_cst = ws.r_cst / (ws.diag_cst + 1.0f-9)

        if ws.levels == 4
            # Restriction to Level 4
            fill!(ws.r_l4, 0.0f0)
            nNodes_c = (ws.n_cst_x + 1) * (ws.n_cst_y + 1) * (ws.n_cst_z + 1)

            @cuda threads=512 blocks=cld(nNodes_c, 512) restrict_residual_structured_kernel!(
                ws.r_l4, ws.r_cst, ws.n_cst_x, ws.n_cst_y, ws.n_cst_z, ws.n_l4_x, ws.n_l4_y, ws.n_l4_z
            )

            # Solve on Level 4 (Jacobi)
            @. ws.x_l4 = ws.r_l4 / (ws.diag_l4 + 1.0f-9)

            # Prolongate back to Coarse
            @cuda threads=512 blocks=cld(nNodes_c, 512) prolongate_correction_structured_kernel!(
                ws.x_cst, ws.x_l4, ws.n_cst_x, ws.n_cst_y, ws.n_cst_z, ws.n_l4_x, ws.n_l4_y, ws.n_l4_z
            )
        end
        
        # Prolongate back to Medium
        @cuda threads=512 blocks=cld(nNodes_m, 512) prolongate_correction_structured_kernel!(
            ws.x_med, ws.x_cst, ws.nc_x, ws.nc_y, ws.nc_z, ws.n_cst_x, ws.n_cst_y, ws.n_cst_z
        )
    end
    
    # Prolongate back to Fine
    fill!(ws.z_fine_full, 0.0f0)
    @cuda threads=512 blocks=cld(nNodes_f, 512) prolongate_correction_structured_kernel!(
        ws.z_fine_full, ws.x_med, ws.nf_x, ws.nf_y, ws.nf_z, ws.nc_x, ws.nc_y, ws.nc_z
    )
    
    # Add correction to solution on free DOFs
    @cuda threads=512 blocks=cld(n_free, 512) contract_add_kernel!(z_fine_free, ws.z_fine_full, map_gpu, n_free)
    
    return nothing
end

function clear_multigrid_cache!()
    ws = GLOBAL_MG_CACHE
    if ws.is_initialized
        if ws.r_med !== nothing; CUDA.unsafe_free!(ws.r_med); ws.r_med = nothing; end
        if ws.x_med !== nothing; CUDA.unsafe_free!(ws.x_med); ws.x_med = nothing; end
        if ws.diag_med !== nothing; CUDA.unsafe_free!(ws.diag_med); ws.diag_med = nothing; end
        if ws.density_med !== nothing; CUDA.unsafe_free!(ws.density_med); ws.density_med = nothing; end
        
        if ws.r_cst !== nothing; CUDA.unsafe_free!(ws.r_cst); ws.r_cst = nothing; end
        if ws.x_cst !== nothing; CUDA.unsafe_free!(ws.x_cst); ws.x_cst = nothing; end
        if ws.diag_cst !== nothing; CUDA.unsafe_free!(ws.diag_cst); ws.diag_cst = nothing; end
        if ws.density_cst !== nothing; CUDA.unsafe_free!(ws.density_cst); ws.density_cst = nothing; end
        
        if ws.r_l4 !== nothing; CUDA.unsafe_free!(ws.r_l4); ws.r_l4 = nothing; end
        if ws.x_l4 !== nothing; CUDA.unsafe_free!(ws.x_l4); ws.x_l4 = nothing; end
        if ws.diag_l4 !== nothing; CUDA.unsafe_free!(ws.diag_l4); ws.diag_l4 = nothing; end
        if ws.density_l4 !== nothing; CUDA.unsafe_free!(ws.density_l4); ws.density_l4 = nothing; end
        
        if ws.r_fine_full !== nothing; CUDA.unsafe_free!(ws.r_fine_full); ws.r_fine_full = nothing; end
        if ws.z_fine_full !== nothing; CUDA.unsafe_free!(ws.z_fine_full); ws.z_fine_full = nothing; end

        ws.is_initialized = false
    end
end

end