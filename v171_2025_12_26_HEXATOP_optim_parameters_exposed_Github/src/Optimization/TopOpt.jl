// # FILE: .\src\Optimization\TopOpt.jl
// 
module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using Base.Threads
using ..Element
using ..Mesh
using ..GPUExplicitFilter
using ..Helpers

export update_density!, reset_filter_cache!

mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

function apply_emergency_box_filter(density::Vector{Float32}, nx::Int, ny::Int, nz::Int)
    println("    [EMERGENCY FILTER] Applying 3x3x3 box filter (CPU)...")
    nElem = length(density)
    filtered = copy(density)
    
    Threads.@threads for k in 2:nz-1
        for j in 2:ny-1
            for i in 2:nx-1
                e = i + (j-1)*nx + (k-1)*nx*ny
                if e < 1 || e > nElem; continue; end
                
                sum_rho = 0.0f0
                count = 0
                for dk in -1:1, dj in -1:1, di in -1:1
                    neighbor_i = i + di; neighbor_j = j + dj; neighbor_k = k + dk
                    if neighbor_i >= 1 && neighbor_i <= nx &&
                       neighbor_j >= 1 && neighbor_j <= ny &&
                       neighbor_k >= 1 && neighbor_k <= nz
                         neighbor_idx = neighbor_i + (neighbor_j-1)*nx + (neighbor_k-1)*nx*ny
                         if neighbor_idx >= 1 && neighbor_idx <= nElem
                             sum_rho += density[neighbor_idx]
                             count += 1
                         end
                    end
                end
                filtered[e] = (count > 0) ? (sum_rho / count) : density[e]
            end
        end
    end
    return filtered
end

function create_transition_zone(protected_mask::BitVector, nx::Int, ny::Int, nz::Int, depth::Int=3)
    nElem = length(protected_mask)
    transition_zone = falses(nElem)
    
    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        if e < 1 || e > nElem; continue; end
        if protected_mask[e]; continue; end
        
        found_protected = false
        for dk in -depth:depth, dj in -depth:depth, di in -depth:depth
            ni = i + di; nj = j + dj; nk = k + dk
            if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                    found_protected = true; break
                end
            end
        end
        if found_protected; transition_zone[e] = true; end
    end
    return transition_zone
end

function blend_transition_zone!(density::Vector{Float32}, 
                                filtered_density::Vector{Float32},
                                protected_mask::BitVector,
                                transition_zone::BitVector,
                                original_density::Vector{Float32},
                                nx::Int, ny::Int, nz::Int,
                                blend_depth::Int=3)
    nElem = length(density)
    Threads.@threads for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                e = i + (j-1)*nx + (k-1)*nx*ny
                if e < 1 || e > nElem || !transition_zone[e]; continue; end
                
                min_dist = blend_depth + 1.0
                for dk in -blend_depth:blend_depth, dj in -blend_depth:blend_depth, di in -blend_depth:blend_depth
                    ni = i + di; nj = j + dj; nk = k + dk
                    if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                        neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                        if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                            dist = sqrt(Float32(di^2 + dj^2 + dk^2))
                            min_dist = min(min_dist, dist)
                        end
                    end
                end
                
                alpha = clamp(min_dist / blend_depth, 0.0f0, 1.0f0)
                smooth_alpha = alpha * alpha * (3.0f0 - 2.0f0 * alpha)
                density[e] = (1.0f0 - smooth_alpha) * original_density[e] + smooth_alpha * filtered_density[e]
            end
        end
    end
end

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_stress_allowable::Float32, 
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict, 
                         elements::Matrix{Int};
                         force_no_cull::Bool=false,
                         cutoff_threshold::Float32=0.05f0,
                         specified_radius::Union{Float32, Nothing}=nothing,
                         max_culling_ratio::Float32=0.05f0,
                         update_damping::Float32=0.5f0)  

    nElem = length(density)
    
    if any(isnan, l1_stress_norm_field)
        return 0.0f0, 0.0f0, 0.0f0, 0.0, 0, 0.0
    end

    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]
    
    nElem_x = Int(geom_params["nElem_x_computed"]) 
    nElem_y = Int(geom_params["nElem_y_computed"])
    nElem_z = Int(geom_params["nElem_z_computed"])
    dx = Float32(geom_params["dx_computed"])
    dy = Float32(geom_params["dy_computed"])
    dz = Float32(geom_params["dz_computed"])
    
    # --- PROPOSE DENSITY (Adaptive / Multiplicative) ---
    proposed_density_field = zeros(Float32, nElem)
    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            current_l1_stress = l1_stress_norm_field[e]
            
            stress_ratio = current_l1_stress / l1_stress_allowable
            
            # Multiplicative update (Variable damping)
            new_val = density[e] * (stress_ratio ^ update_damping)
            
            proposed_density_field[e] = clamp(new_val, min_density, 1.0f0)
        else
            proposed_density_field[e] = original_density[e]
        end
    end

    # --- RADIUS DETERMINATION ---
    R_final = 0.0f0
    if specified_radius !== nothing
        R_final = specified_radius
    else
        avg_element_size = (dx + dy + dz) / 3.0f0
        target_d_phys = Float32(get(opt_params, "minimum_feature_size_physical", 0.0))
        floor_d_elems = Float32(get(opt_params, "minimum_feature_size_elements", 3.0)) 
        d_min_phys = max(target_d_phys, floor_d_elems * avg_element_size)

        t = Float32(iter) / Float32(number_of_iterations)
        t = clamp(t, 0.0f0, 1.0f0)

        gamma = Float32(get(opt_params, "radius_decay_exponent", 1.8))
        r_max_mult = Float32(get(opt_params, "radius_max_multiplier", 4.0))
        r_min_mult = Float32(get(opt_params, "radius_min_multiplier", 0.5))
        
        decay_factor = 1.0f0 - (t^gamma)
        r_baseline = (r_max_mult * d_min_phys) * decay_factor + (r_min_mult * d_min_phys)
        R_floor = 1.0f0 * avg_element_size
        R_final = max(r_baseline, R_floor)
    end
    
    filtered_density_field = proposed_density_field
    filter_time = 0.0
    
    if R_final > 1e-4
        t_start = time()
        filtered_density_field = GPUExplicitFilter.apply_explicit_filter!(
            proposed_density_field, 
            nElem_x, nElem_y, nElem_z,
            dx, dy, dz, R_final,
            min_density 
        )
        filter_time = time() - t_start
        
        if any(isnan, filtered_density_field)
            # --- GPU FILTER FAILURE WARNING ---
            println("\n\u001b[33m>>> [WARN] GPU Filter Stability Limit Exceeded (NaN detected). Switching to CPU Fallback.\u001b[0m")
            filtered_density_field = apply_emergency_box_filter(proposed_density_field, nElem_x, nElem_y, nElem_z)
        end
    end
    
    filtered_density_field = clamp.(filtered_density_field, min_density, max_density)
    
    # --- TRANSITION ZONE BLENDING ---
    avg_element_size = (dx + dy + dz) / 3.0f0
    blend_depth = max(3, round(Int, R_final / avg_element_size / 2))
    transition_zone = create_transition_zone(protected_elements_mask, nElem_x, nElem_y, nElem_z, blend_depth)
    
    blend_transition_zone!(density, filtered_density_field, protected_elements_mask, 
                           transition_zone, original_density, nElem_x, nElem_y, nElem_z, blend_depth)

    # --- ADAPTIVE CULLING LOGIC ---
    n_active_before = count(d -> d > min_density, density)
    
    # Calculate how many we are ALLOWED to remove this step
    target_removal_count = floor(Int, n_active_before * max_culling_ratio)
    
    candidate_scores = Float32[]
    candidate_indices = Int[]
    sizehint!(candidate_scores, nElem)
    sizehint!(candidate_indices, nElem)

    if !force_no_cull
        for e in 1:nElem
            if !protected_elements_mask[e] && !transition_zone[e] && density[e] > min_density
                push!(candidate_scores, filtered_density_field[e])
                push!(candidate_indices, e)
            end
        end
    end
    
    num_candidates = length(candidate_scores)
    effective_cutoff = cutoff_threshold

    if num_candidates > 0 && target_removal_count > 0
        # Sort candidates to find the threshold score for the bottom N elements
        limit_k = min(target_removal_count, num_candidates)
        
        if limit_k > 0
            score_boundary = partialsort(candidate_scores, limit_k)
            effective_cutoff = max(cutoff_threshold, score_boundary)
            
            if effective_cutoff > 0.8f0
                effective_cutoff = 0.8f0
            end
        end
    end
    
    cull_count = 0
    update_method = get(opt_params, "density_update_method", "hard")

    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] && !transition_zone[e]
            
            should_cull = filtered_density_field[e] < effective_cutoff
            
            if should_cull
                density[e] = min_density
            else
                if update_method == "hard"
                    density[e] = 1.0f0
                else
                    density[e] = filtered_density_field[e]
                end
            end
        end
    end

    # Re-enforce protections
    Threads.@threads for e in 1:nElem
        if protected_elements_mask[e]
            density[e] = original_density[e]
        end
    end
    
    n_active_after = count(d -> d > min_density, density)
    actual_removed = n_active_before - n_active_after

    n_chunks = Threads.nthreads()
    chunk_size = cld(nElem, n_chunks)
    partial_changes = Vector{Float32}(undef, n_chunks)
    @sync for (i, chunk_range) in enumerate(Iterators.partition(1:nElem, chunk_size))
        Threads.@spawn begin
            local_change = 0.0f0
            for e in chunk_range
                if !protected_elements_mask[e] 
                   local_change += abs(filtered_density_field[e] - density[e]) 
                end
            end
            partial_changes[i] = local_change
        end
    end
    mean_change = sum(partial_changes) / Float32(nElem)

    println("\n╔═══════════════════════════════════════════════════════════════╗")
    println(@sprintf("║  CONTROLLED TOPOLOGY (Iter %d) ", iter))
    println(@sprintf("║  Base Cutoff: %.4f | Dynamic Cutoff: %.4f", cutoff_threshold, effective_cutoff))
    println(@sprintf("║  Allowed Removal: %d | Actual Removed: %d", target_removal_count, actual_removed))
    println(@sprintf("║  Internal Allowable Stress: %.4f", l1_stress_allowable))
    println("╚═══════════════════════════════════════════════════════════════╝\n")
    
    return mean_change, R_final, effective_cutoff, filter_time, 0, 0.0
end

end