// # FILE: .\src\Main.jl
using Pkg
using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics
using CUDA
using YAML

println("\n>>> SCRIPT START: Loading Modules...")
flush(stdout)

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const LIMITS_FILE = joinpath(PROJECT_ROOT, "configs", "_machine_limits.jl")

if isfile(LIMITS_FILE)
    include(LIMITS_FILE)
    println(">>> [PRE-INIT] Loaded Machine Limits.")
else
    @eval module MachineLimits
        const MAX_GMG_ELEMENTS = 5_000_000
        const MAX_JACOBI_ELEMENTS = 10_000_000
    end
    println(">>> [PRE-INIT] No machine limits found. Using safe defaults.")
end

module HEXA
    using LinearAlgebra
    using SparseArrays
    using Printf
    using Base.Threads
    using JSON
    using Dates
    using Statistics
    using CUDA
    using YAML

    using ..MachineLimits

    const C_RESET   = "\u001b[0m"
    const C_BOLD    = "\u001b[1m"
    const C_RED     = "\u001b[31m"
    const C_GREEN   = "\u001b[32m"
    const C_YELLOW  = "\u001b[33m"
    const C_BLUE    = "\u001b[34m"
    const C_MAGENTA = "\u001b[35m"
    const C_CYAN    = "\u001b[36m"
    # ANSI 256-color code for Orange
    const C_ORANGE  = "\u001b[38;5;208m"

    const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

    include("Utils/Diagnostics.jl")
    include("Utils/Helpers.jl")
    using .Diagnostics
    using .Helpers

    include("Core/Element.jl")
    include("Core/Boundary.jl")
    include("Core/Stress.jl")
    using .Element
    using .Boundary
    using .Stress

    include("Mesh/Mesh.jl")
    include("Mesh/MeshUtilities.jl")
    include("Mesh/MeshPruner.jl")
    include("Mesh/MeshRefiner.jl")
    include("Mesh/MeshShapeProcessing.jl")
    using .Mesh
    using .MeshUtilities
    using .MeshPruner
    using .MeshRefiner
    using .MeshShapeProcessing

    include("Solvers/CPUSolver.jl")
    include("Solvers/GPUGeometricMultigrid.jl")
    include("Solvers/GPUSolver.jl")
    include("Solvers/DirectSolver.jl")
    include("Solvers/IterativeSolver.jl")
    include("Solvers/Solver.jl")
    
    using .CPUSolver
    using .GPUGeometricMultigrid
    using .GPUSolver
    using .DirectSolver
    using .IterativeSolver
    using .Solver

    include("IO/Configuration.jl")
    include("IO/ExportVTK.jl")
    include("IO/Postprocessing.jl")
    
    include("Optimization/GPUExplicitFilter.jl")
    include("Optimization/TopOpt.jl")
    
    using .Configuration
    using .ExportVTK
    using .Postprocessing
    using .GPUExplicitFilter
    using .TopologyOptimization

    function __init__()
        Diagnostics.print_success("HEXA Finite Element Solver initialized")
        Helpers.clear_gpu_memory()
        flush(stdout)
    end
    
    function apply_hardware_profile!(config::Dict)
        gpu_type = get(config, "gpu_profile", "RTX")
        if gpu_type == "AUTO" && CUDA.functional()
            dev_name = CUDA.name(CUDA.device())
            if occursin("V100", dev_name); gpu_type = "V100"; end
            if occursin("A100", dev_name) || occursin("H100", dev_name); gpu_type = "H200"; end
            Diagnostics.print_info("Auto-Detected GPU: $dev_name -> Profile: $gpu_type")
        else
            Diagnostics.print_info("Using Configured Profile: $gpu_type")
        end
        
        mesh_conf = get(config, "mesh_settings", Dict())
        solver = get(config, "solver_parameters", Dict())
        
        config["force_float64"] = false
        if uppercase(gpu_type) in ["H", "H200", "H100", "A100"]
            Diagnostics.print_substep("High-Performance Data Center GPU (H/A-Series). Precision: Float64.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-12)
            solver["diagonal_shift_factor"] = 1.0e-10
            solver["solver_type"] = "gpu"
            config["force_float64"] = true
        elseif uppercase(gpu_type) == "V100"
            Diagnostics.print_substep("Legacy Data Center GPU (Tesla V100). Precision: Float64.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-10)
            solver["diagonal_shift_factor"] = 1.0e-9
            solver["solver_type"] = "gpu"
            config["force_float64"] = true
        else 
            Diagnostics.print_substep("Consumer/Workstation GPU (RTX-Series). Precision: Float32.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-6)
            solver["solver_type"] = "gpu"
            config["force_float64"] = false
        end
        config["mesh_settings"] = mesh_conf
        config["solver_parameters"] = solver
        config["hardware_profile_applied"] = gpu_type
        
        config["machine_limits"] = Dict(
            "MAX_GMG_ELEMENTS" => MachineLimits.MAX_GMG_ELEMENTS,
            "MAX_JACOBI_ELEMENTS" => MachineLimits.MAX_JACOBI_ELEMENTS
        )
    end

    function run_main(input_file=nothing)
        try
            if input_file === nothing
                input_file = joinpath(PROJECT_ROOT, "configs", "optimization_cases.yaml")
            end
            
            if !isfile(input_file)
                error("Input file not found: $input_file")
            end

            raw_config = Configuration.load_configuration(input_file)

            if haskey(raw_config, "batch_queue")
                queue = raw_config["batch_queue"]
                Diagnostics.print_banner("BATCH EXECUTION STARTED: $(length(queue)) Runs")
                
                for (i, run_def) in enumerate(queue)
                    job_name = get(run_def, "job_name", "Run_$i")
                    Diagnostics.print_banner("BATCH RUN $i/$((length(queue))): $job_name", color="\u001b[35m")
                    
                    domain_file = get(run_def, "domain_config", "")
                    solver_file = get(run_def, "solver_config", "")
                    overrides = get(run_def, "overrides", Dict())

                    if isempty(domain_file) || isempty(solver_file)
                        Diagnostics.print_error("Skipping $job_name: Missing config file paths.")
                        continue
                    end

                    if !isabspath(domain_file); domain_file = joinpath(PROJECT_ROOT, domain_file); end
                    if !isabspath(solver_file); solver_file = joinpath(PROJECT_ROOT, solver_file); end

                    try
                        merged_config = Configuration.load_and_merge_configurations(domain_file, solver_file, overrides)
                        _run_safe(merged_config, job_name)
                    catch e_run
                        Diagnostics.print_error("Job $job_name Failed: $e_run")
                        showerror(stdout, e_run, catch_backtrace())
                    end
                    
                    Diagnostics.print_success("Finished Batch Run: $job_name")
                    GC.gc()
                    if CUDA.functional(); CUDA.reclaim(); end
                end
                Diagnostics.print_banner("BATCH QUEUE COMPLETE")

            else
                base_job_name = get(raw_config, "job_name", splitext(basename(input_file))[1])
                _run_safe(raw_config, base_job_name)
            end

        catch e
            if isa(e, InterruptException)
                Diagnostics.print_banner("USER INTERRUPT", color="\u001b[33m")
                println(">>> Simulation stopped by user.")
            else
                Diagnostics.print_banner("FATAL ERROR DETECTED", color="\u001b[31m")
                showerror(stderr, e, catch_backtrace())
            end
            flush(stdout)
        finally
            if CUDA.functional()
                Diagnostics.print_info("Finalizing: Cleaning up GPU Memory...")
                Helpers.clear_gpu_memory()
                Diagnostics.print_success("GPU Memory Released.")
                flush(stdout)
            end
        end
    end

    function _run_safe(current_config::Dict, run_name::String="Simulation")
        Diagnostics.print_banner("HEXA TOPOLOGY OPTIMIZER: $run_name")
        Diagnostics.print_info("Clearing GPU Memory from previous runs...")
        
        if CUDA.functional()
            Helpers.clear_gpu_memory()
            CUDA.device!(0) 
            dev = CUDA.device()
            name = CUDA.name(dev)
            total_mem = CUDA.total_memory()
            
            if CUDA.runtime_version() >= v"11.2"
                threshold = min(total_mem * 0.10, 5 * 1024^3) 
                try
                    pool = CUDA.memory_pool(dev)
                    CUDA.attribute!(pool, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(threshold))
                    Diagnostics.print_success("CUDA memory pool configured successfully")
                catch e
                    Diagnostics.print_warn("Memory pool configuration failed (non-critical): $e")
                end
            end
            
            mem_gb = total_mem / 1024^3
            Diagnostics.print_success("GPU Detected: $name ($(round(mem_gb, digits=2)) GB)")
        else
            Diagnostics.print_warn("No CUDA GPU detected. Running in CPU Mode (Slow).")
        end
        GC.gc()
        
        apply_hardware_profile!(current_config)
        
        restart_conf = get(current_config, "restart_configuration", Dict())
        enable_restart = get(restart_conf, "enable_restart", false)
        restart_path = get(restart_conf, "file_path", "")
        
        config = Dict{Any,Any}()
        density = Float32[]
        start_iter = 1
        restart_radius = 0.0f0
        restart_threshold = 0.0f0
        is_restart_active = false

        if enable_restart
             if isfile(restart_path)
                Diagnostics.print_banner("RESTART MODE ACTIVE", color="\u001b[35m")
                Diagnostics.print_info("Loading checkpoint: $restart_path")
                saved_config, density, saved_iter, restart_radius, restart_threshold = Configuration.load_checkpoint(restart_path)
                config = merge(saved_config, current_config)
                apply_hardware_profile!(config)
                start_iter = saved_iter + 1
                is_restart_active = true
            else
                Diagnostics.print_warn("Restart requested but file not found: '$restart_path'")
                Diagnostics.print_warn("Falling back to FRESH START.")
                config = current_config
                is_restart_active = false
            end
        else
            Diagnostics.print_info("Starting Fresh Simulation")
            config = current_config
            is_restart_active = false
        end
        
        hard_stop_iter = get(config, "hard_stop_after_iteration", -1)
        if hard_stop_iter > -1
            Diagnostics.print_info("HARD STOP ENABLED: Execution will stop after iteration $hard_stop_iter.")
        end

        out_settings = get(config, "output_settings", Dict())
        default_freq = get(out_settings, "export_frequency", 5)
        save_bin_freq = get(out_settings, "save_bin_frequency", default_freq)
        save_stl_freq = get(out_settings, "save_STL_frequency", default_freq)
        save_vtk_freq = get(out_settings, "save_VTK_frequency", default_freq)

        save_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
        save_vectors_bool = (lowercase(string(save_vec_val)) == "yes" || save_vec_val == true)
        
        RESULTS_DIR = joinpath(PROJECT_ROOT, "RESULTS", run_name)
        if !isdir(RESULTS_DIR); mkpath(RESULTS_DIR); end
        Diagnostics.print_info("Output Directory: $RESULTS_DIR")

        raw_log_name = get(out_settings, "log_filename", "simulation_log.txt")
        log_base, log_ext = splitext(basename(raw_log_name))
        log_filename = joinpath(RESULTS_DIR, "$(log_base)_$(run_name)$(log_ext)")
        crash_log_filename = joinpath(RESULTS_DIR, "crash_report_$(run_name).txt")

        iso_threshold_val = get(out_settings, "iso_surface_threshold", 0.8)
        iso_threshold = Float32(iso_threshold_val)
        
        if !is_restart_active
            Diagnostics.init_log_file(log_filename, config)
        else
            Diagnostics.log_status("--- RESTARTING SIMULATION (Iter $start_iter) ---")
        end
        
        geom = Configuration.setup_geometry(config)
        nodes, elements, dims = generate_mesh(geom.nElem_x, geom.nElem_y, geom.nElem_z; dx = geom.dx, dy = geom.dy, dz = geom.dz)
        initial_target_count = size(elements, 1)
        
        if is_restart_active && length(density) != initial_target_count
            error("Restart Mismatch: Checkpoint density size ($(length(density))) != Generated Mesh size ($initial_target_count).")
        end
        
        domain_bounds = (min_pt=[0.0f0,0.0f0,0.0f0], len_x=geom.dx*geom.nElem_x, len_y=geom.dy*geom.nElem_y, len_z=geom.dz*geom.nElem_z)
        config["geometry"]["nElem_x_computed"] = geom.nElem_x
        config["geometry"]["nElem_y_computed"] = geom.nElem_y
        config["geometry"]["nElem_z_computed"] = geom.nElem_z
        config["geometry"]["dx_computed"] = geom.dx
        config["geometry"]["dy_computed"] = geom.dy
        config["geometry"]["dz_computed"] = geom.dz
        config["geometry"]["max_domain_dim"] = geom.max_domain_dim
        
        nNodes = size(nodes, 1)
        ndof = nNodes * 3
        bc_data = config["boundary_conditions"]
        forces_data = config["external_forces"]
        
        bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
        F_external = zeros(Float32, ndof)
        Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
        Diagnostics.print_success("Boundary Conditions & External Forces Mapped.")

        E = Float32(config["material"]["E"])
        nu = Float32(config["material"]["nu"])
        
        material_density = Float32(get(config["material"], "material_density", 0.0))
        gravity_accel = Float32(get(config["material"], "gravity_acceleration", 9.81))
        delta_T = Float32(get(config["material"], "delta_temperature", 0.0))
        if abs(delta_T) > 1e-6; Diagnostics.print_info("THERMOELASTICITY ENABLED: Delta T = $delta_T"); end

        original_density = ones(Float32, size(elements, 1)) 
        protected_elements_mask = falses(size(elements, 1)) 
        alpha_field = zeros(Float32, size(elements, 1))

        if !is_restart_active
            density, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
        else
            _, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
        end
        
        opt_params = config["optimization_parameters"]
        min_density = Float32(get(opt_params, "min_density", 1.0e-3))
        max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))
        max_culling_ratio = Float32(get(opt_params, "max_culling_ratio", 0.05))
        base_name = run_name 
        
        mesh_conf = get(config, "mesh_settings", Dict())
        max_safety_iterations = get(config, "number_of_iterations", 200)

        Diagnostics.print_banner("OPTIMIZATION SCHEDULE")
        println("    Total Iterations: $max_safety_iterations")

        raw_active_target = get(mesh_conf, "final_target_of_active_elements", initial_target_count)
        final_target_active = isa(raw_active_target, String) ? parse(Int, replace(raw_active_target, "_" => "")) : Int(raw_active_target)
        max_growth_rate = Float64(get(mesh_conf, "max_growth_rate", 1.2))
        raw_bg_limit = get(mesh_conf, "max_background_elements", 800_000_000)
        hard_elem_limit = isa(raw_bg_limit, String) ? parse(Int, replace(raw_bg_limit, "_" => "")) : Int(raw_bg_limit)
        Diagnostics.print_info("Hard Element Limit: $(Base.format_bytes(hard_elem_limit * 100)) approx ($hard_elem_limit elems)")

        # --- CONTROL LOGIC INITIALIZATION (ADAPTIVE FEEDBACK) ---
        l1_stress_allowable = Float32(get(config["material"], "l1_stress_allowable", 1.0))
        target_l1_stress = Float32(get(opt_params, "target_l1_stress", l1_stress_allowable))
        
        # PARAMETERS EXTRACTED FROM CONFIG
        STABILITY_TARGET = Float64(get(opt_params, "convergence_stability_tolerance", 0.02))
        STRESS_ACCURACY_TARGET = Float64(get(opt_params, "convergence_stress_error_tolerance", 0.01))
        CONSECUTIVE_ITERS_REQUIRED = Int(get(opt_params, "convergence_required_streak", 5))
        
        # Control Gains
        cg_drop_base = Float32(get(opt_params, "control_gain_drop_base", 0.02))
        cg_drop_prop = Float32(get(opt_params, "control_gain_drop_proportional", 0.15))
        cg_raise_base = Float32(get(opt_params, "control_gain_raise_base", 0.01))
        cg_raise_prop = Float32(get(opt_params, "control_gain_raise_proportional", 0.05))
        
        # Cutoff Control
        cutoff_dec_step = Float32(get(opt_params, "control_cutoff_decrement_step", 0.01))
        cutoff_inc_step = Float32(get(opt_params, "control_cutoff_increment_step", 0.005))
        cutoff_max_limit = Float32(get(opt_params, "control_cutoff_max_limit", 0.4))
        initial_cutoff = Float32(get(opt_params, "initial_culling_cutoff", 0.05))
        
        # Damping
        update_damping_exp = Float32(get(opt_params, "update_damping_exponent", 0.5))
        
        # Radius
        radius_decay_nom = Float32(get(opt_params, "radius_decay_nominal", 0.99))
        radius_decay_agg = Float32(get(opt_params, "radius_decay_aggressive", 0.95))

        Diagnostics.print_info("CONTROL MODE: Adaptive Stress Targeting")
        Diagnostics.print_info("Target L1 Stress = $(round(target_l1_stress, digits=3))")

        internal_l1_allowable = l1_stress_allowable 
        U_full = zeros(Float32, ndof)
        
        # Radius initialization
        avg_element_size = (geom.dx + geom.dy + geom.dz) / 3.0f0
        target_d_phys = Float32(get(opt_params, "minimum_feature_size_physical", 0.0))
        floor_d_elems = Float32(get(opt_params, "minimum_feature_size_elements", 3.0)) 
        d_min_phys = max(target_d_phys, floor_d_elems * avg_element_size)
        R_max = d_min_phys * Float32(get(opt_params, "radius_max_multiplier", 4.0))
        R_min = d_min_phys * Float32(get(opt_params, "radius_min_multiplier", 1.0))
        
        current_radius = is_restart_active ? restart_radius : R_max
        current_cutoff = initial_cutoff
        
        iter = start_iter
        keep_running = true
        
        prev_density = copy(density)
        stability_metric = 1.0
        consecutive_converged_iters = 0
        
        Diagnostics.print_banner("STARTING MAIN LOOP")
        Diagnostics.print_info("Log File: $log_filename")
        
        flush(stdout) 

        while keep_running
            iter_start_time = time()
            status_msg = "Optimizing"
            config["current_outer_iter"] = iter
            
            # --- MESH REFINEMENT CHECK ---
            current_active = count(d -> d > 0.01, density)
            nominal_ref_thresh = Float64(get(mesh_conf, "nominal_refinement_threshold", 0.8))
            
            if current_active < (final_target_active * nominal_ref_thresh) && size(elements, 1) < final_target_active
                 prev_elem_count = size(elements, 1)
                 nodes, elements, density, alpha_field, dims = MeshRefiner.refine_mesh_and_fields(
                    nodes, elements, density, alpha_field, dims, final_target_active, domain_bounds;
                    max_growth_rate = max_growth_rate, hard_element_limit = hard_elem_limit
                )
                GC.gc()
                
                if size(elements, 1) > prev_elem_count
                    status_msg = "Refined"
                    
                    consecutive_converged_iters = 0
                    
                    current_dx = domain_bounds.len_x / (dims[1]-1)
                    avg_element_size = current_dx 
                    R_min = max(target_d_phys, floor_d_elems * avg_element_size)
                    
                    config["geometry"]["nElem_x_computed"] = dims[1]-1
                    config["geometry"]["nElem_y_computed"] = dims[2]-1
                    config["geometry"]["nElem_z_computed"] = dims[3]-1
                    config["geometry"]["dx_computed"] = current_dx
                    config["geometry"]["dy_computed"] = domain_bounds.len_y / (dims[2]-1)
                    config["geometry"]["dz_computed"] = domain_bounds.len_z / (dims[3]-1)
                    
                    geom = (nElem_x=dims[1]-1, nElem_y=dims[2]-1, nElem_z=dims[3]-1, 
                            dx=config["geometry"]["dx_computed"], dy=config["geometry"]["dy_computed"], dz=config["geometry"]["dz_computed"], 
                            shapes=geom.shapes, actual_elem_count=size(elements, 1), max_domain_dim=geom.max_domain_dim)

                    Diagnostics.print_substep("[Refinement] Re-mapping Boundary Conditions & Forces...")
                    nNodes = size(nodes, 1)
                    ndof = nNodes * 3
                    bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
                    F_external = zeros(Float32, ndof)
                    Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
                    _, original_density, protected_elements_mask, _ = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
                    
                    U_full = zeros(Float32, ndof)
                    TopologyOptimization.reset_filter_cache!()
                    prev_density = copy(density)
                end
            end

            # Enforce constraints
            Threads.@threads for e in 1:size(elements, 1)
                if protected_elements_mask[e]; density[e] = original_density[e]; end
            end
            
            F_total = copy(F_external)
            
            if abs(delta_T) > 1e-6
                 Boundary.compute_global_thermal_forces!(F_total, nodes, elements, alpha_field, delta_T, E, nu, density)
            end
            
            Diagnostics.print_substep("FEA Solve (Iter " * C_ORANGE * "$iter" * C_RESET * ")")
            sol_tuple = Solver.solve_system(
                nodes, elements, E, nu, bc_indicator, F_total;
                density=density, config=config, min_stiffness_threshold=min_density, 
                prune_voids=true, u_prev=U_full 
            )
            U_new = sol_tuple[1]
            last_residual = sol_tuple[2]
            prec_used = sol_tuple[3]
            U_full = U_new
            
            if CUDA.functional(); GC.gc(); CUDA.reclaim(); end
            
            compliance = dot(F_total, U_full)
            strain_energy = 0.5 * compliance
            
            Diagnostics.print_substep("Calculating Stress Field...")
            
            principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field = Stress.compute_stress_field(nodes, elements, U_full, E, nu, density; return_voigt=false)
            
            # --- INITIAL EXPORT CHECK ---
            try
                if iter == 1
                    Diagnostics.print_info("Exporting INITIAL REFERENCE STATE (Iter 0)...")
                    do_bin_init = (save_bin_freq > 0); do_stl_init = (save_stl_freq > 0); do_vtk_init = (save_vtk_freq > 0)
                    if do_bin_init || do_stl_init || do_vtk_init
                        Postprocessing.export_iteration_results(0, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                                principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                                principal_max_dir_field, principal_min_dir_field, 
                                                                density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(current_radius), config=config, 
                                                                save_bin=do_bin_init, save_stl=do_stl_init, save_vtk=do_vtk_init)
                    end
                    if hard_stop_iter == 0; println(">>> HARD STOP: Stopping after background analysis (Iter 0)."); keep_running = false; break; end
                end
            catch e_export
                Diagnostics.print_warn("Initial export failed ($e_export). Proceeding with optimization.")
                Diagnostics.write_crash_log(crash_log_filename, "INITIAL_EXPORT", e_export, stacktrace(catch_backtrace()), iter, config, density)
            end
            
            # --- ADAPTIVE CONTROL LOGIC (Second Code Logic) ---
            active_stress_indices = findall(d -> d > 0.1f0, density)
            avg_l1_stress = isempty(active_stress_indices) ? 0.0f0 : Float32(mean(view(l1_stress_norm_field, active_stress_indices)))
            
            stress_error = (avg_l1_stress - target_l1_stress) / target_l1_stress
            
            adaptive_culling_ratio = max_culling_ratio
            
            if stress_error > 0.0
                # Overstress: Weaken material, STOP CULLING, lower cutoff
                adaptive_culling_ratio = 0.0 
                drop_factor = cg_drop_base + cg_drop_prop * clamp(stress_error, 0.0f0, 1.0f0)
                internal_l1_allowable /= (1.0f0 + drop_factor)
                current_cutoff = max(0.001f0, current_cutoff - cutoff_dec_step)
                Diagnostics.print_substep("Control: Stress HIGH ($(round(avg_l1_stress,digits=2))). CULLING DISABLED. Cutoff Dropped.")
            else
                # Understress: Strengthen material, ALLOW CULLING, raise cutoff
                adaptive_culling_ratio = max_culling_ratio 
                raise_factor = cg_raise_base + cg_raise_prop * clamp(abs(stress_error), 0.0f0, 1.0f0)
                internal_l1_allowable *= (1.0f0 + raise_factor)
                current_cutoff = min(cutoff_max_limit, current_cutoff + cutoff_inc_step)
                Diagnostics.print_substep("Control: Stress OK. Optimizing. Raising Cutoff to $(round(current_cutoff,digits=3)).")
            end
            
            internal_l1_allowable = clamp(internal_l1_allowable, l1_stress_allowable * 0.01f0, l1_stress_allowable * 100.0f0)
            
            # Radius Decay Control based on stability
            decay_factor = radius_decay_nom
            if stability_metric < STABILITY_TARGET
                 decay_factor = radius_decay_agg
            end
            current_radius = max(R_min, current_radius * decay_factor)

            Diagnostics.print_substep("Topology Update & Filtering...")
            
            # --- CALL TO TOPOLOGY OPTIMIZATION (Using Second Code Signature) ---
            mean_change, _, actual_cutoff, filter_time, _, _ = TopologyOptimization.update_density!(
                density, l1_stress_norm_field, protected_elements_mask, E, Float32(internal_l1_allowable), 
                iter, max_safety_iterations, 
                original_density, min_density, max_density_clamp, config, elements;
                force_no_cull = false,
                cutoff_threshold = current_cutoff,
                specified_radius = current_radius,
                max_culling_ratio = Float32(adaptive_culling_ratio),
                update_damping = update_damping_exp
            )
            
            if actual_cutoff > current_cutoff
                current_cutoff = actual_cutoff
            end
            
            # --- CONVERGENCE CHECK (Robust) ---
            active_indices = findall(x -> x > min_density, density)
            n_active = length(active_indices)
            
            if n_active > 0
                changed_elements = count(i -> abs(density[i] - prev_density[i]) > 0.01, active_indices)
                stability_metric = changed_elements / n_active
            else
                stability_metric = 0.0
            end
            
            stress_metric = abs(stress_error)
            radius_converged = (current_radius <= (R_min * 1.05))
            
            is_currently_converged = (stability_metric < STABILITY_TARGET) && 
                                     (stress_metric < STRESS_ACCURACY_TARGET) && 
                                     radius_converged

            if is_currently_converged
                consecutive_converged_iters += 1
            else
                consecutive_converged_iters = 0
            end
            
            # --- DETAILED LOGGING (MATCHING SECOND CODE) ---
            println(C_YELLOW * "    [CONVERGENCE CHECK] Iter " * C_ORANGE * "$iter" * C_RESET)
            println("      > Stability:  $(round(stability_metric*100, digits=2))% (Target < $(round(STABILITY_TARGET*100,digits=1))%)")
            println("      > Stress Err: $(round(stress_error*100, digits=2))% (Target +/- $(round(STRESS_ACCURACY_TARGET*100,digits=1))%)")
            println("      > Robustness: $consecutive_converged_iters / $CONSECUTIVE_ITERS_REQUIRED steps")
            
            if consecutive_converged_iters >= CONSECUTIVE_ITERS_REQUIRED
                Diagnostics.print_success("OPTIMIZATION CONVERGENCE REACHED (Stable for $consecutive_converged_iters steps)!")
                keep_running = false
            end
            
            prev_density = copy(density)
            
            # --- LOGGING & EXPORT ---
            iter_time = time() - iter_start_time
            cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
            
            vol_total = length(density)
            vol_frac = sum(density)/vol_total
            
            Diagnostics.write_iteration_log(
                log_filename, iter, cur_dims_str, vol_total, n_active, 
                current_radius, current_cutoff, compliance, strain_energy, avg_l1_stress, 
                vol_frac, mean_change, status_msg, iter_time, last_residual, prec_used
            )

            is_last_iter = (!keep_running) || (hard_stop_iter > 0 && iter >= hard_stop_iter) || (iter >= max_safety_iterations)
            
            do_bin = (save_bin_freq > 0) && (iter % save_bin_freq == 0)
            do_stl = ((save_stl_freq > 0) && (iter % save_stl_freq == 0)) || is_last_iter
            do_vtk = (save_vtk_freq > 0) && (iter % save_vtk_freq == 0)
            
            should_export = do_bin || do_stl || do_vtk || is_last_iter 

            if should_export
                Diagnostics.print_substep("Exporting results...")
                try
                    Postprocessing.export_iteration_results(iter, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                            principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                            principal_max_dir_field, principal_min_dir_field, 
                                                            density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(current_radius), config=config, save_bin=do_bin, save_stl=do_stl, save_vtk=do_vtk)
                catch e_export
                    Diagnostics.print_error("Post-processing failed at Iter $iter. Logged to crash_report. Continuing simulation.")
                    Diagnostics.write_crash_log(crash_log_filename, "ITER_EXPORT", e_export, stacktrace(catch_backtrace()), iter, config, density)
                end
            end
            
            if hard_stop_iter > 0 && iter >= hard_stop_iter; println(">>> HARD STOP: Reached target iteration $hard_stop_iter."); keep_running = false; break; end
            
            if iter >= max_safety_iterations
                Diagnostics.print_success("Reached total iteration limit ($max_safety_iterations).")
                keep_running = false
            end
            
            if CUDA.functional(); Helpers.clear_gpu_memory(); end
            iter += 1
            GC.gc()
            flush(stdout) 
        end
        Diagnostics.log_status("Finished.")
    end

end

function bootstrap()
    println(">>> [BOOTSTRAP] Parsing arguments and launching module...")
    flush(stdout)
    
    config_file = nothing
    if length(ARGS) >= 1
        config_file = ARGS[1]
    end

    HEXA.run_main(config_file)
end

bootstrap()