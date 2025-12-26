// # FILE: .\src\IO\ExportVTK.jl
module ExportVTK 

using Printf 
using Base64 

export export_solution_vti, export_solution

"""
    export_solution_vti(...)

Writes the simulation results to a VTK XML Image Data file (.vti).
Optimized to optionally include Principal Stresses and Directions based on config.
"""
function export_solution_vti(dims::Tuple{Int,Int,Int}, 
                             spacing::Tuple{Float32,Float32,Float32}, 
                             origin::Tuple{Float32,Float32,Float32},
                             density::Vector{Float32}, 
                             l1_stress::Vector{Float32},
                             von_mises::Vector{Float32},
                             principal_vals::Matrix{Float32},
                             principal_max_dirs::Matrix{Float32},
                             principal_min_dirs::Matrix{Float32},
                             config::Dict,
                             filename::String)

    nx, ny, nz = dims
    n_cells = length(density)
    
    if !endswith(filename, ".vti"); filename *= ".vti"; end

    
    out_settings = get(config, "output_settings", Dict())
    save_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
    write_vectors = (lowercase(string(save_vec_val)) == "yes" || save_vec_val == true)

    open(filename, "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        
        # WholeExtent="x1 x2 y1 y2 z1 z2" (0-based node indices, so dims are cell counts)
        extent = "0 $nx 0 $ny 0 $nz"
        dx, dy, dz = spacing
        ox, oy, oz = origin
        
        write(io, "  <ImageData WholeExtent=\"$extent\" Origin=\"$ox $oy $oz\" Spacing=\"$dx $dy $dz\">\n")
        write(io, "    <Piece Extent=\"$extent\">\n")
        write(io, "      <CellData Scalars=\"Density\" Vectors=\"MaxPrincipalDirection\">\n")
        
        
        current_offset = 0
        
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        
        write(io, "        <DataArray type=\"Float32\" Name=\"L1_Stress\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        
        write(io, "        <DataArray type=\"Float32\" Name=\"VonMises\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)

        if write_vectors
            
            write(io, "        <DataArray type=\"Float32\" Name=\"PrincipalValues\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            
            write(io, "        <DataArray type=\"Float32\" Name=\"MaxPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            
            write(io, "        <DataArray type=\"Float32\" Name=\"MinPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)
        end

        write(io, "      </CellData>\n")
        write(io, "    </Piece>\n")
        write(io, "  </ImageData>\n")
        
        
        write(io, "  <AppendedData encoding=\"raw\">\n")
        write(io, "_") 
        
        
        function write_array(arr)
            n_bytes = UInt32(length(arr) * sizeof(Float32))
            write(io, n_bytes)
            write(io, arr)
        end

        write_array(density)
        write_array(l1_stress)
        write_array(von_mises)

        if write_vectors
            
            write_array(vec(principal_vals)) 
            
            write_array(vec(principal_max_dirs))
            
            write_array(vec(principal_min_dirs))
        end
        
        write(io, "\n  </AppendedData>\n")
        write(io, "</VTKFile>\n")
    end
end

function export_solution(nodes, elements, U, F, bc, p_field, vm, voigt, l1, p_max_dir, p_min_dir; 
                         density=nothing, filename="out.vtk", config=nothing, kwargs...)
                          
    if config !== nothing
        geom = config["geometry"]
        nx = Int(geom["nElem_x_computed"])
        ny = Int(geom["nElem_y_computed"])
        nz = Int(geom["nElem_z_computed"])
        dx = Float32(geom["dx_computed"])
        dy = Float32(geom["dy_computed"])
        dz = Float32(geom["dz_computed"])
        
        
        export_solution_vti((nx, ny, nz), (dx, dy, dz), (0f0, 0f0, 0f0), 
                            density, l1, vm, p_field, p_max_dir, p_min_dir, config, filename)
    else
        # Fallback removed as config is guaranteed in current architecture
        println("[WARN] Export skipped: No configuration provided.")
    end
end

end