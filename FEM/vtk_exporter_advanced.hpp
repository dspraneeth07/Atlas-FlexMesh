#pragma once
// FEM/vtk_exporter_advanced.hpp — Production-Grade VTK/ParaView Export
// Enhanced visualization pipeline for publication-quality figures.
#include "fem/fem_types.hpp"
#include <vector>
#include <cstring>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace atlas::fem {

// Nodal solution container: `std::vector<double>` (ux,uy,uz,...)

class VTKExporterAdvanced {
public:
   
    // VTU EXPORT: Single time step with multiple scalar/vector fields
    
    struct FieldData {
        std::string name;
        std::vector<double> values;     // Either per-node or per-element
        int components = 1;             // 1=scalar, 3=vector, 9=tensor
        bool is_cell_data = false;      // false=point data, true=cell data
    };
    
    static void export_to_vtu(
        const std::string& filename,
        const Mesh& mesh,
        const std::vector<double>& u_nodal,
        const std::vector<FieldData>& fields) noexcept
    {
        std::ofstream vtu(filename, std::ios::binary);
        if (!vtu) {
            std::fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename.c_str());
            return;
        }
        
        // Write VTU header (XML format, binary)
        vtu << "<?xml version=\"1.0\"?>\n";
        vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        vtu << "  <UnstructuredGrid>\n";
        
        int n_points = mesh.nodes.size();
        int n_cells = mesh.elements.size();
        
        vtu << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\"" 
            << n_cells << "\">\n";
        
        // POINTS: Compute deformed coordinates (x + u)
        vtu << "      <Points>\n";
        vtu << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">\n";
        
        std::vector<double> deformed_coords;
        deformed_coords.reserve(mesh.nodes.size() * 3);
        for (size_t i=0; i<mesh.nodes.size(); ++i) {
            deformed_coords.push_back(mesh.nodes[i].x + u_nodal[i*3 + 0]);
            deformed_coords.push_back(mesh.nodes[i].y + u_nodal[i*3 + 1]);
            deformed_coords.push_back(mesh.nodes[i].z + u_nodal[i*3 + 2]);
        }
        
        write_binary_array(vtu, deformed_coords);
        vtu << "        </DataArray>\n";
        vtu << "      </Points>\n";
        
        // CELLS: Element connectivity
        vtu << "      <Cells>\n";
        vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">\n";
        
        std::vector<int> connectivity;
        connectivity.reserve(n_cells * 4);
        for (const auto& elem : mesh.elements) {
            for (const auto& nid : elem.nodes) {
                connectivity.push_back(static_cast<int>(nid));
            }
        }
        write_binary_array(vtu, connectivity);
        vtu << "        </DataArray>\n";
        
        // Offsets (end of each cell)
        vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">\n";
        std::vector<int> offsets;
        for (int i=1; i<=n_cells; ++i) {
            offsets.push_back(i * 4);  // 4 nodes per tet
        }
        write_binary_array(vtu, offsets);
        vtu << "        </DataArray>\n";
        
        // Cell types (10 = VTK_TETRA)
        vtu << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n";
        std::vector<unsigned char> types(n_cells, 10);  // VTK_TETRA
        write_binary_array(vtu, types);
        vtu << "        </DataArray>\n";
        vtu << "      </Cells>\n";
        
        // POINT DATA: Scalars and vectors
        vtu << "      <PointData>\n";
        
        for (const auto& field : fields) {
            if (field.is_cell_data) continue;
            
            if (field.components == 1) {
                vtu << "        <DataArray type=\"Float64\" Name=\"" << field.name 
                    << "\" format=\"binary\">\n";
            } else {
                vtu << "        <DataArray type=\"Float64\" Name=\"" << field.name 
                    << "\" NumberOfComponents=\"" << field.components << "\" format=\"binary\">\n";
            }
            
            write_binary_array(vtu, field.values);
            vtu << "        </DataArray>\n";
        }
        
        vtu << "      </PointData>\n";
        
        // CELL DATA
        vtu << "      <CellData>\n";
        for (const auto& field : fields) {
            if (!field.is_cell_data) continue;
            
            vtu << "        <DataArray type=\"Float64\" Name=\"" << field.name 
                << "\" format=\"binary\">\n";
            write_binary_array(vtu, field.values);
            vtu << "        </DataArray>\n";
        }
        vtu << "      </CellData>\n";
        
        vtu << "    </Piece>\n";
        vtu << "  </UnstructuredGrid>\n";
        vtu << "</VTKFile>\n";
        
        vtu.close();
        std::printf("✓ VTU exported: %s (%d points, %d cells)\n", filename.c_str(), n_points, n_cells);
    }
    
    // PVD EXPORT: Time series (for convergence studies, adaptive refinement)
    
    struct TimeStepData {
        double time;                       // Time or refinement level
        std::string vtu_filename;          // Relative path to .vtu file
        Mesh mesh;
        std::vector<double> u_nodal;
        std::vector<FieldData> fields;
    };
    
    static void export_convergence_series(
        const std::string& pvd_filename,
        const std::vector<TimeStepData>& time_steps) noexcept
    {
        std::ofstream pvd(pvd_filename);
        if (!pvd) {
            std::fprintf(stderr, "ERROR: Cannot open %s for writing\n", pvd_filename.c_str());
            return;
        }
        
        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
        pvd << "  <Collection>\n";
        
        // Write each time step
        int step = 0;
        for (const auto& ts : time_steps) {
            // Export individual VTU
            std::string vtu_file = "convergence_step_" + std::to_string(step) + ".vtu";
            export_to_vtu(vtu_file, ts.mesh, ts.u_nodal, ts.fields);
            
            // Add to collection
            pvd << "    <DataSet timestep=\"" << ts.time << "\" group=\"\" part=\"0\"\n";
            pvd << "             file=\"" << vtu_file << "\"/>\n";
            
            step++;
        }
        
        pvd << "  </Collection>\n";
        pvd << "</VTKFile>\n";
        pvd.close();
        
        std::printf("✓ PVD series exported: %s (%zu time steps)\n", 
            pvd_filename.c_str(), time_steps.size());
    }
    
    // ADAPTATION VISUALIZATION: Show mesh refinement levels
    
    static void export_adaptation_history(
        const std::string& output_dir,
        const std::vector<Mesh>& refined_meshes,
        const std::vector<std::vector<double>>& solutions) noexcept
    {
        std::printf("\nAdaptation History Export:\n");
        std::printf("─────────────────────────\n");
        
        std::vector<TimeStepData> history;
        
        for (size_t level=0; level<refined_meshes.size(); ++level) {
            TimeStepData ts;
            ts.time = static_cast<double>(level);
            ts.mesh = refined_meshes[level];
            ts.u_nodal = solutions[level];
            
            // Add refinement metric as cell data
            FieldData refinement_level;
            refinement_level.name = "refinement_level";
            refinement_level.is_cell_data = true;
            refinement_level.values.assign(refined_meshes[level].elements.size(), 
                                          static_cast<double>(level));
            ts.fields.push_back(refinement_level);
            
            history.push_back(ts);
            
            std::printf("  Level %zu: %zu elements\n", 
                level, refined_meshes[level].elements.size());
        }
        
        std::string pvd_path = output_dir + "/adaptation_history.pvd";
        export_convergence_series(pvd_path, history);
    }
    
    // PUBLICATION QUALITY EXPORT: Multi-field snapshot
    
    static void export_publication_snapshot(
        const std::string& filename,
        const Mesh& mesh,
        const std::vector<double>& u_nodal,
        const std::vector<double>& von_mises_field,
        const std::vector<double>& strain_energy_field,
        const std::vector<double>& error_indicator) noexcept
    {
        std::vector<FieldData> fields;
        
        // Von Mises stress
        FieldData von_mises;
        von_mises.name = "von_Mises_stress";
        von_mises.values.assign(von_mises_field.data(), von_mises_field.data() + von_mises_field.size());
        fields.push_back(von_mises);
        
        // Strain energy
        FieldData energy;
        energy.name = "strain_energy_density";
        energy.values.assign(strain_energy_field.data(), strain_energy_field.data() + strain_energy_field.size());
        fields.push_back(energy);
        
        // Error indicator
        FieldData error;
        error.name = "error_indicator";
        error.values.assign(error_indicator.data(), error_indicator.data() + error_indicator.size());
        fields.push_back(error);
        
        export_to_vtu(filename, mesh, u_nodal, fields);
    }
    
private:
    // -----------------------------------------------------------------------
    // BINARY WRITE UTILITIES
    // -----------------------------------------------------------------------
    
    template<typename T>
    static void write_binary_array(std::ofstream& f, const std::vector<T>& data) noexcept {
        // Write length prefix
        uint32_t size_bytes = data.size() * sizeof(T);
        f.write(reinterpret_cast<char*>(&size_bytes), sizeof(uint32_t));
        
        // Write data
        f.write(reinterpret_cast<const char*>(data.data()), size_bytes);
    }
};

} // namespace atlas::fem
