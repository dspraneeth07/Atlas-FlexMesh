#include "fem/adaptive_fem_engine.hpp"
#include "fem/hardware_metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

#if __has_include(<Eigen/Sparse>)
    #include <Eigen/Sparse>
    #include <Eigen/IterativeLinearSolvers>
    #define ATLAS_HAVE_EIGEN 1
#endif

namespace fs = std::filesystem;

namespace {

struct ValidationProblem {
    atlas::fem::SparseCSR A;
    std::vector<double> b;
    std::vector<double> x_exact;
    std::size_t n{0};
};

struct ResultRow {
    std::string backend;
    std::string problem;
    int threads{1};
    std::size_t n{0};
    double wall_seconds{0.0};
    double energy_joules{std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t cycles{0};
    std::uint64_t instructions{0};
    std::uint64_t llc_misses{0};
    double residual_l2{0.0};
    double residual_linf{0.0};
    std::string note;
};

double l2_norm(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) sum += v * v;
    return std::sqrt(sum);
}

double linf_norm(const std::vector<double>& values) {
    double result = 0.0;
    for (double v : values) result = std::max(result, std::abs(v));
    return result;
}

ValidationProblem build_poisson_problem(std::size_t grid_n) {
    ValidationProblem problem;
    problem.n = grid_n * grid_n;
    problem.A.n_rows = static_cast<std::uint32_t>(problem.n);
    problem.A.row_ptr.resize(problem.n + 1);

    auto index = [grid_n](std::size_t i, std::size_t j) {
        return i * grid_n + j;
    };

    const double h = 1.0 / static_cast<double>(grid_n + 1);
    const double inv_h2 = 1.0 / (h * h);

    problem.x_exact.resize(problem.n);
    for (std::size_t i = 0; i < grid_n; ++i) {
        const double x = static_cast<double>(i + 1) * h;
        for (std::size_t j = 0; j < grid_n; ++j) {
            const double y = static_cast<double>(j + 1) * h;
            problem.x_exact[index(i, j)] = std::sin(3.14159265358979323846 * x)
                                         * std::sin(3.14159265358979323846 * y);
        }
    }

    problem.A.col_idx.reserve(problem.n * 5);
    problem.A.val.reserve(problem.n * 5);
    problem.b.assign(problem.n, 0.0);

    for (std::size_t i = 0; i < grid_n; ++i) {
        for (std::size_t j = 0; j < grid_n; ++j) {
            const std::size_t row = index(i, j);
            problem.A.row_ptr[row] = static_cast<std::uint32_t>(problem.A.val.size());

            problem.A.col_idx.push_back(static_cast<std::uint32_t>(row));
            problem.A.val.push_back(4.0 * inv_h2);

            if (i > 0) {
                problem.A.col_idx.push_back(static_cast<std::uint32_t>(index(i - 1, j)));
                problem.A.val.push_back(-inv_h2);
            }
            if (i + 1 < grid_n) {
                problem.A.col_idx.push_back(static_cast<std::uint32_t>(index(i + 1, j)));
                problem.A.val.push_back(-inv_h2);
            }
            if (j > 0) {
                problem.A.col_idx.push_back(static_cast<std::uint32_t>(index(i, j - 1)));
                problem.A.val.push_back(-inv_h2);
            }
            if (j + 1 < grid_n) {
                problem.A.col_idx.push_back(static_cast<std::uint32_t>(index(i, j + 1)));
                problem.A.val.push_back(-inv_h2);
            }
        }
    }
    problem.A.row_ptr[problem.n] = static_cast<std::uint32_t>(problem.A.val.size());

    std::vector<double> rhs(problem.n, 0.0);
    problem.A.matvec(problem.x_exact.data(), rhs.data());
    problem.b = std::move(rhs);
    return problem;
}

ResultRow run_internal_cg(const ValidationProblem& problem, int threads) {
    omp_set_num_threads(threads);
    std::vector<double> x(problem.n, 0.0);

    atlas::hpc::ScopedHardwareProfiler profiler;
    profiler.start();
    const auto t0 = std::chrono::steady_clock::now();
    const int iterations = atlas::fem::pcg_solve_ilu0(problem.A, problem.b.data(), x.data(), static_cast<std::uint32_t>(problem.n), 1e-10, 4000);
    const auto t1 = std::chrono::steady_clock::now();
    const auto metrics = profiler.stop();

    std::vector<double> residual(problem.n, 0.0);
    problem.A.matvec(x.data(), residual.data());
    for (std::size_t i = 0; i < problem.n; ++i) {
        residual[i] = problem.b[i] - residual[i];
    }

    ResultRow row;
    row.backend = "atlas_internal_pcg_ilu0";
    row.problem = "poisson_2d";
    row.threads = threads;
    row.n = problem.n;
    row.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
    row.energy_joules = metrics.energy_joules.value_or(std::numeric_limits<double>::quiet_NaN());
    row.cycles = metrics.cpu_cycles;
    row.instructions = metrics.instructions;
    row.llc_misses = metrics.llc_misses;
    row.residual_l2 = l2_norm(residual) / std::max(l2_norm(problem.b), 1e-300);
    row.residual_linf = linf_norm(residual);
    row.note = "iterations=" + std::to_string(iterations);
    return row;
}

#if defined(ATLAS_HAVE_EIGEN)
ResultRow run_eigen_cg(const ValidationProblem& problem, int threads) {
    Eigen::setNbThreads(threads);

    using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    SparseMat A(static_cast<int>(problem.n), static_cast<int>(problem.n));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(problem.A.val.size());
    for (std::uint32_t row = 0; row < problem.A.n_rows; ++row) {
        for (std::uint32_t k = problem.A.row_ptr[row]; k < problem.A.row_ptr[row + 1]; ++k) {
            triplets.emplace_back(static_cast<int>(row), static_cast<int>(problem.A.col_idx[k]), problem.A.val[k]);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd b(static_cast<int>(problem.n));
    Eigen::VectorXd x(static_cast<int>(problem.n));
    for (std::size_t i = 0; i < problem.n; ++i) b(static_cast<int>(i)) = problem.b[i];
    x.setZero();

    atlas::hpc::ScopedHardwareProfiler profiler;
    profiler.start();
    const auto t0 = std::chrono::steady_clock::now();
    Eigen::ConjugateGradient<SparseMat, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<double>> solver;
    solver.setMaxIterations(4000);
    solver.setTolerance(1e-10);
    solver.compute(A);
    x = solver.solve(b);
    const auto t1 = std::chrono::steady_clock::now();
    const auto metrics = profiler.stop();

    Eigen::VectorXd residual = b - A * x;
    ResultRow row;
    row.backend = "eigen_cg_incomplete_cholesky";
    row.problem = "poisson_2d";
    row.threads = threads;
    row.n = problem.n;
    row.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
    row.energy_joules = metrics.energy_joules.value_or(std::numeric_limits<double>::quiet_NaN());
    row.cycles = metrics.cpu_cycles;
    row.instructions = metrics.instructions;
    row.llc_misses = metrics.llc_misses;
    row.residual_l2 = residual.norm() / std::max(b.norm(), 1e-300);
    row.residual_linf = residual.cwiseAbs().maxCoeff();
    row.note = "iterations=" + std::to_string(solver.iterations()) + ", error=" + std::to_string(solver.error());
    return row;
}
#endif

std::optional<ResultRow> run_external_backend(const std::string& env_var,
                                              const std::string& backend,
                                              const ValidationProblem& problem,
                                              int threads) {
    const char* command_base = std::getenv(env_var.c_str());
    if (!command_base || std::string_view(command_base).empty()) return std::nullopt;

    const fs::path output = fs::temp_directory_path() / (backend + "_atlas_validation.csv");
    std::string command = std::string(command_base)
        + " --atlas-problem poisson_2d"
        + " --atlas-size " + std::to_string(problem.n)
        + " --atlas-threads " + std::to_string(threads)
        + " --atlas-precision double"
        + " --atlas-output \"" + output.string() + "\"";

    const int rc = std::system(command.c_str());
    if (rc != 0 || !fs::exists(output)) {
        return std::nullopt;
    }

    std::ifstream in(output);
    if (!in) return std::nullopt;

    ResultRow row;
    row.backend = backend;
    row.problem = "poisson_2d";
    row.threads = threads;
    row.n = problem.n;

    std::string line;
    if (!std::getline(in, line)) return std::nullopt;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        row.note = line;
        break;
    }
    return row;
}

void write_csv(const std::vector<ResultRow>& rows, const fs::path& path) {
    std::ofstream out(path);
    out << "backend,problem,threads,n,wall_seconds,energy_joules,cycles,instructions,llc_misses,residual_l2,residual_linf,note\n";
    for (const auto& row : rows) {
        out << row.backend << ','
            << row.problem << ','
            << row.threads << ','
            << row.n << ','
            << std::setprecision(17) << row.wall_seconds << ','
            << std::setprecision(17) << row.energy_joules << ','
            << row.cycles << ','
            << row.instructions << ','
            << row.llc_misses << ','
            << std::setprecision(17) << row.residual_l2 << ','
            << std::setprecision(17) << row.residual_linf << ','
            << '"' << row.note << '"' << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    std::size_t grid_n = 96;
    int threads = std::max(1, omp_get_max_threads());
    fs::path csv_path = "external_validation_results.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--n" && i + 1 < argc) grid_n = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (arg == "--threads" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    omp_set_num_threads(threads);
    const auto problem = build_poisson_problem(grid_n);

    std::vector<ResultRow> rows;
    rows.push_back(run_internal_cg(problem, threads));

#if defined(ATLAS_HAVE_EIGEN)
    rows.push_back(run_eigen_cg(problem, threads));
#else
    std::cout << "Eigen headers not available; skipping Eigen baseline.\n";
#endif

    if (auto row = run_external_backend("ATLAS_PETSC_CMD", "petsc", problem, threads)) rows.push_back(*row);
    if (auto row = run_external_backend("ATLAS_DEALII_CMD", "deal.ii", problem, threads)) rows.push_back(*row);
    if (auto row = run_external_backend("ATLAS_MFEM_CMD", "mfem", problem, threads)) rows.push_back(*row);
    if (auto row = run_external_backend("ATLAS_ABAQUS_CMD", "abaqus", problem, threads)) rows.push_back(*row);

    write_csv(rows, csv_path);

    std::cout << "External validation results written to " << csv_path.string() << '\n';
    for (const auto& row : rows) {
        std::cout << std::setw(28) << std::left << row.backend
                  << " time=" << std::setw(10) << std::right << row.wall_seconds << " s"
                  << " residual_l2=" << row.residual_l2
                  << " energy=" << (std::isfinite(row.energy_joules) ? std::to_string(row.energy_joules) : std::string("n/a"))
                  << '\n';
    }
    return 0;
}
