#include <iostream>
#include <cstdint>
#include <filesystem>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <curand_kernel.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkXMLPolyDataWriter.h>

#define EXPECTED_ARG_COUNT 3 // <program_name> <particle_count> <steps>

__host__ void printUsage(const std::string &programName);

#define PI 3.14159265358979323846

#define THREAD_COUNT_1D 1024

#define DEFAULT_PARTICLE_COUNT 1'000
#define DEFAULT_STEPS 2'000
#define STEP_SIZE 0.001 // Also known as R

#define WRITE_FREQUENCY 2
#define OUTPUT_DIR "output"

struct Particle
{
    double x, y, z;
};

struct FunctorInitParticles
{
    Particle *particles;

    __host__ __device__ FunctorInitParticles(Particle *particles) : particles(particles) {}

    __device__ void operator()(const size_t idx) const
    {
        particles[idx].x = 0.0;
        particles[idx].y = 0.0;
        particles[idx].z = 0.0;
    }
};

struct FunctorRandomWalk
{
    Particle *positions;
    curandState *states;

    __host__ __device__ FunctorRandomWalk(Particle *positions, curandState *states) : positions(positions), states(states) {}

    __device__ void operator()(const size_t idx) const
    {
        curandState localState = states[idx];

        double theta = PI * curand_uniform(&localState);
        double phi = 2.0 * PI * curand_uniform(&localState);

        double dx = STEP_SIZE * sin(theta) * cos(phi);
        double dy = STEP_SIZE * sin(theta) * sin(phi);
        double dz = STEP_SIZE * cos(theta);

        positions[idx].x += dx;
        positions[idx].y += dy;
        positions[idx].z += dz;

        states[idx] = localState;
    }
};

__global__ void initCurandStates(curandState *states, const size_t stateCount, const size_t seed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stateCount)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

int main(int argc, char *argv[])
{
    size_t particleCount = DEFAULT_PARTICLE_COUNT;
    size_t steps = DEFAULT_STEPS;

    if (argc == EXPECTED_ARG_COUNT)
    {
        particleCount = std::stoul(argv[1]);
        steps = std::stoul(argv[2]);
    }
    else
    {
        printUsage(argv[0]);
        std::cout << "Using default values: " << particleCount << " particles, " << steps << " steps\n";
    }

    if (!std::filesystem::exists(OUTPUT_DIR))
    {
        std::filesystem::create_directory(OUTPUT_DIR);
    }

    thrust::device_vector<Particle> d_particles(particleCount);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(particleCount),
                     FunctorInitParticles(thrust::raw_pointer_cast(d_particles.data())));

    curandState *d_states;
    cudaMalloc(&d_states, particleCount * sizeof(curandState));
    dim3 dimBlock(THREAD_COUNT_1D, 1, 1);
    dim3 dimGrid((particleCount + THREAD_COUNT_1D - 1) / THREAD_COUNT_1D, 1, 1);
    initCurandStates<<<dimGrid, dimBlock>>>(d_states, particleCount, 0);
    cudaDeviceSynchronize();

    for (size_t step = 0; step < steps; ++step)
    {
        if (step % WRITE_FREQUENCY == 0)
        {
            thrust::host_vector<Particle> h_particles = d_particles;
            cudaDeviceSynchronize();
            vtkNew<vtkPoints> points;
            for (size_t i = 0; i < particleCount; ++i)
            {
                points->InsertNextPoint(h_particles[i].x, h_particles[i].y, h_particles[i].z);
            }
            std::stringstream filename;
            filename << OUTPUT_DIR << "/random_walk_step_" << step << ".vtp";
            vtkNew<vtkPolyData> polyData;
            polyData->SetPoints(points);
            vtkNew<vtkXMLPolyDataWriter> writer;
            writer->SetFileName(filename.str().c_str());
            writer->SetInputData(polyData);
            writer->Write();
        }
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(particleCount),
                         FunctorRandomWalk(thrust::raw_pointer_cast(d_particles.data()), d_states));
    }

    cudaFree(d_states);

    return EXIT_SUCCESS;
}

__host__ void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName << " <particle_count> <steps>\n";
    std::cout << "   <particle_count> - number of particles to simulate\n";
    std::cout << "   <steps>          - number of simulation steps to perform\n";
}