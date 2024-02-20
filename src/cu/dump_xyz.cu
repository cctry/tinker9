#include "ff/image.h"
#include "ff/molecule.h"
#include "ff/atom.h"

namespace tinker {
    void fromGPU_xyz(real* h_x, real* h_y, real* h_z)
    {
        cudaMemcpy(h_x, x, n * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, y, n * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_z, z, n * sizeof(real), cudaMemcpyDeviceToHost);
    }

    void toGPU_xyz(real* h_x, real* h_y, real* h_z)
    {
        cudaMemcpy(x, h_x, n * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(y, h_y, n * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(z, h_z, n * sizeof(real), cudaMemcpyHostToDevice);
    }
}
