#ifndef TINKER_GPU_E_POLAR_H_
#define TINKER_GPU_E_POLAR_H_

#include "decl_real.h"
#include "gpu/decl_elec.h"

TINKER_NAMESPACE_BEGIN
namespace gpu {
extern int epolar_electyp;
extern std::string epolar_electyp_str;

extern real* polarity;
extern real* pdamp;
extern real* polarity_inv;

extern real* ep;
extern int* nep;
extern real* vir_ep;

int use_epolar();
void get_epolar_type(int& typ, std::string& typ_str);
void e_polar_data(int op);
}
TINKER_NAMESPACE_END

extern "C" {
void tinker_gpu_epolar_coulomb0();
void tinker_gpu_epolar_coulomb1();
void tinker_gpu_epolar_coulomb3();
void tinker_gpu_epolar_coulomb4();
void tinker_gpu_epolar_coulomb5();
void tinker_gpu_epolar_coulomb6();

void tinker_gpu_epolar_ewald0();
void tinker_gpu_epolar_ewald1();
void tinker_gpu_epolar_ewald3();
void tinker_gpu_epolar_ewald4();
void tinker_gpu_epolar_ewald5();
void tinker_gpu_epolar_ewald6();

void tinker_gpu_epolar0();
void tinker_gpu_epolar1();
void tinker_gpu_epolar3();
void tinker_gpu_epolar4();
void tinker_gpu_epolar5();
void tinker_gpu_epolar6();
}

#endif
