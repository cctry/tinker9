#ifndef TINKER_E_TORTOR_H_
#define TINKER_E_TORTOR_H_

#include "energy_buffer.h"
#include "ext/tinker/detail/ktrtor.hh"
#include "rc_man.h"

TINKER_NAMESPACE_BEGIN
// module bitor
TINKER_EXTERN int nbitor;
TINKER_EXTERN int (*ibitor)[5];

// module tortor
TINKER_EXTERN int ntortor;
TINKER_EXTERN int (*itt)[3];

// module ktrtor
TINKER_EXTERN int *tnx, *tny; // of size maxntt
// of size (maxtgrd,maxntt) i.e. [maxntt][maxtgrd]
TINKER_EXTERN real (*ttx)[ktrtor::maxtgrd];
TINKER_EXTERN real (*tty)[ktrtor::maxtgrd];
// of size (maxtgrd2,maxntt) i.e. [maxntt][maxtgrd2]
TINKER_EXTERN real (*tbf)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tbx)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tby)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tbxy)[ktrtor::maxtgrd2];

TINKER_EXTERN int* chkttor_ia_; // of size ntortor

TINKER_EXTERN real ttorunit;

TINKER_EXTERN BondedEnergy ett_handle;

void etortor_data(rc_op op);

void etortor(int vers);
TINKER_NAMESPACE_END

#endif
