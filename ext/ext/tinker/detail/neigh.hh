#ifndef TINKER_MOD_NEIGH_HH_
#define TINKER_MOD_NEIGH_HH_

#include "util_macro.h"

TINKER_NAMESPACE_BEGIN namespace neigh {
extern int& maxvlst;
extern int& maxelst;
extern int& maxulst;
extern int*& nvlst;
extern int*& vlst;
extern int*& nelst;
extern int*& elst;
extern int*& nulst;
extern int*& ulst;
extern double& lbuffer;
extern double& pbuffer;
extern double& lbuf2;
extern double& pbuf2;
extern double& vbuf2;
extern double& vbufx;
extern double& dbuf2;
extern double& dbufx;
extern double& cbuf2;
extern double& cbufx;
extern double& mbuf2;
extern double& mbufx;
extern double& ubuf2;
extern double& ubufx;
extern double*& xvold;
extern double*& yvold;
extern double*& zvold;
extern double*& xeold;
extern double*& yeold;
extern double*& zeold;
extern double*& xuold;
extern double*& yuold;
extern double*& zuold;
extern int& dovlst;
extern int& dodlst;
extern int& doclst;
extern int& domlst;
extern int& doulst;

#ifdef TINKER_MOD_CPP_
extern "C" int TINKER_MOD(neigh, maxvlst);
extern "C" int TINKER_MOD(neigh, maxelst);
extern "C" int TINKER_MOD(neigh, maxulst);
extern "C" int* TINKER_MOD(neigh, nvlst);
extern "C" int* TINKER_MOD(neigh, vlst);
extern "C" int* TINKER_MOD(neigh, nelst);
extern "C" int* TINKER_MOD(neigh, elst);
extern "C" int* TINKER_MOD(neigh, nulst);
extern "C" int* TINKER_MOD(neigh, ulst);
extern "C" double TINKER_MOD(neigh, lbuffer);
extern "C" double TINKER_MOD(neigh, pbuffer);
extern "C" double TINKER_MOD(neigh, lbuf2);
extern "C" double TINKER_MOD(neigh, pbuf2);
extern "C" double TINKER_MOD(neigh, vbuf2);
extern "C" double TINKER_MOD(neigh, vbufx);
extern "C" double TINKER_MOD(neigh, dbuf2);
extern "C" double TINKER_MOD(neigh, dbufx);
extern "C" double TINKER_MOD(neigh, cbuf2);
extern "C" double TINKER_MOD(neigh, cbufx);
extern "C" double TINKER_MOD(neigh, mbuf2);
extern "C" double TINKER_MOD(neigh, mbufx);
extern "C" double TINKER_MOD(neigh, ubuf2);
extern "C" double TINKER_MOD(neigh, ubufx);
extern "C" double* TINKER_MOD(neigh, xvold);
extern "C" double* TINKER_MOD(neigh, yvold);
extern "C" double* TINKER_MOD(neigh, zvold);
extern "C" double* TINKER_MOD(neigh, xeold);
extern "C" double* TINKER_MOD(neigh, yeold);
extern "C" double* TINKER_MOD(neigh, zeold);
extern "C" double* TINKER_MOD(neigh, xuold);
extern "C" double* TINKER_MOD(neigh, yuold);
extern "C" double* TINKER_MOD(neigh, zuold);
extern "C" int TINKER_MOD(neigh, dovlst);
extern "C" int TINKER_MOD(neigh, dodlst);
extern "C" int TINKER_MOD(neigh, doclst);
extern "C" int TINKER_MOD(neigh, domlst);
extern "C" int TINKER_MOD(neigh, doulst);

int& maxvlst = TINKER_MOD(neigh, maxvlst);
int& maxelst = TINKER_MOD(neigh, maxelst);
int& maxulst = TINKER_MOD(neigh, maxulst);
int*& nvlst = TINKER_MOD(neigh, nvlst);
int*& vlst = TINKER_MOD(neigh, vlst);
int*& nelst = TINKER_MOD(neigh, nelst);
int*& elst = TINKER_MOD(neigh, elst);
int*& nulst = TINKER_MOD(neigh, nulst);
int*& ulst = TINKER_MOD(neigh, ulst);
double& lbuffer = TINKER_MOD(neigh, lbuffer);
double& pbuffer = TINKER_MOD(neigh, pbuffer);
double& lbuf2 = TINKER_MOD(neigh, lbuf2);
double& pbuf2 = TINKER_MOD(neigh, pbuf2);
double& vbuf2 = TINKER_MOD(neigh, vbuf2);
double& vbufx = TINKER_MOD(neigh, vbufx);
double& dbuf2 = TINKER_MOD(neigh, dbuf2);
double& dbufx = TINKER_MOD(neigh, dbufx);
double& cbuf2 = TINKER_MOD(neigh, cbuf2);
double& cbufx = TINKER_MOD(neigh, cbufx);
double& mbuf2 = TINKER_MOD(neigh, mbuf2);
double& mbufx = TINKER_MOD(neigh, mbufx);
double& ubuf2 = TINKER_MOD(neigh, ubuf2);
double& ubufx = TINKER_MOD(neigh, ubufx);
double*& xvold = TINKER_MOD(neigh, xvold);
double*& yvold = TINKER_MOD(neigh, yvold);
double*& zvold = TINKER_MOD(neigh, zvold);
double*& xeold = TINKER_MOD(neigh, xeold);
double*& yeold = TINKER_MOD(neigh, yeold);
double*& zeold = TINKER_MOD(neigh, zeold);
double*& xuold = TINKER_MOD(neigh, xuold);
double*& yuold = TINKER_MOD(neigh, yuold);
double*& zuold = TINKER_MOD(neigh, zuold);
int& dovlst = TINKER_MOD(neigh, dovlst);
int& dodlst = TINKER_MOD(neigh, dodlst);
int& doclst = TINKER_MOD(neigh, doclst);
int& domlst = TINKER_MOD(neigh, domlst);
int& doulst = TINKER_MOD(neigh, doulst);
#endif
} TINKER_NAMESPACE_END

#endif
