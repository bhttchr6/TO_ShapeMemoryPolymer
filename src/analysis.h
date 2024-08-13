#include "petscvec.h"
#include <petscksp.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscis.h>
#include "petscdmda.h"
#include "petscviewer.h"
#include <petscsys.h>
 #include <petsc/private/vecscatterimpl.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<numeric>
#include<cmath>
#include<iomanip>
#include<vector>
#include<time.h>
#include"MMA.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
typedef Eigen::SparseMatrix<double> SpMat;
//typedef Eigen::Map<Eigen::VectorXd > vecMap;

typedef struct {
  PetscScalar ux_dof;
  PetscScalar uy_dof;
} ElasticityDOF;

struct all_const{
		int mark_den;
		std::vector <Eigen::MatrixXd> D_iter;
		VectorXd phi_g_total;
		MatrixXd alphar;
		
		MatrixXd C_star;
		MatrixXd F_bar;
		MatrixXd E_bar;
		MatrixXd O_bar;
		MatrixXd M_bar;
		MatrixXd N_bar;
		MatrixXd A_g_inv_B_r;
		MatrixXd diB_r;
		MatrixXd A_g_inv_B_g;

 		VectorXd delta_phi_g;
 		int iter_heat;

 		VectorXd K1;
};


void stiffness(MatrixXd&);

PetscErrorCode physics( Vec &xPhys, Vec &hole_elem, PetscInt nelx, PetscInt nelh_x, PetscInt nely, PetscInt nelh_y, PetscScalar time_spec, PetscInt a, PetscScalar &Obj_val, Vec &dfdrho, DM &elas_da, DM &prop_da, PetscInt optit);



static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM ,PetscInt **_lx,PetscInt **_ly);

static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(ElasticityDOF **fields_F,MatStencil u_eqn[],PetscScalar Fe_u[]);

static PetscErrorCode BCApply_WEST(DM elas_da,PetscMPIInt,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b);

static PetscErrorCode ApplyBC(DM elas_da, PetscMPIInt , Mat KTAN, Vec Fint,IS *is,Mat *KFF, Vec *Fint_f);

 PetscErrorCode assembly(DM elas_da, DM prop_da, PetscMPIInt, PetscInt, PetscScalar delta_t, Vec& rho, Vec &hole_elem,  PetscScalar penal, PetscScalar, PetscScalar*,  Vec& Fint, Vec& dFint_dx, Mat& Ktan,Vec&, Vec*,PetscInt flag);

 static PetscErrorCode DMDABCApplyCompression(DM elas_da,PetscMPIInt rank,Mat A,Vec f);

static PetscErrorCode DMDAViewGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[]);


void shapel(double psi, double eta, MatrixXd C, double& DET, MatrixXd& B, PetscInt int_pt, PetscInt e, PetscInt iter);

void SMP_cycle(PetscMPIInt rank, MatrixXd epsilon_total, PetscInt e, PetscScalar T, PetscScalar *T_hist, PetscScalar delta_t, VectorXd &rho, VectorXd &holes, PetscScalar penal, PetscInt iter, PetscInt int_pt, MatrixXd B, MatrixXd& sigma_vis, MatrixXd& Dmat_vis, MatrixXd& dfint_dx, PetscInt sens_flag);



void param(int phase, VectorXd rho, VectorXd &holes, double penal, int e, MatrixXd& KEQ, MatrixXd& dKEQ_dx, MatrixXd& KNEQ, MatrixXd& dKNEQ_dx, double& neta, double& dneta_dx, MatrixXd& I, PetscInt int_pt, PetscInt iter);

void stiffness(double nu, double E, double dnu_dx, double dE_dx, int pole, MatrixXd& dK_dx);

void paracon(MatrixXd Keq, double neta, double delta_t, MatrixXd Kneq, MatrixXd I,MatrixXd& H, MatrixXd& A, MatrixXd& B);

//void df_du(DM elas_da, DM prop_da, PetscInt nelx, PetscInt nely, PetscScalar delta_t, PetscScalar penal, Vec& xPhys, PetscInt step_num, PetscInt step_den, PetscScalar T, PetscScalar *T_hist, PetscInt iter, PetscInt iter_heat, Mat& dFint_du);
 PetscErrorCode df_du(DM elas_da, DM prop_da, PetscInt rank,  PetscInt nelx, PetscInt nely, PetscScalar delta_t,PetscScalar penal, Vec& xPhys, Vec &hole_elem, PetscInt step_num, PetscInt step_den, PetscScalar T, PetscScalar *T_hist,PetscInt iter, PetscInt iter_heat, Mat& DFDUcoup);

void func_eir(MatrixXd &out_coeff_eir, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num, struct all_const &values,MatrixXd& val);

void func_eig(MatrixXd &out_coeff_eig, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num, struct all_const &values, MatrixXd& val);

void func_eg(MatrixXd &out_coeff_eg, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, struct all_const &values, MatrixXd& val);

void func_ei(MatrixXd &out_coeff_ei,MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num, struct all_const &values, MatrixXd& val);

void func_eis(MatrixXd &out_coeff_eis, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, struct all_const &values, MatrixXd& val);

void func_er(MatrixXd &coeff_er, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, struct all_const &values, MatrixXd& val);

PetscErrorCode  dfdrho_func(DM elas_da, DM prop_da,PetscInt rank, PetscInt step, Vec &dfdrho, Vec &psi_n, Vec &dRdx);

PetscErrorCode filter(Vec &x, DM elas_da,DM prop_da, PetscScalar rmin, Mat &H, Vec &Hs);

PetscErrorCode  func_cons(PetscInt nelx, PetscInt nely, PetscScalar l, PetscScalar w, PetscScalar volfrac, Vec &dfdrho, DM elas_da, DM prop_da, PetscScalar Vlim, Vec &x, Vec &gx_i, Vec &dgdrho);

PetscErrorCode  AllocateMMAwithRestart(PetscInt m, Vec &x, Vec &xPhys, PetscInt *itr, MMA **mma);

PetscErrorCode  Force_ext(DM elas_da,  PetscInt rank, Vec &Fext, PetscInt nelx, PetscInt nely , PetscScalar val);

 PetscErrorCode FFext(DM elas_da, Vec &Fext, PetscInt nelh_x, PetscInt nelh_y, PetscInt rank, PetscInt nelx, PetscInt nely, PetscScalar Force_amt, PetscInt &des_dof_local);

inline PetscBool fexists(const std::string& filename);

PetscErrorCode Reduced_dens(DM prop_da, Vec &hole_elem, IS &is_red, Vec &full_vec_1, Vec &full_vec_2, Vec *red_vec_1, Vec *red_vec_2);

