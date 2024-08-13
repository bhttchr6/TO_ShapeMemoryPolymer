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
#include"analysis.h"
#include<vector>
#include<time.h>
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
typedef Eigen::SparseMatrix<double> SpMat;
using std::pow;
#define ngp 2
#define ndof 2
#define ndim 2
#define Tmax 350
#define Tmin 330
#define Tg 340
#define plane_strain 1
#define plane_stress 0

#define GAUSS_POINTS   4
#define NSD   2

using namespace Eigen;
typedef Eigen::Map<MatrixXd,0,Eigen::Stride<1,2> > MatMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, Dynamic, RowMajor> > matMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, 1> > vecMap;


 
 PetscErrorCode df_du(DM elas_da, DM prop_da,PetscInt rank, PetscInt nelx, PetscInt nely, PetscScalar delta_t,PetscScalar penal, Vec& xPhys,Vec &hole_elem, PetscInt step_num, PetscInt step_den,PetscScalar T, PetscScalar *T_hist,PetscInt iter, PetscInt iter_heat,Mat& DFDUcoup)
{

PetscInt         si, sj, ei, ej, nx, ny,e=0,size_x, sizeg_x, sizeg_y, size_y,ctr_x,ctr_y,xp_size, *blk;
DM               cdm, u_cda;
DMDACoor2d       **local_coords;
Vec              coords,local_F,u_local, *u_local_hist=NULL,xp, holesLocalVec;
ElasticityDOF    **ff;
MatStencil        s_u[8];
PetscErrorCode   ierr;
PetscScalar     *u_local_array=NULL, *xp_array=NULL,*holes_array=NULL,  **u_local_hist_array=NULL;




int nel=nelx*nely;

// GET TEMPERATURE DISTRIBUTION
VectorXd temp(iter+1);;
for(int i=0;i<iter+1;i++){
temp(i)=T_hist[i];
}

//temp(iter+1)=T_hist[iter];
//if (rank==0)std::cout<< iter<<std::endl;std::cout<< "*****"<<std::endl;std::cout<< temp<<std::endl;std::cout<< "*****"<<std::endl;


//GET ELEMENTAL DENSITY DISTRIBUTION
 DMCreateLocalVector(prop_da, &xp);
   ierr = DMGlobalToLocalBegin(prop_da, xPhys, INSERT_VALUES, xp); 
   ierr = DMGlobalToLocalEnd(prop_da, xPhys, INSERT_VALUES, xp); 
   VecGetArray(xp,&xp_array);
   VecGetLocalSize(xp,&xp_size); // get size of local density vector
    VectorXd rho(xp_size);

// get holes tags
 DMCreateLocalVector(prop_da, &holesLocalVec);
 ierr = DMGlobalToLocalBegin(prop_da, hole_elem, INSERT_VALUES, holesLocalVec); CHKERRQ(ierr);
 ierr = DMGlobalToLocalEnd(prop_da, hole_elem, INSERT_VALUES, holesLocalVec); CHKERRQ(ierr);
 VecGetArray(holesLocalVec,&holes_array);
VectorXd holes(xp_size);


for(PetscInt i=0;i<xp_size;i++){
rho(i)=xp_array[i];
holes(i) = holes_array[i];
}

VecRestoreArray(xp, &xp_array);
VecRestoreArray(holesLocalVec, &holes_array);
//GET LOCAL COORDINATES VECTOR
   DMGetCoordinateDM(elas_da, &cdm);
   DMGetCoordinatesLocal(elas_da, &coords); // get local coordinates with ghost cells
   DMDAVecGetArray(cdm,coords,&local_coords);  // convert Vec coords into array/structure




//
//int nnodes=(nelx+1)*(nely+1);
//int neq=nnodes*2;
//int n_nodes_x=nelx+1;
//int n_nodes_y=nely+1;



// define other variables
VectorXd delta_phi_g(iter);
VectorXd k(iter);


MatrixXd H_r(3,3);
MatrixXd H_r_inv(3,3);
MatrixXd dHr_dx(3,3);

MatrixXd A_r(3,3);
MatrixXd dAr_dx(3,3);


MatrixXd B_r(3,3);
MatrixXd dBr_dx(3,3);

MatrixXd Kneq_r(3,3);
MatrixXd dKneqr_dx(3,3);

MatrixXd Keq_r(3,3);
MatrixXd dKeqr_dx(3,3);

double neta_r;
double dnetar_dx;

MatrixXd H_g(3,3);
MatrixXd H_g_inv(3,3);
MatrixXd dHg_dx(3,3);

MatrixXd A_g(3,3);
MatrixXd A_g_inv(3,3);
MatrixXd dAg_dx(3,3);


MatrixXd B_g(3,3);
MatrixXd dBg_dx(3,3);


MatrixXd Kneq_g(3,3);
MatrixXd dKneqg_dx(3,3);

MatrixXd Keq_g(3,3);
MatrixXd dKeqg_dx(3,3);

double neta_g;
double dnetag_dx;

MatrixXd I(3,3);

MatrixXd C(iter,3);

int int_pt=0;
int w=1;//weights

VectorXd gp(ngp);
 gp<<0.577350269189626,-0.577350269189626;// gauss quadrature points
double DET;
MatrixXd B(3,8);
//MatrixXd dFint_du(neq,neq);



I=MatrixXd::Identity(3,3);

MatrixXd out_coeff_eir(3,3);
MatrixXd out_coeff_eig(3,3);
MatrixXd out_coeff_eis(3,3);
MatrixXd out_coeff_ei(3,3);
MatrixXd out_coeff_eir_term2(3,3);



MatrixXd deirn_dern(3,3);
MatrixXd deign_dern(3,3);
MatrixXd dein_dern(3,3);
MatrixXd deisn_dern(3,3);
MatrixXd dC_dern(3,3);
MatrixXd deirn_dern_term2(3,3);	
MatrixXd dern_den(3,3);
MatrixXd dfint_dern(3,3);
MatrixXd dfint_den(3,3);
MatrixXd alphar(3,3);
MatrixXd in_coeff_eir(3,3);
MatrixXd in_coeff_eig(3,3);
MatrixXd C_star(3,3);
MatrixXd F_bar(3,3);
MatrixXd E_bar(3,3);
MatrixXd O_bar(3,3);
MatrixXd M_bar(3,3);
MatrixXd N_bar(3,3);
MatrixXd A_g_inv_B_r(3,3);
MatrixXd diB_r(3,3);
MatrixXd A_g_inv_B_g(3,3);

//GET MESH INFO
  DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // w/o the ghost cells
  DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of nodes
  DMDAGetCorners(elas_da, NULL, NULL,0,&size_x,&size_y,NULL); // gets values w/o including ghost cells
  DMDAGetGhostCorners(elas_da, NULL,NULL,0, &sizeg_x,&sizeg_y,NULL);




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//										ELEMENTAL LOOP 


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ctr_y=0;
for (ej = sj; ej < sj+ny; ej++) {
ctr_x=0;
    for (ei = si; ei < si+nx; ei++) {


 PetscScalar el_coords[8];
 PetscScalar *dfintdu_array;
 PetscCalloc1(8*8, &dfintdu_array);  // allocate zeroed space
  matMap dfintdu(dfintdu_array, 8, 8);    // map eigen matrix to memory


// GET ELEMENTAL COORDINATES //
   el_coords[NSD*0+0] = local_coords[ej][ei].x;      el_coords[NSD*0+1] = local_coords[ej][ei].y;
   el_coords[NSD*1+0] = local_coords[ej][ei+1].x;    el_coords[NSD*1+1] = local_coords[ej][ei+1].y;
   el_coords[NSD*2+0] = local_coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = local_coords[ej+1][ei+1].y;
   el_coords[NSD*3+0] = local_coords[ej+1][ei].x;    el_coords[NSD*3+1] = local_coords[ej+1][ei].y;

  MatMap el_coords_eig(el_coords, 4,2);        // convert CCur from scalar to MatrixXd




    //          

	for(int i=0;i<ngp;i++)
	{
		for(int j=0;j<ngp;j++)
		{

		std::vector <Eigen::MatrixXd> D_iter(iter);

		MatrixXd D(3,3);
		MatrixXd D_step(3,3);
		MatrixXd D_step_inv(3,3);
		MatrixXd D_step_den_inv(3,3);
		MatrixXd D_step_den(3,3);
		VectorXd K1(iter);
		K1=VectorXd::Zero(iter);
//	
		double eta=gp(i);
		double psi=gp(j);
		 int_pt=int_pt+1;
		//double rho_e=rho(e);
		shapel(psi,eta,el_coords_eig,DET,B, int_pt, e,iter);
		//shapel(psi,eta,Ccur,DET,B);

		int phase=1;
		param(phase,rho,holes,penal,e,Keq_r,dKeqr_dx,Kneq_r,dKneqr_dx, neta_r,dnetar_dx,I, int_pt, iter);
		paracon(Keq_r,neta_r,delta_t, Kneq_r, I, H_r,A_r,B_r);

		phase=2;
		param(phase,rho,holes,penal,e,Keq_g,dKeqg_dx,Kneq_g,dKneqg_dx, neta_g,dnetag_dx,I, int_pt, iter);
		paracon(Keq_g,neta_g,delta_t, Kneq_g, I, H_g,A_g,B_g);

		A_g_inv=A_g.inverse() ;               
H_r_inv=H_r.inverse();
H_g_inv=H_g.inverse();


double rho_e=rho(e);

		int eta_i_aux=15000;
		int eta_i_smp=10000;
		double eta_i=eta_i_smp+rho_e*(eta_i_aux-eta_i_smp);
		// aux and SMP material
		
		double c_1=2.76e-05;
		double q=3;
		double c_2=4;


		VectorXd phi_g_total(iter+1);
                VectorXd phi_g_1(iter+1);
                VectorXd phi_r_1(iter+1);
                VectorXd phi_g_2(iter+1);
                VectorXd phi_r_2(iter+1);

		// phi_g and phi_r for SMP1
		phi_g_1(0) = 1-(1/(1+c_1*pow((Tmax-temp(0)),c_2)));
		phi_r_1(0) = 1-phi_g_1(0);

		// phi_g and phi_r for SMP2
		double Tg2 = 345;
		phi_g_2(0) =  1-1/(1+exp(-0.66*(temp(0)-Tg2)));
		phi_r_2(0) = 1-phi_g_2(0);


		// phi_g mixture		
		phi_g_total(0) = phi_g_1(0)+rho_e*(phi_g_2(0)-phi_g_1(0));

		VectorXd phi_r_total(iter+1);
		phi_r_total(0)=1-phi_g_total(0);

	// loop over the iterations

		for(int i=0;i<iter;i++){
		
		// for SMP1
		phi_g_1(i+1) = 1-(1/(1+c_1*pow(Tmax-temp(i+1),c_2)));
		phi_r_1(i+1) = 1-phi_g_1(i+1);

		// for SMP2
		phi_g_2(i+1) = 1-1/(1+exp(-0.66*(temp(i+1)-Tg2)));
		phi_r_2(i+1) = 1-phi_g_2(i+1);

		// phi_g mixture
		phi_g_total(i+1) = phi_g_1(i+1)+ rho_e*(phi_g_2(i+1)-phi_g_1(i+1));
		phi_r_total(i+1)=1-phi_g_total(i+1);
		delta_phi_g(i)=phi_g_total(i+1)-phi_g_total(i);

		k(i)=1;
		if(temp(i+1)>temp(i)){
			K1(i)=1/(1-(delta_phi_g(i)/phi_g_total(i+1)));
		}  // if heating
		D=phi_r_total(i+1)*I+phi_g_total(i+1)*(A_g_inv*A_r)+(delta_t/eta_i)*A_r+delta_phi_g(i)*I;
                //D_iter.push_back(D);
                D_iter[i]=D;
		}//end of iter lopp

		int mark_num=step_num-1;
		int mark_den=step_den;

              


		 out_coeff_eir=-phi_g_total(iter)*(A_g_inv*B_r)+(delta_t/eta_i)*B_r;
		 out_coeff_eig=phi_g_total(iter)*(A_g_inv*B_g);
		 out_coeff_eis=-I;
	 	 out_coeff_ei=-I;




		if(step_num+1>=iter_heat){ out_coeff_eis=-K1(step_num)*I;}
		

		// define structure & variables
		alphar=((delta_t/neta_r)*(H_r_inv*Kneq_r));
		in_coeff_eir=H_r_inv*I;
		in_coeff_eig=(H_g_inv*I);
		C_star=(delta_t/neta_g)*(H_g_inv*Kneq_g);
		F_bar=A_g_inv*B_g;
		E_bar=-A_g_inv*B_r;
		O_bar=A_g_inv*A_r;
		M_bar=(delta_t/eta_i)*A_r;
		N_bar=-(delta_t/eta_i)*B_r;
		A_g_inv_B_r=A_g_inv*B_r;
		diB_r=(delta_t/eta_i)*B_r;
		A_g_inv_B_g=A_g_inv*B_g;
				
		all_const values;
		values.mark_den=mark_den;
                values.D_iter=D_iter;
		values.phi_g_total=phi_g_total;
		values.alphar=alphar;
		
		
		values.C_star=C_star;
		values.F_bar=F_bar;
		values.E_bar=E_bar;
		values.O_bar=O_bar;
		values.M_bar=M_bar;
		values.N_bar=N_bar;
		values.A_g_inv_B_r=A_g_inv_B_r;
		values.diB_r=diB_r;
		values.A_g_inv_B_g=A_g_inv_B_g;

		values.delta_phi_g=delta_phi_g;
		values.iter_heat=iter_heat;
		values.K1=K1;
//double tic=clock();		
                func_eir(out_coeff_eir, in_coeff_eir, in_coeff_eig,mark_num, values,deirn_dern);
		func_eig(out_coeff_eig, in_coeff_eir, in_coeff_eig, mark_num,values, deign_dern);
		func_ei(out_coeff_ei, in_coeff_eir, in_coeff_eig,mark_num,values, dein_dern);
		func_eis(out_coeff_eis,in_coeff_eir, in_coeff_eig ,mark_num,values,deisn_dern);



		dC_dern=deirn_dern+deign_dern+dein_dern+deisn_dern;
//if(rank==0){if(e==0){if(int_pt==1){if((step_num==8)&(step_den==2)){std::cout<<iter_heat<<std::endl;std::cout<<dC_dern<<std::endl;}}}}

		out_coeff_eir_term2=B_r;

		func_eir(out_coeff_eir_term2, in_coeff_eir, in_coeff_eig, mark_num,values, deirn_dern_term2);


//double toc=clock()-tic;
//double t_sens=((float)toc)/CLOCKS_PER_SEC;
//std::cout<< e << t_sens <<std::endl;
		D_step=D_iter[step_num];
		D_step_inv=D_step.inverse();

		D_step_den=D_iter[step_den];
		D_step_den_inv=D_step_den.inverse();

		dfint_dern=A_r*(D_step_inv*dC_dern)-deirn_dern_term2;

		dern_den=D_step_den_inv*I;

		dfint_den=dfint_dern*dern_den;

//for(int i=0;i<ngp;i++)
//	{
//		for(int j=0;j<ngp;j++)
//		{
//		double eta=gp(i);
//		double psi=gp(j);
//		 int_pt=int_pt+1;
		
//		shapel(psi,eta,el_coords_eig,DET,B);      // call shape functions

		dfintdu+=B.transpose()*dfint_den*B*w*DET;
	   
	}// end of j loop
}// end of i loop



		// displacement //
		  // node 0 //
	  s_u[0].i = ei;s_u[0].j = ej;s_u[0].c = 0;          // Ux0 //
	  s_u[1].i = ei;s_u[1].j = ej;s_u[1].c = 1;          // Uy0 //

	  // node 1 //
	  s_u[2].i = ei+1;s_u[2].j = ej;s_u[2].c = 0;        // Ux1 //
	  s_u[3].i = ei+1;s_u[3].j = ej;s_u[3].c = 1;        // Uy1 //

	  // node 2 //
	  s_u[4].i = ei+1;s_u[4].j = ej+1;s_u[4].c = 0;      // Ux2 //
	  s_u[5].i = ei+1;s_u[5].j = ej+1;s_u[5].c = 1;      // Uy2 //

	  // node 3 //
	  s_u[6].i = ei;s_u[6].j = ej+1;s_u[6].c = 0;        // Ux3 //
	  s_u[7].i = ei;s_u[7].j = ej+1;s_u[7].c = 1;        // Uy3 //


// ==================================================================================================================assemble for all elements
       MatSetValuesStencil(DFDUcoup, 8, s_u, 8, s_u, dfintdu_array, ADD_VALUES);

PetscFree(dfintdu_array);
e=e+1;                                     // update the element number
ctr_x=ctr_x+1;

       }//end ei
ctr_y=ctr_y+1;
   }//end ej

MatAssemblyBegin(DFDUcoup, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(DFDUcoup, MAT_FINAL_ASSEMBLY);
DMDAVecRestoreArray(cdm,coords,&local_coords);


VecDestroy(&xp);
VecDestroy(&holesLocalVec);

PetscFunctionReturn(0);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//===================================================================================================================================================================================
void func_eir(MatrixXd &out_coeff_eir, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num,  all_const &values,MatrixXd& val)
{


int mark_den;
mark_den=values.mark_den;

MatrixXd val_er(3,3);
//val_er=MatrixXd::Zero(3,3);

MatrixXd alphar;
alphar=values.alphar;

MatrixXd out_coeff_er(3,3);
out_coeff_er=out_coeff_eir*alphar;

func_er(out_coeff_er,in_coeff_eir, in_coeff_eig, mark_num,values, val_er);
MatrixXd val_eir(3,3);
val_eir=MatrixXd::Zero(3,3);

if(mark_num>mark_den){
	MatrixXd out_coeff_new(3,3);
	//out_coeff_new=MatrixXd::Zero(3,3);

	out_coeff_new=out_coeff_eir*in_coeff_eir;
	func_eir(out_coeff_new,in_coeff_eir, in_coeff_eig,mark_num-1,values, val_eir);

}

val=val_er+val_eir;
}




//====================================================================================================================================================================================================

void func_eig(MatrixXd &out_coeff_eig, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num, all_const &values, MatrixXd& val)
{
MatrixXd C(3,3);
C=values.C_star;

int mark_den;
mark_den=values.mark_den;

MatrixXd out_coeff_new(3,3);
//out_coeff_new=MatrixXd::Zero(3,3);

out_coeff_new=out_coeff_eig*C;
//std::cout<< out_coeff_new <<std::endl;

//define prinmary output
MatrixXd val_1(3,3);
//val_1=MatrixXd::Zero(3,3);

func_eg(out_coeff_new, in_coeff_eir, in_coeff_eig,mark_num,values, val_1);

//define secondary output
MatrixXd val_out(3,3);
val_out=MatrixXd::Zero(3,3);
if(mark_num>mark_den){
	MatrixXd out_coeff_new_2(3,3);
	//out_coeff_new_2=MatrixXd::Zero(3,3);

	out_coeff_new_2=out_coeff_eig*in_coeff_eig;
	func_eig(out_coeff_new_2, in_coeff_eir, in_coeff_eig,mark_num-1,values, val_out);

}

val=val_out+val_1;

}

//====================================================================================================================================================================================================

void func_eg(MatrixXd &out_coeff_eg, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, all_const &values, MatrixXd& val)
{
MatrixXd F_bar(3,3);
F_bar=values.F_bar;

MatrixXd E_bar(3,3);
E_bar=values.E_bar;

MatrixXd O_bar(3,3);
O_bar=values.O_bar;

MatrixXd val_1(3,3);
val_1=MatrixXd::Zero(3,3);

MatrixXd val_2(3,3);
val_2=MatrixXd::Zero(3,3);

MatrixXd val_er(3,3);
val_er=MatrixXd::Zero(3,3);


int mark_den;
mark_den=values.mark_den;



if(mark_num>mark_den){
	MatrixXd out_coeff_1(3,3);
	//out_coeff_1=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_2(3,3);
	//out_coeff_2=MatrixXd::Zero(3,3);

	out_coeff_1=out_coeff_eg*F_bar;
	out_coeff_2=out_coeff_eg*E_bar;

	func_eig(out_coeff_1, in_coeff_eir, in_coeff_eig,mark_num-1,values, val_1);
	func_eir(out_coeff_2,in_coeff_eir, in_coeff_eig, mark_num-1,values, val_2);

}
MatrixXd out_coeff_3(3,3);
//out_coeff_3=MatrixXd::Zero(3,3);

out_coeff_3=out_coeff_eg*O_bar;
func_er(out_coeff_3, in_coeff_eir, in_coeff_eig,mark_num,values, val_er);

val=val_er+val_1+val_2;
}//func end

//====================================================================================================================================================================================


void func_ei(MatrixXd &out_coeff_ei, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, all_const &values, MatrixXd& val)
{
double in_coeff_ei=1;

MatrixXd M_bar(3,3);
M_bar=values.M_bar;

MatrixXd N_bar(3,3);
N_bar=values.N_bar;

MatrixXd val_er(3,3);
val_er=MatrixXd::Zero(3,3);


int mark_den;
mark_den=values.mark_den;


MatrixXd out_coeff_er(3,3);
out_coeff_er=out_coeff_ei*M_bar;

func_er(out_coeff_er, in_coeff_eir, in_coeff_eig,mark_num,values, val_er);

MatrixXd val_1(3,3);
val_1=MatrixXd::Zero(3,3);

MatrixXd val_2(3,3);
val_2=MatrixXd::Zero(3,3);

if(mark_num>mark_den){
	MatrixXd out_coeff_1(3,3);
	//out_coeff_1=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_2(3,3);
	//out_coeff_2=MatrixXd::Zero(3,3);

	out_coeff_1=out_coeff_ei*in_coeff_ei;
	out_coeff_2=out_coeff_ei*N_bar;

	func_ei(out_coeff_1, in_coeff_eir, in_coeff_eig,mark_num-1,values, val_1);
	func_eir(out_coeff_2, in_coeff_eir, in_coeff_eig, mark_num-1,values, val_2);

}

val=val_er+val_1+val_2;

}//func_end


//==================================================================================================================================================================================

void func_eis(MatrixXd &out_coeff_eis,MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig, int mark_num, all_const &values, MatrixXd& val)

{
val=MatrixXd::Zero(3,3);

int iter_heat;
iter_heat=values.iter_heat;

if(mark_num+1>=iter_heat){

       // std::cout<< mark_num<<std::endl;

      	MatrixXd I(3,3);
	I=MatrixXd::Identity(3,3);

	MatrixXd val_1(3,3);
	val_1=MatrixXd::Zero(3,3);
   
        VectorXd K;
        K=values.K1;
	
	MatrixXd in_coeff_eis(3,3);
	in_coeff_eis=K(mark_num)*I;

	MatrixXd out_coeff_1(3,3);
	//out_coeff_1=MatrixXd::Zero(3,3);

	out_coeff_1=out_coeff_eis*in_coeff_eis;
	func_eis(out_coeff_1,in_coeff_eir, in_coeff_eig, mark_num-1,values,val);
}


if(mark_num+1<iter_heat){
	MatrixXd val_1(3,3);
	MatrixXd in_coeff_eis(3,3);
	val_1=MatrixXd::Zero(3,3);

	MatrixXd I(3,3);
	I=MatrixXd::Identity(3,3);

	 in_coeff_eis=I;
	MatrixXd P_bar(3,3);
	//P_bar=MatrixXd::Zero(3,3);

	
	
        VectorXd delta_phi_g;
	delta_phi_g=values.delta_phi_g;

	int mark_den;
	mark_den=values.mark_den;        

        P_bar=delta_phi_g(mark_num)*I;

//if((mark_num==3) & (mark_den==3)) std::cout<<P_bar<<std::endl;


	MatrixXd val_er(3,3);
	val_er=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_2(3,3);
	//out_coeff_2=MatrixXd::Zero(3,3);

	out_coeff_2=out_coeff_eis*P_bar;
        func_er(out_coeff_2, in_coeff_eir, in_coeff_eig,mark_num,values, val_er);

	

	if(mark_num>mark_den){
	MatrixXd out_coeff_1(3,3);
	//out_coeff_1=MatrixXd::Zero(3,3);
	
	out_coeff_1=out_coeff_eis*in_coeff_eis;
	func_eis(out_coeff_1,in_coeff_eir, in_coeff_eig, mark_num-1,values, val_1);
	}
	val=val_er+val_1;
        //std::cout<<"****mark_num****"<<mark_num<<"*****"<<std::endl;
       //std::cout<<"****val_Er ****"<<val_er<<"*****"<<std::endl;	
}

}// func end

//===========================================================================================================================================================================================

void func_er(MatrixXd &coeff_er, MatrixXd &in_coeff_eir,MatrixXd &in_coeff_eig,int mark_num, all_const &values, MatrixXd& val)
{
MatrixXd A(3,3);
//A=MatrixXd::Zero(3,3);

MatrixXd B(3,3);
//B=MatrixXd::Zero(3,3);

MatrixXd D_mark_num(3,3);

std::vector <Eigen::MatrixXd> D;
D=values.D_iter;


int mark_den;
mark_den=values.mark_den;

VectorXd phi_g_total;
phi_g_total=values.phi_g_total;



int iter_heat;
iter_heat=values.iter_heat;

VectorXd K;
K=values.K1;

MatrixXd A_g_inv_B_r(3,3);
MatrixXd A_g_inv_B_g(3,3);
MatrixXd diB_r(3,3);


A_g_inv_B_r=values.A_g_inv_B_r;
diB_r=values.diB_r;
A_g_inv_B_g=values.A_g_inv_B_g;

D_mark_num=D[mark_num];
MatrixXd D_mark_num_inv(3,3);

double d_00=D_mark_num(0,0);
double d_01=D_mark_num(0,1);
double d_02=D_mark_num(0,2);
double d_10=D_mark_num(1,0);
double d_11=D_mark_num(1,1);
double d_12=D_mark_num(1,2);
double d_20=D_mark_num(2,0);
double d_21=D_mark_num(2,1);
double d_22=D_mark_num(2,2);

double m_00=d_11*d_22-d_12*d_21;
double m_01=d_10*d_22-d_12*d_20;
double m_02=d_10*d_21-d_11*d_20;
double m_10=d_01*d_22-d_02*d_21;
double m_11=d_00*d_22-d_02*d_20;
double m_12=d_00*d_21-d_01*d_20;
double m_20=d_01*d_12-d_02*d_11;
double m_21=d_00*d_12-d_02*d_10;
double m_22=d_00*d_11-d_01*d_10;

MatrixXd Coff(3,3);
Coff<< m_00, -m_10, m_20,
       -m_01, m_11, -m_21,
       m_02, -m_12, m_22;

double deter=d_00*(d_11*d_22-d_12*d_21)-d_01*(d_10*d_22-d_12*d_20)+d_02*(d_10*d_21-d_11*d_20);

D_mark_num_inv=Coff/deter;

//D_mark_num_inv=D_mark_num.inverse();
A=D_mark_num_inv*(-phi_g_total(mark_num+1)*(A_g_inv_B_r)+diB_r);
B=D_mark_num_inv*(phi_g_total(mark_num+1)*(A_g_inv_B_g));


MatrixXd coeff_ei(3,3);
//coeff_ei=MatrixXd::Zero(3,3);

MatrixXd coeff_eis(3,3);
//coeff_eis=MatrixXd::Zero(3,3);

coeff_ei=-D_mark_num_inv;



coeff_eis=coeff_ei;


if(mark_num+1>=iter_heat){coeff_eis=K(mark_num)*coeff_ei;}

val=MatrixXd::Zero(3,3);

//if(mark_num<mark_den){std::cout<<val<<std::endl;}

if(mark_num==mark_den){val=coeff_er;}

if(mark_num>mark_den){
	MatrixXd out_coeff_1(3,3);
	//out_coeff_1=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_2(3,3);
	//out_coeff_2=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_3(3,3);
	//out_coeff_3=MatrixXd::Zero(3,3);

	MatrixXd out_coeff_4(3,3);
	//out_coeff_4=MatrixXd::Zero(3,3);

	MatrixXd val_1(3,3);
	//val_1=MatrixXd::Zero(3,3);

	MatrixXd val_2(3,3);
	//val_2=MatrixXd::Zero(3,3);

	MatrixXd val_3(3,3);
	//val_3=MatrixXd::Zero(3,3);

	MatrixXd val_4(3,3);
	//val_4=MatrixXd::Zero(3,3);

	out_coeff_1=coeff_er*A;
	out_coeff_2=coeff_er*B;
	out_coeff_3=coeff_er*coeff_ei;
	out_coeff_4=coeff_er*coeff_eis;


        func_eir(out_coeff_1, in_coeff_eir, in_coeff_eig, mark_num-1,values, val_1);
	func_eig(out_coeff_2, in_coeff_eir, in_coeff_eig, mark_num-1,values, val_2);
	func_ei(out_coeff_3,  in_coeff_eir, in_coeff_eig, mark_num-1,values, val_3);
	func_eis(out_coeff_4, in_coeff_eir, in_coeff_eig, mark_num-1,values, val_4);


	val=val_1+val_2+val_3+val_4;

}

}//end of func











