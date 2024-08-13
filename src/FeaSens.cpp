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
#include <iomanip>
using namespace Eigen;

using Eigen::MatrixXd;
using Eigen::VectorXd; 
using Eigen::IOFormat;
typedef Eigen::Map<MatrixXd,0,Eigen::Stride<1,2> > MatMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, Dynamic, RowMajor> > matMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, 1> > vecMap;
#define ngp 2
#define ndof 2
#define ndim 2
#define Tmax 350
//#define Tmin 350
#define Tmin 330
#define Tg 340
#define plane_strain 1
#define plane_stress 0

#define GAUSS_POINTS   4
#define NSD   2

 

PetscErrorCode physics( Vec &xPhys, Vec &hole_elem, PetscInt nelx, PetscInt nelh_x, PetscInt nely, PetscInt nelh_y, PetscScalar time_spec, PetscInt a, PetscScalar &Obj_val, Vec &dfdrho, DM &elas_da, DM &prop_da, PetscInt optit) 
{

  PetscMPIInt      rank,size;
  PetscInt         nel, stw, nel_p, nen, i, mxl, myl, cpu_x, cpu_y, *lx=NULL, *ly=NULL, prop_dof, prop_stencil_width, M, N, gxs, gys,gxm, gym , neq, nnodes, e, iter, interm_size;
  PetscInt         nnodesx,nnodesy, sex, sey,lower_node_x, lower_node_y,size_x,size_y,ein,ltogmsize,len_f, size_rho,size_test=16,size_dfdx,len_p,set_dof, set_dof_bottom, *set_dof_interm=NULL, set_dof_top;
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE;
  //DM               elas_da,prop_da;
  DM               cdm,vel_cda;
  DMDACoor2d       **local_coords;
  Vec              local,global;
  PetscScalar      value, one=1.0,penal=1.0;
  const PetscInt   *el_indices=NULL,*g_idx=NULL,*idx=NULL,*local_indices=NULL;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  PetscReal         dx, dy;
  Vec               coords,U,Fint,X,local_F,vec_indices,Fext_petsc,Residual_f,delta_U_f,U_f,Fext_f,Fext,Fint_f, Residual;
  Mat               KTAN, KFF,*KFF_dup_pt=NULL, KFF_dup, dFint_du, KTAN_dup;
  IS                is;
  MatNullSpace      matnull;
  ElasticityDOF     **ff;
  KSP                ksp_E;
  ISLocalToGlobalMapping ltogm;
  PetscReal      norm,norm_0, *disp_val=NULL;
  //Vec             U_hist[10];
   Vec             *U_hist=NULL, dFint_dx,dfint_dx_dup;

// Define the variables
 
 nel=nelx*nely;
 neq=(nelx+1)*(nely+1)*ndof;
 iter=0;

 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  VecScatter     scat;


// Define the Vectors and matrices needed 
DMCreateMatrix(elas_da, &KTAN);
ierr = DMGetCoordinates(elas_da,&U);CHKERRQ(ierr);
ierr = MatNullSpaceCreateRigidBody(U,&matnull);CHKERRQ(ierr);
ierr = MatSetNearNullSpace(KTAN,matnull);CHKERRQ(ierr);
ierr = MatNullSpaceDestroy(&matnull);CHKERRQ(ierr);
MatCreateVecs(KTAN, &Fint,&Fext);
MatCreateVecs(KTAN, NULL,&U);



MatZeroEntries(KTAN);
VecZeroEntries(Fint);
VecZeroEntries(Fext);
VecZeroEntries(U);



//        create KSP solver
//KSPCreate(PETSC_COMM_WORLD,&ksp_E);
//KSPSetType(ksp_E, KSPCG);
//KSPSetTolerances(ksp_E,1.e-10,1.e-50,1.0e5,200);
//KSPSetOptionsPrefix(ksp_E,"elas_");  

// enforce bc's
DMGetLocalToGlobalMapping(elas_da,&ltogm);
ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);
ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);


// SMP cycle properties

PetscReal C_R=-1;
PetscScalar delta_t=5;
PetscScalar t=0;   
PetscScalar t_loading_start=t;
PetscScalar t_loading_fin=t_loading_start+((Tmax-Tmin)/PetscAbsReal(C_R));
//PetscPrintf(PETSC_COMM_WORLD,"t_loading_fin %f \n",t_loading_fin);
//PetscScalar t_loading_fin=15;
PetscScalar T=Tmax;
PetscScalar *T_hist=NULL;
Mat *KT_step=NULL, *KTAN_step=NULL;
Vec *dFdx_step=NULL;

PetscInt tot_its=time_spec/delta_t; // total number of iterations


// define and declare history variables
PetscMalloc1(tot_its+1, &U_hist);
for(i=0;i<tot_its+1;i++){
MatCreateVecs(KTAN, NULL,&U_hist[i]);
}



PetscMalloc1(tot_its+1, &T_hist);
PetscMalloc1(tot_its, &KT_step); // define the number of storage cells needed for KT
PetscMalloc1(tot_its, &dFdx_step); // define the number of storage cells needed for dfint_dx
PetscMalloc1(tot_its, &KTAN_step); // define the number of storage cells needed for KTAN_step

//PetscPrintf(PETSC_COMM_WORLD, "tot_its[%d] \n", tot_its);

// Create dFint_dx(nel,8) matrix
VecCreate(PETSC_COMM_WORLD, &dFint_dx);
VecSetType(dFint_dx, VECSTANDARD);

DMDAGetElements(elas_da,&nel_p,&nen,&el_indices);
//printf( "rank [%d] nel_p[%d] \n", rank, nel_p); 

VecSetSizes(dFint_dx, nel_p*8, PETSC_DECIDE);
VecGetSize(dFint_dx, &size_dfdx);
//VecView(dFint_dx, PETSC_VIEWER_STDOUT_WORLD);
//PetscPrintf(PETSC_COMM_WORLD, "rank [%d] size_dfdx[%d] \n", rank, size_dfdx); 

VecSetBlockSize(dFint_dx, 8);
ISLocalToGlobalMapping l2g;
PetscInt vStart, vEnd, *gInd=NULL;
VecGetOwnershipRange(dFint_dx, &vStart, &vEnd);

//assign global indices for blocks
PetscMalloc1(nel_p, &gInd);
for(int i=0;i<nel_p;i++){
gInd[i]=(vStart/8)+i;
//printf("rank [%d] gInd [%d] \n",rank, gInd[i]);
}
ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,8, nel_p, gInd, PETSC_COPY_VALUES, &l2g);
 //ISLocalToGlobalMappingView(l2g,PETSC_VIEWER_STDOUT_WORLD);            // view mapping
PetscFree(gInd);
VecSetLocalToGlobalMapping(dFint_dx, l2g);
ISLocalToGlobalMappingDestroy(&l2g);

T_hist[0]=T;
VecCopy(U,U_hist[0]);


// Define residual vector
MatCreateVecs(KTAN,NULL,&Residual);

//=================================================== BC conditions and Loading conditions ======================================================================
//PetscScalar Force_amt = 0.01;
//FFext(elas_da, Fext, nelh_x, nelh_y, rank, nelx, nely, Force_amt);
//VecView(Fext, PETSC_VIEWER_STDOUT_WORLD);











//===================================================================================================================================================================//

PetscInt des_dof_local=0;


//111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
//******************************************************* N-R iterations *******************************************************
// Cooling with deformations

if(time_spec>t_loading_start)
{
   while(t+delta_t<=PetscMin(t_loading_fin,time_spec))
   {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time


T=T+C_R*delta_t;                                // update temperature

// Evaluate KTAN & Fint 
assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function



// Find Fext
PetscScalar Force_amt = 0.4;
FFext(elas_da, Fext, nelh_x, nelh_y, rank, nelx, nely, Force_amt, des_dof_local);


// Evaluate Residual vector
VecWAXPY(Residual,-1,Fint,Fext);

//calculate norm value
ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs
VecNorm(Residual_f,NORM_2,&norm);


// Evaluate U_f
MatDestroy(&KFF);
ISDestroy(&is);
ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs

// Create KSP
KSPCreate(PETSC_COMM_WORLD,&ksp_E);
KSPSetTolerances(ksp_E,1.e-10,1.e-50,1.0e5,200);
KSPSetOptionsPrefix(ksp_E,"elas_");

		
PC pc;

while (norm>1e-6)
{

		// Define delta_U_f
		VecDuplicate(Residual_f, &delta_U_f);

		// Solve for delta_U_f
		KSPSetOperators(ksp_E,KFF,KFF);
		KSPGetPC(ksp_E, &pc);
		PCSetType(pc, PCLU);
		PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);
		VecZeroEntries(Fint);
		KSPSolve(ksp_E,Residual_f,delta_U_f);



		// Update U_f
                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

		// Update U
                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);
                //calculate KTAN and Fint
           
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function

		VecDestroy(&U_f);
		MatDestroy(&KFF);
		ISDestroy(&is);

                // apply BC FORM KFF AND Fintf
                ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs
		

              
               // Calculate Residual
		VecWAXPY(Residual,-1,Fint,Fext);
		VecDestroy(&Residual_f);
		MatDestroy(&KFF);
		ISDestroy(&is);
		ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs


		// Evaluate Residual norm
                VecNorm(Residual_f,NORM_2,&norm);
            

PetscPrintf(PETSC_COMM_WORLD,"||Norm of error %g||\n",(double)norm);
VecDestroy(&delta_U_f);

}

// Destroy KSP
KSPDestroy(&ksp_E);

// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function

// Update U_hist
VecCopy(U,U_hist[iter]);
T_hist[iter]=T; 

//if(iter==4) VecView(U, PETSC_VIEWER_STDOUT_WORLD);

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"C+D iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
ISDestroy(&is);

 }
}



//2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
PetscScalar t_relax_start=t_loading_fin;
PetscScalar relax_duration=15;
PetscScalar t_relax_fin=t_relax_start+relax_duration;

// Set Fext vector
PetscScalar Force_amt = 0.0;
FFext(elas_da, Fext, nelh_x, nelh_y, rank, nelx, nely, Force_amt, des_dof_local);



// Relaxation cycle
if(time_spec>t_relax_start)
{
   while(t+delta_t<=PetscMin(t_relax_fin,time_spec))
    {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time



T=T;                                            // update temperature


// Evaluate KTAN & Fint
assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function


// Evaluate Residual

VecWAXPY(Residual,-1,Fint,Fext);

// Evaluate Residual norm

ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs
VecNorm(Residual_f,NORM_2,&norm);
 

// Evaluate U_f
MatDestroy(&KFF);
ISDestroy(&is);
ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs


// Set-up KSP
KSPCreate(PETSC_COMM_WORLD,&ksp_E);
KSPSetTolerances(ksp_E,1.e-10,1.e-50,1.0e5,200);
KSPSetOptionsPrefix(ksp_E,"elas_");

PC pc;

while (norm>1e-6)
{

		// form delta_U_f
		VecDuplicate(Residual_f, &delta_U_f);

		// Solve for delta_U_f
		KSPSetOperators(ksp_E,KFF,KFF);
		KSPGetPC(ksp_E, &pc);
		PCSetType(pc, PCLU);
		PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);


		VecZeroEntries(Fint);
		
		KSPSolve(ksp_E,Residual_f,delta_U_f);

		// update U_f
                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

		// Update U
                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);

                //calculate KTAN and Fint
            
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys,hole_elem,  penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function
		
		
		// Calculate U_f
		VecDestroy(&U_f);
		MatDestroy(&KFF);
		ISDestroy(&is);
		ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs
              
               //Calculate residual

		VecWAXPY(Residual,-1,Fint,Fext);
		


		// Evaluate REsidual norm
		VecDestroy(&Residual_f);
		MatDestroy(&KFF);
		ISDestroy(&is);
		ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs
                VecNorm(Residual_f,NORM_2,&norm);

		
             
PetscPrintf(PETSC_COMM_WORLD,"||Norm of error %g||\n",(double)norm);
VecDestroy(&delta_U_f);
}

KSPDestroy(&ksp_E);

// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function


//Update Histroy variables
VecCopy(U,U_hist[iter]);                       // initial value of u history

T_hist[iter]=T;                               // store history values

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"R iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
ISDestroy(&is);
  }
}



//33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
// Heating Phase
PetscInt iter_heat=iter+1;

//PetscReal H_R=0.25;
PetscReal H_R=1;
PetscScalar t_heat_start=t_relax_fin;
PetscScalar t_heat_fin=t_heat_start+((Tmax-Tmin)/PetscAbsReal(H_R));
//PetscPrintf(PETSC_COMM_WORLD,"t_heat_fin %f \n",t_heat_fin);
if(time_spec>t_heat_start)
{
   while(t+delta_t<=PetscMin(t_heat_fin,time_spec))
    {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time


T=T+H_R*delta_t;                                // update temperature



// Calculate KTAN and Fint
assembly(elas_da, prop_da, rank, iter, delta_t, xPhys,hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function


//calculate residual
VecWAXPY(Residual,-1,Fint,Fext);


//calculate norm value
ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs
VecNorm(Residual_f,NORM_2,&norm);


 // Form U_f
MatDestroy(&KFF);
ISDestroy(&is);
ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs

//Set-up KSP
KSPCreate(PETSC_COMM_WORLD,&ksp_E);
KSPSetTolerances(ksp_E,1.e-10,1.e-50,1.0e5,200);
KSPSetOptionsPrefix(ksp_E,"elas_");

PC pc;

while (norm>1e-6)
{
		
		// Form delta_U_f
		VecDuplicate(Residual_f, &delta_U_f);

		// solve for delta_U_f
		KSPSetOperators(ksp_E,KFF,KFF);
		KSPGetPC(ksp_E, &pc);
		PCSetType(pc, PCLU);
		PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);
		VecZeroEntries(Fint);

		
		KSPSolve(ksp_E,Residual_f,delta_U_f);


		// Update U_f
                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

		// Update U
                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);
                //calculate KTAN and Fint
            
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys,hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function

		
		
		// form U_f
		VecDestroy(&U_f);
		MatDestroy(&KFF);
		ISDestroy(&is);
		ApplyBC(elas_da,rank, KTAN, U, &is,&KFF, &U_f); //is gives the free d.o.fs
              
               //Calculate residual
                VecWAXPY(Residual,-1,Fint,Fext);


		// Evaluate Residual norm
		VecDestroy(&Residual_f);
		MatDestroy(&KFF);
		ISDestroy(&is);
		ApplyBC(elas_da,rank, KTAN, Residual, &is,&KFF, &Residual_f); //is gives the free d.o.fs
                VecNorm(Residual_f,NORM_2,&norm);
              
PetscPrintf(PETSC_COMM_WORLD,"||Norm of error %g||\n",(double)norm);
VecDestroy(&delta_U_f);
}
KSPDestroy(&ksp_E);
// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, hole_elem, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function


//Store history variables
VecCopy(U,U_hist[iter]);                       // initial value of u history
T_hist[iter]=T;                               // store history values

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"H iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
ISDestroy(&is);

  }
}

//VecView(dFdx_step[iter-1], PETSC_VIEWER_STDOUT_WORLD);
//VecView(U, PETSC_VIEWER_STDOUT_WORLD);
PetscPrintf(PETSC_COMM_WORLD,"%%%%%%%%  FEA COMPLETED %%%%%%%%%%\n");
//===================================================================================================================================================================
ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs


// Get the d.o.f where the objective is evaluated
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PetscInt des_dof= 0;
MPI_Barrier(PETSC_COMM_WORLD);
//printf("rank [%d] des_dof_local[%d] \n",rank,  des_dof_local);
MPI_Reduce(&des_dof_local, &des_dof,1, MPIU_INT,MPI_SUM, 0, PETSC_COMM_WORLD);
PetscPrintf(PETSC_COMM_WORLD,"des_dof[%d] \n",des_dof);

//Bcast the desired dof value to all processors
MPI_Barrier(PETSC_COMM_WORLD);
MPI_Bcast(&des_dof, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
//printf("rank [%d] des_dof[%d] \n",rank,  des_dof);


PetscInt low_proc, high_proc;
PetscScalar leaf_val=0.0;
VecGetOwnershipRange(U, &low_proc,&high_proc); 
//printf("rank [%d]  low_prc[%d]  high_proc[%d]  \n", rank, low_proc, high_proc);
PetscInt get_from =des_dof;
if((get_from>=low_proc)&&(get_from<high_proc)){VecGetValues(U, 1, &get_from, &leaf_val);}


// Get objective from all processors
//printf("obj_val[%f] \n", leaf_val);
MPI_Barrier(PETSC_COMM_WORLD);
MPI_Reduce(&leaf_val, &Obj_val,1, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD); 
 
PetscPrintf(PETSC_COMM_WORLD,"obj_val %f\n",Obj_val);

//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


//======= clean up history begin====
// Free U_hist variable
for(i=0;i<tot_its+1;i++){
VecDestroy(&U_hist[i]);
}
//======= clean up history end======



PetscInt iscount=0;
PetscInt issize_local;
const PetscInt *isindices;
ISGetLocalSize(is, &issize_local);
ISGetIndices(is, &isindices);
for(PetscInt i=0;i<issize_local;i++) { if(isindices[i]<des_dof) iscount++;} 
//printf("rank[%d] cnt[%d] \n",rank, iscount);

ISRestoreIndices(is, &isindices);

PetscInt n_free_dof_less_than_des_dof=0;
MPI_Barrier(PETSC_COMM_WORLD);
MPI_Reduce(&iscount, &n_free_dof_less_than_des_dof,1, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD);

MPI_Barrier(PETSC_COMM_WORLD);
MPI_Bcast(&n_free_dof_less_than_des_dof, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

//printf("rank[%d] n_free_dof_less_than_des_Dof[%d] \n",rank, n_free_dof_less_than_des_dof);



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//										SENSITIVITY ANALYSIS 
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//DECLARE DFDRHO
PC pc;

//ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs



//Vec dfdrho;
DMCreateGlobalVector(prop_da, &dfdrho); 
VecSet(dfdrho, 0.0);

Vec lambda_test;
VecDuplicate(Fint_f, &lambda_test);
Vec L_test;
VecDuplicate(Fint_f, &L_test);
VecSet(L_test, 0.0);
PetscInt add_at_index_test = n_free_dof_less_than_des_dof;
VecSetValues(L_test, 1, &add_at_index_test, &one, INSERT_VALUES);
VecAssemblyBegin(L_test);
VecAssemblyEnd(L_test);

KSP ksp_l;
KSPCreate(PETSC_COMM_WORLD,&ksp_l);

KSPSetOptionsPrefix(ksp_l,"lambda_"); 
//KSPSetType(ksp_l, KSPCG); 
KSPSetTolerances(ksp_l,1.e-10,1.e-50,1.0e5,200);
KSPGetPC(ksp_l, &pc);
PCSetType(pc, PCLU);
PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);
KSPSetOperators(ksp_l,KFF,KFF);
KSPConvergedReason converged;
KSPSolve(ksp_l,L_test,lambda_test);
//KSPReasonView(ksp_l, PETSC_VIEWER_STDOUT_WORLD);
KSPGetConvergedReason(ksp_l, &converged);
if(converged <0){SETERRQ(PETSC_COMM_WORLD,1,"KSP sens solver did not converge\n");}

VecScale(lambda_test,-1.0);




//VecView(lambda_test, PETSC_VIEWER_STDOUT_WORLD);

//PetscViewerASCIIOpen( PETSC_COMM_WORLD, "lambda.m",&viewer);
//PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//VecView(lambda, viewer);
//PetscViewerPopFormat(viewer);


// Create a Matrix of lambda
Vec *lambda_iter,*lambda_steps;
PetscMalloc1(iter, &lambda_iter);
for(PetscInt i=0;i<iter;i++){
MatCreateVecs(KFF,NULL,&lambda_iter[i]);
}


PetscMalloc1(iter, &lambda_steps);

//Store the lambda of last step
Vec lambda_m_f;

VecDuplicate(Fint_f, &lambda_m_f);
VecCopy(lambda_test, lambda_m_f);

//lambda_iter[iter-1]=lambda_m_f;
VecCopy(lambda_m_f, lambda_iter[iter-1]);


//VecView(lambda_iter[iter-1], PETSC_VIEWER_STDOUT_WORLD);
//Mat KFF_dense;
//MatConvert(KFF, MATDENSE, MAT_INITIAL_MATRIX, &KFF_dense);
// PetscViewer    viewer;
//PetscViewerASCIIOpen( PETSC_COMM_WORLD, "KFFdense.mat",&viewer);
//PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//MatView(KFF_dense, viewer);
//PetscViewerPopFormat(viewer);


//******************************************************** lambda_M calculated *************************************************************************************************************



// Create RHS vector
Vec RHS; 
VecDuplicate(lambda_m_f, &RHS);
VecCopy(lambda_m_f, RHS);


Vec lambda_b;
VecDuplicate(Fint_f, &lambda_b);
VecCopy(Fint_f, lambda_b);
VecSet(lambda_b, 0.0);


Mat DFDUcoup;
// Create coupling Matrix
DMCreateMatrix(elas_da, &DFDUcoup);
//PetscInt step_num=3;
//PetscInt step_den=0;
//PetscInt iter_heat=10;


KSP ksp_lambda;
KSPCreate(PETSC_COMM_WORLD,&ksp_lambda);
KSPSetOptionsPrefix(ksp_lambda,"lambda_iter"); 
KSPSetTolerances(ksp_lambda,1.e-10,1.e-50,1.0e5,200);
KSPGetPC(ksp_lambda, &pc);
PCSetType(pc, PCLU);
PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);
KSPConvergedReason converged_lam;





Vec Y;
VecDuplicate(lambda_m_f, &Y);
VecCopy(lambda_m_f, Y);
VecSet(Y, 0.0);

Mat KTAN_T;
MatDuplicate(KFF, MAT_DO_NOT_COPY_VALUES, &KTAN_T);

PetscInt Cstart, Cend;


Mat DFDUcoup_f;
Vec lambda_f;
//MatDuplicate(KFF,MAT_DO_NOT_COPY_VALUES, &DFDUcoup_f);
//MatZeroEntries(DFDUcoup_f);
//VecDuplicate(Fint_f, &lambda_f);

//MatView(KT_step[0], PETSC_VIEWER_STDOUT_WORLD);
Vec lambda_copy;


//Mat DFDUcoup_dense;

//df_du(elas_da, prop_da, rank, nelx,nely, delta_t,penal, xPhys, 8, 2, T, T_hist, iter, iter_heat, DFDUcoup);

 //ApplyBC(elas_da,rank, DFDUcoup, Fint, &is,&DFDUcoup_f, &Fint_f); //is gives the free d.o.fs

//MatConvert(DFDUcoup_f, MATDENSE, MAT_INITIAL_MATRIX, &DFDUcoup_dense);
// PetscViewer    viewer;
//PetscViewerASCIIOpen( PETSC_COMM_WORLD, "dFDUcoup21.mat",&viewer);
//PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//MatView(DFDUcoup_dense, viewer);
//PetscViewerPopFormat(viewer);

 //PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer);
//PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_DENSE);
// MatView(DFDUcoup,viewer);

for(PetscInt i=iter-2;i>-1;i=i-1)
{
VecSet(RHS, 0.0);

VecSet(Y, 0.0);
VecSet(lambda_b,0.0);

//std::cout<< "i "<<i  <<std::endl;

	for(PetscInt k=i+1;k<iter;k++){
		MatZeroEntries(DFDUcoup);
                
		//PetscPrintf(PETSC_COMM_WORLD, "k[%d] i[%d] iter[%d]\n", k, i,iter);
           	df_du(elas_da, prop_da, rank, nelx,nely, delta_t,penal, xPhys,hole_elem, k, i, T, T_hist, iter, iter_heat, DFDUcoup);
		ISDestroy(&is);
		
		 VecDestroy(&Fint_f);
                 ApplyBC(elas_da,rank, DFDUcoup, Fint, &is,&DFDUcoup_f, &Fint_f); //is gives the free d.o.fs
		

		//VecView(lambda_iter[k], PETSC_VIEWER_STDOUT_WORLD);
		//VecView(lambda_iter[k], PETSC_VIEWER_STDOUT_WORLD);
       		MatMult(DFDUcoup_f, lambda_iter[k], Y);
          
		VecAXPY(RHS, 1.0, Y);
 		MatDestroy(&DFDUcoup_f);
	       
	}// k loop


// if(i==3)VecView(RHS, PETSC_VIEWER_STDOUT_WORLD);
//VecView(RHS, PETSC_VIEWER_STDOUT_WORLD);

//MatCopy(KT_step[i], KTAN_T, SAME_NONZERO_PATTERN);
////MatCreateTranspose(KTAN_step[i], &KTAN_T); 

VecDuplicate(lambda_b, &lambda_copy);

KSPSetOperators(ksp_lambda,KT_step[i],KT_step[i]);

KSPSolve(ksp_lambda,RHS,lambda_b);
//KSPReasonView(ksp_lambda, PETSC_VIEWER_STDOUT_WORLD);
KSPGetConvergedReason(ksp_lambda, &converged_lam);

VecScale(lambda_b, -1.0);

if(converged_lam <0){SETERRQ(PETSC_COMM_WORLD,1,"KSP lambda solver did not converge\n");}

//VecView(lambda_b, PETSC_VIEWER_STDOUT_WORLD);
VecCopy(lambda_b,lambda_iter[i]);	
 //lambda_iter[i]=lambda_b;
VecCopy(lambda_b, lambda_copy);
lambda_steps[i]=lambda_copy;

//VecView(lambda_iter[i], PETSC_VIEWER_STDOUT_WORLD);


}// i loop

lambda_steps[iter-1]=lambda_m_f;
//VecView(lambda_iter[0], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[1], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[2], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[3], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[4], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[5], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[6], PETSC_VIEWER_STDOUT_WORLD);
//VecView(lambda_iter[7], PETSC_VIEWER_STDOUT_WORLD);
//******************************************** lambda calculated for all the steps *******************************************************



// CALCULATE DFDRHO
Vec *psi;
PetscCalloc1(iter, &psi);
for (PetscInt i=0;i<iter;i++){
MatCreateVecs(KTAN, NULL, &psi[i]);
}

Vec lambda_final_copy;

Vec I;
VecDuplicate(Fint, &I);
VecCopy(Fint,I);
VecSet(I, 0.0);

Vec lambda_final;
VecDuplicate(Fint, &lambda_final);
VecCopy(Fint, lambda_final);


Vec lambda_i;
VecDuplicate(lambda_m_f, &lambda_i);
VecCopy(lambda_m_f,lambda_i);

PetscScalar *lambda_local=NULL;


VecSet(lambda_final,0.0);

VecSet(lambda_i,0.0);


//VecCopy(lambda_steps[0],lambda_i);
//VecView(lambda_i, PETSC_VIEWER_STDOUT_WORLD);
//VecDestroy(&lambda_i);
const PetscInt *freedofs=NULL;
PetscInt free_p_size; 
ISGetIndices(is, &freedofs);   // global freedofs
ISGetLocalSize(is, &free_p_size);
//ISGetIndices(is, &freedofs);

for(PetscInt i=0;i<iter;i++){


//VecDuplicate(lambda_final, &lambda_final_copy);

//lambda_i=lambda_steps[i];
VecCopy(lambda_iter[i], lambda_i);

VecGetArray(lambda_i, &lambda_local);



	for(int j=0;j<free_p_size;j++){
	PetscInt put_idx=freedofs[j];
	PetscScalar in_val=lambda_local[j];
        //if(rank==0) printf("j [%d] put idx [%d] \n",j, put_idx);
	VecSetValues(lambda_final, 1, &put_idx,&in_val, INSERT_VALUES); 


	}

VecAssemblyBegin(lambda_final);
VecAssemblyEnd(lambda_final);

//VecCopy(lambda_final, lambda_final_copy);
VecCopy(lambda_final, psi[i]);
//psi[i]=lambda_final_copy;
VecRestoreArray(lambda_i, &lambda_local);


}
ISRestoreIndices(is, &freedofs);
VecDestroy(&lambda_i);

//VecView(psi[0], PETSC_VIEWER_STDOUT_WORLD);
//VecView(psi[1], PETSC_VIEWER_STDOUT_WORLD);
//VecView(psi[2], PETSC_VIEWER_STDOUT_WORLD);
//VecView(psi[4], PETSC_VIEWER_STDOUT_WORLD);



// SOLVE FOR FINAL VALUE
Vec psi_n;
VecDuplicate(lambda_final,&psi_n);
//VecCopy(lambda_final, psi_n);
//VecSet(psi_n, 0.0);


Vec dRdx;
VecDuplicate(dFint_dx, &dRdx);
//VecCopy(dFint_dx, dRdx);
//VecSet(dRdx, 0.0);

for(int i=0;i<iter;i++){

//psi_n=psi[i];

VecCopy(psi[i], psi_n);

//dRdx=dFdx_step[i];
VecCopy(dFdx_step[i], dRdx);
//VecView(psi_n, PETSC_VIEWER_STDOUT_WORLD);

//VecView(dRdx, PETSC_VIEWER_STDOUT_WORLD);

PetscInt time_step=i;

dfdrho_func(elas_da, prop_da,rank, time_step, dfdrho, psi_n, dRdx);
//VecView(dfdrho, PETSC_VIEWER_STDOUT_WORLD);


}

//df_du(elas_da, prop_da, nelx,nely, delta_t,penal, xPhys, step_num, step_den, T, T_hist, iter, iter_heat, DFDUcoup);

PetscPrintf(PETSC_COMM_WORLD," %%%%%%%%%%%%%%%%%% SENSITIVITIES ANALYSIS COMPLETED %%%%%%%%%%%%%%%%%%%%%%% \n");  
//VecView(dfdrho, PETSC_VIEWER_STDOUT_WORLD);
//PetscViewerASCIIOpen( PETSC_COMM_WORLD, "dfdrho.m",&viewer);
//PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//VecView(dfdrho, viewer);
//PetscViewerPopFormat(viewer);

// **************************** free memory ****************************************************


 
	
	
	

//FEA=====================================

        MatDestroy(&KTAN);
        VecDestroy(&U);
	VecDestroy(&Fext);
	VecDestroy(&Fint);

        
	PetscFree(g_idx);

       PetscFree(U_hist);

for(i=0;i<iter;i++){
VecDestroy(&dFdx_step[i]);
MatDestroy(&KT_step[i]);
MatDestroy(&KTAN_step[i]);
}

        PetscFree(KT_step);
        PetscFree(KTAN_step);
        PetscFree(dFdx_step);
        PetscFree(T_hist);

        VecDestroy(&dFint_dx);

	VecDestroy(&Residual);

	MatDestroy(&KFF);
       VecDestroy(&Fint_f);
       
 
	ISLocalToGlobalMappingDestroy(&l2g);
        //KSPDestroy(&ksp_E);

//==========================================  
    	
	//#VecDestroy(&dfdrho);  // destroyed in main.cpp

	VecDestroy(&L_test);
	VecDestroy(&lambda_test);
	
	KSPDestroy(&ksp_l);

for(PetscInt i=0;i<iter;i++){
VecDestroy(&lambda_iter[i]);
}

	PetscFree(lambda_iter);

for(PetscInt i=iter-2;i>-1;i=i-1)
{
VecDestroy(&lambda_steps[i]);
}

	PetscFree(lambda_steps);
	VecDestroy(&lambda_m_f);
	VecDestroy(&RHS);
	VecDestroy(&lambda_b);
	MatDestroy(&DFDUcoup);
	KSPDestroy(&ksp_lambda);
	VecDestroy(&Y);
	MatDestroy(&KTAN_T);
	
	VecDestroy(&I);

	VecDestroy(&lambda_final);

	VecDestroy(&lambda_i);

for(PetscInt i=0;i<iter;i++){
	VecDestroy(&psi[i]);
}

VecDestroy(&psi_n);
PetscFree(psi);

VecDestroy(&dRdx);


	ISDestroy(&is);

	


 PetscFunctionReturn(0);

}

//===========================================================================================================================================================================
 PetscErrorCode assembly(DM elas_da, DM prop_da, PetscMPIInt rank,PetscInt iter, PetscScalar delta_t, Vec& xPhys, Vec &hole_elem, PetscScalar penal, PetscScalar T, PetscScalar *T_hist, Vec& Fint, Vec& dFint_dx, Mat& KTAN, Vec& U, Vec* U_hist, PetscInt flag)
{

PetscInt         si, sj, si_wg, sj_wg, si_g, sj_g, ei, ej, nx, ny,e=0,size_x, sizeg_x, sizeg_y, size_y,ctr_x,ctr_y,xp_size, *blk;
DM               cdm, u_cda;
DMDACoor2d       **local_coords;
Vec              coords,local_F,u_local, *u_local_hist=NULL,xp, holesLocalVec;
ElasticityDOF    **ff;
MatStencil        s_u[8];
PetscErrorCode   ierr;
PetscScalar     *u_local_array=NULL, *xp_array=NULL, *holes_array=NULL,  **u_local_hist_array=NULL;

     

IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n","[","]","[","]");

  DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // 
  DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of nodes
  DMDAGetCorners(elas_da, &si_wg, &sj_wg,0,&size_x,&size_y,NULL); // gets values w/o including ghost cells
  DMDAGetGhostCorners(elas_da, &si_g, &sj_g,0, &sizeg_x,&sizeg_y,NULL);

  
//get local displacements for the current step
   DMGetLocalVector(elas_da,&u_local);

   ierr = DMGlobalToLocalBegin(elas_da, U, INSERT_VALUES, u_local); CHKERRQ(ierr);
   ierr = DMGlobalToLocalEnd(elas_da, U, INSERT_VALUES, u_local); CHKERRQ(ierr);
   VecGetArray(u_local,&u_local_array);
   //DMRestoreLocalVector(elas_da,&u_local);

// get local elental density values
  
  DMCreateLocalVector(prop_da, &xp);
//if(iter==5) VecView(u_local, PETSC_VIEWER_STDOUT_WORLD);
   ierr = DMGlobalToLocalBegin(prop_da, xPhys, INSERT_VALUES, xp); CHKERRQ(ierr);
   ierr = DMGlobalToLocalEnd(prop_da, xPhys, INSERT_VALUES, xp); CHKERRQ(ierr);
   VecGetArray(xp,&xp_array);
   VecGetLocalSize(xp,&xp_size); // get size of local density vector

// get holes tags
 DMCreateLocalVector(prop_da, &holesLocalVec);
 ierr = DMGlobalToLocalBegin(prop_da, hole_elem, INSERT_VALUES, holesLocalVec); CHKERRQ(ierr);
 ierr = DMGlobalToLocalEnd(prop_da, hole_elem, INSERT_VALUES, holesLocalVec); CHKERRQ(ierr);
 VecGetArray(holesLocalVec,&holes_array);

    VectorXd rho(xp_size);
    VectorXd holes(xp_size);

for(PetscInt i=0;i<xp_size;i++){
rho(i)=xp_array[i];
holes(i) = holes_array[i];

}
VecRestoreArray(xp, &xp_array);
VecRestoreArray(holesLocalVec, &holes_array);

//if(rank==0){std::cout<<rho<<std::endl;}
// get local displacements for the history steps
PetscMalloc1(iter, &u_local_hist);
PetscMalloc1(iter, &u_local_hist_array);

for(PetscInt i=0;i<iter;i++){
VecDuplicate(u_local,&u_local_hist[i]);
//VecGetArray(u_local_hist[i],&u_local_hist_array[i]);
}


//DMRestoreLocalVector(elas_da,&u_local);


  
  for (PetscInt i=0;i<iter;i++){
   ierr = DMGlobalToLocalBegin(elas_da, U_hist[i], INSERT_VALUES, u_local_hist[i]); CHKERRQ(ierr);
   ierr = DMGlobalToLocalEnd(elas_da, U_hist[i], INSERT_VALUES, u_local_hist[i]); CHKERRQ(ierr);
   VecGetArray(u_local_hist[i],&u_local_hist_array[i]);

}




// get local Force properties

 
  DMGetLocalVector(elas_da,&local_F);
  VecZeroEntries(local_F);
  DMDAVecGetArray(elas_da,local_F,&ff);




   // get local coordinates //
   DMGetCoordinateDM(elas_da, &cdm);
   DMGetCoordinatesLocal(elas_da, &coords); // get local coordinates with ghost cells
   DMDAVecGetArray(cdm,coords,&local_coords);  // convert Vec coords into array/structure

MatZeroEntries(KTAN);
VecZeroEntries(Fint);


//printf("rank [%d] si[%d] sj[%d]\n", rank, si, sj);
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//										ELEMENTAL LOOP 


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ctr_y=0;
for (ej = sj; ej < sj+ny; ej++) {
ctr_x=0;
    for (ei = si; ei < si+nx; ei++) {

   

double rho_e=rho(e);
       PetscScalar el_coords[8];
       PetscScalar el_disp[8];
       PetscScalar **el_hist_disp;
       PetscScalar *FintArray, *KArray, *dFdxArray;
       VectorXd gp(ngp);

      // VectorXd ue(8);
       PetscScalar DET,eta,psi;
       MatrixXd B(3,8);    
       VectorXd sigma(3);
       MatrixXd sigma_vis(iter, 3);

       MatrixXd Dmat(3,3);
       MatrixXd Dmat_vis(3,3);
       PetscInt int_pt;
       VectorXd epsilon(3);
       MatrixXd epsilon_hist(iter,3);
       MatrixXd ue_hist(iter,8);
       VectorXd ue_star(3);
       MatrixXd epsilon_total(iter+1,3);
       MatrixXd dfint_dx(iter,3);
      VectorXd dfint_dx_row(3);
//initialize
      ierr = PetscCalloc1(8, &FintArray); CHKERRQ(ierr);  // allocate zeroed space
      ierr = PetscCalloc1(8*8, &KArray); CHKERRQ(ierr);  // allocate zeroed space
      ierr = PetscCalloc1(8, &dFdxArray); CHKERRQ(ierr);  // allocate zeroed space

      ierr = PetscCalloc1(iter, &el_hist_disp);
      ierr=  PetscCalloc1(1,&blk);

      for(PetscInt i=0;i<iter;i++){
      PetscMalloc1(8, &el_hist_disp[i]);
      }

      vecMap elFint(FintArray,8 );  // map eigen vector to memory
      vecMap dFdx(dFdxArray,8 );   // map eigen vector to memory
      matMap elK(KArray, 8, 8);    // map eigen matrix to memory
      gp<<0.577350269189626,-0.577350269189626;// gauss quadrature points
      PetscScalar w=1.0;

     // ue=VectorXd::Zero(8);
      int_pt=0;

       // get elemental displacements
//if((sizeg_x-size_x)==1){

// for current step
    PetscInt val1, val2;
    //val2=(ctr_y+1)*(sizeg_x+1)*2;
  // val1=ctr_y*(sizeg_x+1)*2;

val2=(ctr_y+1)*(sizeg_x)*2;
   val1=ctr_y*(sizeg_x)*2;

  el_disp[0]=u_local_array[ctr_x*2+val1];
  el_disp[1]=u_local_array[ctr_x*2+1+val1]; 
  el_disp[2]=u_local_array[val1+(ctr_x+1)*2];
  el_disp[3]=u_local_array[val1+1+(ctr_x+1)*2];
  el_disp[4]=u_local_array[val2+(ctr_x+1)*2];
  el_disp[5]=u_local_array[val2+1+(ctr_x+1)*2]; 
  el_disp[6]=u_local_array[val2+ctr_x*2]; 
  el_disp[7]=u_local_array[val2+1+ctr_x*2];

// for history steps
  
for(PetscInt i=0;i<iter;i++){
   PetscInt val1, val2;
  // val2=(ctr_y+1)*(size_x+1)*2;
  // val1=ctr_y*(size_x+1)*2;

val2=(ctr_y+1)*(sizeg_x)*2;
   val1=ctr_y*(sizeg_x)*2;

  el_hist_disp[i][0]=u_local_hist_array[i][ctr_x*2+val1];
  el_hist_disp[i][1]=u_local_hist_array[i][ctr_x*2+1+val1]; 
  el_hist_disp[i][2]=u_local_hist_array[i][val1+(ctr_x+1)*2];
  el_hist_disp[i][3]=u_local_hist_array[i][val1+1+(ctr_x+1)*2];
  el_hist_disp[i][4]=u_local_hist_array[i][val2+(ctr_x+1)*2];
  el_hist_disp[i][5]=u_local_hist_array[i][val2+1+(ctr_x+1)*2]; 
  el_hist_disp[i][6]=u_local_hist_array[i][val2+ctr_x*2]; 
  el_hist_disp[i][7]=u_local_hist_array[i][val2+1+ctr_x*2];
 

}

//}

if((sizeg_x-size_x)==0){
    PetscInt val1, val2;
   val2=(ctr_y+1)*(size_x)*2;
   val1=ctr_y*(size_x)*2;
   el_disp[0]=u_local_array[ctr_x*2+val1];
   el_disp[1]=u_local_array[ctr_x*2+1+val1]; 
   el_disp[2]=u_local_array[val1+(ctr_x+1)*2]; 
   el_disp[3]=u_local_array[val1+1+(ctr_x+1)*2];
   el_disp[4]=u_local_array[val2+(ctr_x+1)*2];
   el_disp[5]=u_local_array[val2+1+(ctr_x+1)*2]; 
   el_disp[6]=u_local_array[val2+ctr_x*2]; 
   el_disp[7]=u_local_array[val2+1+ctr_x*2];

// for history variables
for(PetscInt i=0;i<iter;i++){
   PetscInt val1, val2;
   val2=(ctr_y+1)*(size_x)*2;
   val1=ctr_y*(size_x)*2;
  el_hist_disp[i][0]=u_local_hist_array[i][ctr_x*2+val1];
  el_hist_disp[i][1]=u_local_hist_array[i][ctr_x*2+1+val1]; 
  el_hist_disp[i][2]=u_local_hist_array[i][val1+(ctr_x+1)*2];
  el_hist_disp[i][3]=u_local_hist_array[i][val1+1+(ctr_x+1)*2];
  el_hist_disp[i][4]=u_local_hist_array[i][val2+(ctr_x+1)*2];
  el_hist_disp[i][5]=u_local_hist_array[i][val2+1+(ctr_x+1)*2]; 
  el_hist_disp[i][6]=u_local_hist_array[i][val2+ctr_x*2]; 
  el_hist_disp[i][7]=u_local_hist_array[i][val2+1+ctr_x*2];
 

}
}



vecMap ue(el_disp,8 );  // map eigen vector to memory

for(PetscInt i=0;i<iter;i++){
vecMap ue_hist_temp(el_hist_disp[i],8);
	for(PetscInt j=0;j<8;j++){
      ue_hist(i,j)=ue_hist_temp(j);

	}
}


for(PetscInt i=0;i<iter;i++){
     PetscFree(el_hist_disp[i]);
      }
PetscFree(el_hist_disp);

//if(iter==5){if(rank==0){if(e==0){std::cout<< ue <<std::endl;}}}


        // get coordinates for the element //
   el_coords[NSD*0+0] = local_coords[ej][ei].x;      el_coords[NSD*0+1] = local_coords[ej][ei].y;
   el_coords[NSD*1+0] = local_coords[ej][ei+1].x;    el_coords[NSD*1+1] = local_coords[ej][ei+1].y;
   el_coords[NSD*2+0] = local_coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = local_coords[ej+1][ei+1].y;
   el_coords[NSD*3+0] = local_coords[ej+1][ei].x;    el_coords[NSD*3+1] = local_coords[ej+1][ei].y;

  MatMap el_coords_eig(el_coords, 4,2);        // convert CCur from scalar to MatrixXd
//if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<ue<<std::endl;}}
  elFint = VectorXd::Zero(8);
  elK = MatrixXd::Zero(8,8); 
//==================== guass quadrature	 =========================================
  // get the element level quantities
   for(int i=0;i<ngp;i++)
	{for(int j=0;j<ngp;j++){
			eta=gp(i);psi=gp(j);
			int_pt=int_pt+1;                     // integration point number
			shapel(psi,eta,el_coords_eig,DET,B, int_pt, e, iter);      // call shape functions
			epsilon=B*ue;                      // calculate small strain

    for(PetscInt k=0;k<iter;k++){
                      ue_star=ue_hist.row(k);
                      epsilon_hist.row(k)=B*ue_star;
                      epsilon_total.row(k)=B*ue_star;
                      }
                        epsilon_total.row(iter)=epsilon;

// if (flag==1) if((i==0)&& (j==0)){if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<epsilon_total<<std::endl;std::cout<<"*************"<<std::endl;std::cout<<epsilon<<std::endl;}}}
 SMP_cycle(rank, epsilon_total, e, T, T_hist, delta_t, rho, holes, penal,iter, int_pt, B, sigma_vis, Dmat_vis, dfint_dx, flag);
  //if (flag==0) {if(e==0){if(iter==5){std::cout<<iter <<sigma_vis<<std::endl;}}}
//std::cout<<iter <<sigma_vis<<std::endl;
		       //stiffness(Dmat);
			//sigma=Dmat*epsilon;               // calculate small stress 
                       Dmat=Dmat_vis;
                       sigma=sigma_vis.row(iter-1);  
                       dfint_dx_row=dfint_dx.row(iter-1);


			elFint+=  B.transpose()*sigma*DET*w;
                        elK+=w*B.transpose()*Dmat*B*DET;
			dFdx+=B.transpose()*dfint_dx_row*DET*w;

//if(int_pt==1){if (flag==0) {if(e==176){if(iter==5){std::cout<<rho_e<<" "<<Dmat.format(HeavyFmt)<<std::endl;}}}}

          }//j loop
}//i loop
//========================================================================================	
//if(iter==4){if(flag==0){if(rank==0){if(e==176){std::cout<<elK.format(HeavyFmt)<<std::endl;}}}}
		//convert fint from vectorXd into vec or scalar




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

         ierr = DMDASetValuesLocalStencil_ADD_VALUES(ff,s_u,FintArray);CHKERRQ(ierr);

	MatSetValuesStencil(KTAN, 8, s_u, 8, s_u, KArray, ADD_VALUES);
       blk[0]=e;
        VecSetValuesBlockedLocal(dFint_dx, 1, blk, dFdxArray, INSERT_VALUES);

	
 e=e+1;                                     // update the element number
//if (rank==1) {if(e==1) {printf("el_disp_0[%g] el_disp_1[%g] el_disp_2[%g] el_disp_3[%g] el_disp_4[%g]\n", el_disp[0], el_disp[1], el_disp[2], el_disp[3], el_disp[4]);}}
ctr_x=ctr_x+1;

             PetscFree(FintArray); CHKERRQ(ierr);  // allocate zeroed space
      ierr = PetscFree(KArray); CHKERRQ(ierr);  // allocate zeroed space
      ierr = PetscFree(dFdxArray); CHKERRQ(ierr);  // allocate zeroed space
             PetscFree(blk);


       }//end ei
ctr_y=ctr_y+1;
   }//end ej


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DMLocalToGlobalBegin(elas_da,local_F,ADD_VALUES,Fint);
DMLocalToGlobalEnd(elas_da,local_F,ADD_VALUES,Fint);
 
//if(iter==5){ VecView(local_F, PETSC_VIEWER_STDOUT_SELF);}
//************************** elemental loop ends ************************************************
MatAssemblyBegin(KTAN, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(KTAN, MAT_FINAL_ASSEMBLY);
//VecView(Fint,PETSC_VIEWER_STDOUT_WORLD); 
VecAssemblyBegin(dFint_dx);
VecAssemblyEnd(dFint_dx);

 DMDAVecRestoreArray(elas_da,local_F,&ff);
 //VecRestoreArray(u_local,&u_local_array);
 DMRestoreLocalVector(elas_da,&local_F);
DMRestoreLocalVector(elas_da,&u_local);

VecDestroy(&xp);
VecDestroy(&holesLocalVec);

for(PetscInt i=0;i<iter;i++){
VecDestroy(&u_local_hist[i]);
}

PetscFree(u_local_hist);
PetscFree(u_local_hist_array);
PetscFree(holes_array);
PetscFree(xp_array);

VecDestroy(&local_F);
VecDestroy(&u_local);
DMDAVecRestoreArray(cdm,coords,&local_coords);


 PetscFunctionReturn(0);
}

//================================================================================================================================================
void stiffness(MatrixXd& Dmat)
{
double E=100;
double nu=0.25;
double lam=nu*E/((1+nu)*(1-2*nu));
double mu=E/(2*(1+nu));
Dmat<< lam+2*mu, lam,     0,
	lam,      lam+2*mu,0,
	0,        0,       mu;
}
//================================================================================================================================================
static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(ElasticityDOF **fields_F,MatStencil u_eqn[],PetscScalar Fe_u[])
{
  PetscInt n;

  PetscFunctionBeginUser;
  for (n = 0; n < 4; n++) {
    fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof     = fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof+Fe_u[2*n];
    fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof = fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof+Fe_u[2*n+1];
  }
  PetscFunctionReturn(0);
}
//=================================================================================================================================================
static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM da,PetscInt **_lx,PetscInt **_ly)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       proc_I,proc_J;
  PetscInt       cpu_x,cpu_y;
  PetscInt       local_mx,local_my;
  Vec            vlx,vly;
  PetscInt       *LX,*LY,i;
  PetscScalar    *_a;
  Vec            V_SEQ;
  VecScatter     ctx;

  PetscFunctionBeginUser;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  DMDAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);

  proc_J = rank/cpu_x;
  proc_I = rank-cpu_x*proc_J;

  ierr = PetscMalloc1(cpu_x,&LX);CHKERRQ(ierr);
  ierr = PetscMalloc1(cpu_y,&LY);CHKERRQ(ierr);

  ierr = DMDAGetElementsSizes(da,&local_mx,&local_my,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&vlx);CHKERRQ(ierr);
  ierr = VecSetSizes(vlx,PETSC_DECIDE,cpu_x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vlx);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&vly);CHKERRQ(ierr);
  ierr = VecSetSizes(vly,PETSC_DECIDE,cpu_y);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vly);CHKERRQ(ierr);

  ierr = VecSetValue(vlx,proc_I,(PetscScalar)(local_mx+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(vly,proc_J,(PetscScalar)(local_my+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vlx);VecAssemblyEnd(vlx);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vly);VecAssemblyEnd(vly);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(vlx,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < cpu_x; i++) LX[i] = (PetscInt)PetscRealPart(_a[i]);
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(vly,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < cpu_y; i++) LY[i] = (PetscInt)PetscRealPart(_a[i]);
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  *_lx = LX;
  *_ly = LY;

  ierr = VecDestroy(&vlx);CHKERRQ(ierr);
  ierr = VecDestroy(&vly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//==================================================================================
static PetscErrorCode BCApply_WEST(DM elas_da,PetscMPIInt rank, PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si,sj,nx,ny,i,j;
  PetscInt               M,N;
  DMDACoor2d             **_coords;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscScalar            *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping ltogm;
  PetscInt                ltogmsize;

  PetscFunctionBeginUser;
  /* enforce bc's */
  ierr = DMGetLocalToGlobalMapping(elas_da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
  ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);
//ISLocalToGlobalMappingView(ltogm,PETSC_VIEWER_STDOUT_WORLD);
for(PetscInt i=0;i<ltogmsize;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,g_idx[i]);// same as just a grid with no
}
    //PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);

  ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(elas_da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);
//DMView(elas_da,PETSC_VIEWER_STDOUT_WORLD);

  /* --- */

  ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt                 local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];
   //printf("rank [%d] local_id[%d] bc_global_id_j[%d] \n",rank,  local_id, bc_global_ids[j]);

   
    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
  }

  ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
  nbcs = 0;
  if (si == 0) nbcs = ny;

  
//for (PetscInt i = 0; i < ny*n_dofs; i++) printf("rank [%d] bc_global_ids[%d] \n",rank, bc_global_ids[i]);

  if (A) {
    ierr = MatZeroRowsColumns(A,nbcs,bc_global_ids,0.0,0,0);CHKERRQ(ierr);
  }

 if (b) {
     VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);
    VecAssemblyBegin(b);
     VecAssemblyEnd(b);
   }

  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//================================================================================================================================================================

static PetscErrorCode BCApply_SOUTH(DM elas_da,PetscMPIInt rank, PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
DM                     cda;
PetscInt               si,sj,nx,ny;
PetscScalar            *bc_vals;
  PetscInt               M,N;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscInt               nbcs;
  PetscInt               n_dofs;
ISLocalToGlobalMapping ltogm;
PetscInt                ltogmsize;
PetscErrorCode ierr;


  PetscFunctionBeginUser;
  
  ierr = DMGetLocalToGlobalMapping(elas_da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
  ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

for(PetscInt i=0;i<ltogmsize;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,g_idx[i]);// same as just a grid with no
}
   // PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);

  ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
  //ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
 // ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(elas_da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

ierr = PetscMalloc1(nx,&bc_global_ids);CHKERRQ(ierr);
ierr = PetscMalloc1(nx,&bc_vals);CHKERRQ(ierr);
for (PetscInt i = 0; i < nx; i++) bc_global_ids[i] = -1;

//printf("rank [%d] nx[%d]  \n",rank, nx);
// go over all the local points
PetscInt local_id =0; 
PetscInt j =0;
	for(PetscInt i=0;i<nx;i++){
		if(sj+j==0){//printf("rank [%d] local_id[%d] \n",rank,  (i));
		bc_global_ids[i] = g_idx[(i+1)*2-d_idx];
		
		bc_vals[i] =  bc_val;
		//if(rank==0)printf("rank [%d] id[%d] val[%f] \n",rank, (i+1)*2-d_idx, bc_vals[i]);
	        }	
	}

//for (PetscInt i = 0; i < nx; i++) printf("rank [%d] bc_global_ids[%d] \n",rank, bc_global_ids[i]);


ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
nbcs = 0;
if(sj==0) nbcs = nx;

if (A) {
    ierr = MatZeroRowsColumns(A,nbcs,bc_global_ids,0.0,0,0);CHKERRQ(ierr);
  }

 if (b) {
     VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);
    VecAssemblyBegin(b);
     VecAssemblyEnd(b);
   }

ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  
  PetscFunctionReturn(0);


}
//==================================================================================================================================================================
static PetscErrorCode ApplyBC(DM elas_da, PetscMPIInt rank, Mat KTAN, Vec Fint, IS *dof,Mat *KFF, Vec *Fint_f)
{
PetscErrorCode ierr;
  PetscInt       start,end,m;
  PetscInt       *unconstrained;
  PetscInt       cnt,i;
  Vec            x;
  PetscScalar    *_x;
  IS             is;
  VecScatter     scat;

ierr = VecDuplicate(Fint,&x);CHKERRQ(ierr);
ierr = BCApply_WEST(elas_da,rank,0,1.0,KTAN,x);CHKERRQ(ierr);// apply bc on x dof
ierr = BCApply_WEST(elas_da,rank,1,1.0,KTAN,x);CHKERRQ(ierr);//apply on y dof
ierr = BCApply_SOUTH(elas_da, rank, 1, 1.0 ,KTAN,x);CHKERRQ(ierr); // apply on y dof at base



 /* define which dofs are not constrained */
  ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&unconstrained);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArray(x,&_x);CHKERRQ(ierr);
  cnt  = 0;
  //for (i = 0; i < m; i+=2) {
   // PetscReal val1,val2;

   // val1 = PetscRealPart(_x[i]);
    //val2 = PetscRealPart(_x[i+1]);

//if(rank==0)printf("rank [%d] i[%d] val1[%f] val2[%f] _x_i[%g] _x_i+1[%g]\n",rank, i,val1,val2, _x[i], _x[i+1]);
   // if (PetscAbs(val1) < 0.1 && PetscAbs(val2) < 0.1) {
    //  unconstrained[cnt] = start + i;
     // cnt++;
     // unconstrained[cnt] = start + i + 1;
    //  cnt++;
    //}
 // }

for (i = 0; i < m; i++) {
    PetscReal val1;

    val1 = PetscRealPart(_x[i]);
    //val2 = PetscRealPart(_x[i+1]);

//if(rank==0)printf("rank [%d] i[%d] val1[%f] val2[%f] _x_i[%g] _x_i+1[%g]\n",rank, i,val1,val2, _x[i], _x[i+1]);
    if (PetscAbs(val1) < 0.1) {
      unconstrained[cnt] = start + i;
      cnt++;
      //unconstrained[cnt] = start + i + 1;
      //cnt++;
    }
  }





for(i=0;i<cnt;i++){
//printf(" rank[%d] i [%d]  glo_in[%d]  \n",rank,i,unconstrained[i]);
}
//PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  ierr = VecRestoreArray(x,&_x);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,unconstrained,PETSC_COPY_VALUES,&is);CHKERRQ(ierr); // IS containes all the free dofs
  ierr = PetscFree(unconstrained);CHKERRQ(ierr); 
  ierr = ISSetBlockSize(is,1);CHKERRQ(ierr);
//ISView(is,PETSC_VIEWER_STDOUT_WORLD);
 //VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  /* define correction for dirichlet in the rhs */
  //ierr = MatMult(KTAN,x,Fint);CHKERRQ(ierr);
 // ierr = VecScale(Fint,-1.0);CHKERRQ(ierr);

  /* get new matrix */
  ierr = MatCreateSubMatrix(KTAN,is,is,MAT_INITIAL_MATRIX,KFF);CHKERRQ(ierr);
  /* get new vector */
  ierr = MatCreateVecs(*KFF,NULL,Fint_f);CHKERRQ(ierr);
VecScatterCreate(Fint,is,*Fint_f,NULL,&scat);
//ierr = VecScatterCreateWithData(Fint,is,*Fint_f,NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat,Fint,*Fint_f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,Fint,*Fint_f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
 //VecView(Fint, PETSC_VIEWER_STDOUT_WORLD);
*dof=is;
VecScatterDestroy(&scat);
VecDestroy(&x);
 PetscFunctionReturn(0);
}

//==========================================================================================================================
PetscErrorCode Reduced_dens(DM prop_da, Vec &hole_elem, IS &is_red, Vec &full_vec_1, Vec &full_vec_2, Vec *red_vec_1, Vec *red_vec_2)
{

PetscErrorCode ierr;
  PetscInt       start,end,m;
  PetscInt       *notholes;
  PetscInt       cnt,i;
  PetscScalar    *_hole_elem;
  //IS             is;
  VecScatter     scatxPhys,scatx;
 Mat             K_holes;
 Mat             K_holes_red;
// Create a matrix of hole elements
DMCreateMatrix(prop_da, &K_holes);
MatZeroEntries(K_holes);





 /* define which dofs are not holes */
  ierr = VecGetLocalSize(hole_elem,&m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&notholes);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(hole_elem,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArray(hole_elem,&_hole_elem);CHKERRQ(ierr);
  cnt  = 0;
  

for (i = 0; i < m; i++) {
    PetscReal val1;

    val1 = PetscRealPart(_hole_elem[i]);
   
//if(rank==0)printf("rank [%d] i[%d] val1[%f] val2[%f] _x_i[%g] _x_i+1[%g]\n",rank, i,val1,val2, _x[i], _x[i+1]);
    if (PetscAbs(val1) == 0.0) {
      notholes[cnt] = start + i;
      cnt++;

    }
  }





for(i=0;i<cnt;i++){
//printf(" rank[%d] i [%d]  glo_in[%d]  \n",rank,i,unconstrained[i]);
}
//PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  ierr = VecRestoreArray(hole_elem,&_hole_elem);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,notholes,PETSC_COPY_VALUES,&is_red);CHKERRQ(ierr); // IS containes all the free dofs
  ierr = PetscFree(notholes);CHKERRQ(ierr); 
  ierr = ISSetBlockSize(is_red,1);CHKERRQ(ierr);

 
//ISView(is,PETSC_VIEWER_STDOUT_WORLD);
//MatView(K_holes, PETSC_VIEWER_STDOUT_WORLD);


/// get new matrix //
  ierr = MatCreateSubMatrix(K_holes,is_red,is_red,MAT_INITIAL_MATRIX, &K_holes_red);CHKERRQ(ierr);


// get new vector //
  ierr = MatCreateVecs(K_holes_red,NULL,red_vec_1);CHKERRQ(ierr);
  ierr = MatCreateVecs(K_holes_red, NULL, red_vec_2); CHKERRQ(ierr);
//ierr = DMCreateGlobalVector(prop_da, xPhys_red);
//ierr = DMCreateGlobalVector(prop_da, x_red);


VecScatterCreate(full_vec_2,is_red,*red_vec_2,NULL,&scatx);
//ierr = VecScatterCreateWithData(Fint,is,*Fint_f,NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatx,full_vec_2,*red_vec_2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatx,full_vec_2,*red_vec_2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
 //VecView(Fint, PETSC_VIEWER_STDOUT_WORLD);

VecScatterCreate(full_vec_1,is_red,*red_vec_2,NULL,&scatxPhys);
//ierr = VecScatterCreateWithData(Fint,is,*Fint_f,NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatxPhys,full_vec_1,*red_vec_1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatxPhys,full_vec_1,*red_vec_1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);




VecScatterDestroy(&scatx);
VecScatterDestroy(&scatxPhys);
MatDestroy(&K_holes);
MatDestroy(&K_holes_red);



 PetscFunctionReturn(0);


}

//=========================================================================================================================
 static PetscErrorCode DMDABCApplyCompression(DM elas_da,PetscMPIInt rank,Mat A,Vec f)
 {

 
  BCApply_WEST(elas_da,rank,0,0.0,A,f);
   BCApply_WEST(elas_da,rank,1,0.0,A,f);
   return(0);
 }
//============================================================================================================================
static PetscErrorCode DMDAViewGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[])
{
  DM             cda;
  Vec            coords,local_fields;
  DMDACoor2d     **_coords;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN];
  const char     *field_name;
  PetscMPIInt    rank;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       n_dofs,d;
  PetscScalar    *_fields;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = PetscSNPrintf(fname,sizeof(fname),"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
  if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
  for (d = 0; d < n_dofs; d++) {
    ierr = DMDAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(da,&local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = VecGetArray(local_fields,&_fields);CHKERRQ(ierr);

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;
      PetscScalar field_d;

      coord_x = _coords[j][i].x;
      coord_y = _coords[j][i].y;

      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",PetscRealPart(coord_x),PetscRealPart(coord_y));CHKERRQ(ierr);
      for (d = 0; d < n_dofs; d++) {
        field_d = _fields[n_dofs*((i-si)+(j-sj)*(nx))+d];
        ierr    = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e ",PetscRealPart(field_d));CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(local_fields,&_fields);CHKERRQ(ierr);
  ierr = VecDestroy(&local_fields);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


//===========================================================================================================================================================================================
PetscErrorCode  dfdrho_func(DM elas_da, DM prop_da, PetscInt rank, PetscInt step, Vec &dfdrho, Vec &psi_n, Vec &dRdx){
    PetscErrorCode ierr;
    PetscFunctionBeginUser;

    PetscInt         si, sj, ei, ej, nx, ny, nel_p, nen,size_x, sizeg_x, sizeg_y, size_y,ctr_x,ctr_y;
    const PetscInt   *el_indices;
    ISLocalToGlobalMapping l2g;
    // get local element ownership
    PetscInt hStart, hEnd, *gInd;
    //ierr = DMPlexGetHeightStratum(optRef.dm_nodes, 0, &hStart, &hEnd); CHKERRQ(ierr);
    // pull adjoint vector into local vector
    Vec localPsi_n;

    ierr = DMGetLocalVector(elas_da, &localPsi_n); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(elas_da, psi_n, INSERT_VALUES, localPsi_n); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(elas_da, psi_n, INSERT_VALUES, localPsi_n); CHKERRQ(ierr);
    //if(step==2) if(rank==0) VecView(localPsi_n, PETSC_VIEWER_STDOUT_SELF);
    PetscScalar *localPsi;
    VecGetArray(localPsi_n, &localPsi);
    // set up array to get block of values from dRdx
    PetscInt bSize;
    PetscInt vStart, vEnd;
    IS blockIS;
    const PetscInt *idx;
    PetscInt nBlk = 1, *blk;
    PetscScalar *dRdxArray = NULL;
    ierr = VecGetBlockSize(dRdx, &bSize); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(dRdx, &vStart, &vEnd); CHKERRQ(ierr);
    ierr = PetscMalloc1(nBlk, &blk); CHKERRQ(ierr);
    ierr = PetscMalloc1(bSize, &dRdxArray); CHKERRQ(ierr);
    vecMap dRdx_i_map(dRdxArray, bSize);
    //if (rank==0) std::cout<< dRdx_i_map<<std::endl;

    PetscInt cSize;
    PetscScalar *psiArray = NULL;
    PetscInt *dfdxIdx = NULL;
    PetscScalar *dfdxVal = NULL;
    // allocate arrays for df/dx_i values and indices
    DMDAGetElements(elas_da,&nel_p,&nen,&el_indices);

    ierr = PetscCalloc1(nel_p, &dfdxIdx); CHKERRQ(ierr);
    ierr = PetscCalloc1(nel_p, &dfdxVal); CHKERRQ(ierr);
    ierr= PetscCalloc1(8, &psiArray);
   

     DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // w/o the ghost cells
     DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of nodes
     DMDAGetCorners(elas_da, NULL, NULL,0,&size_x,&size_y,NULL); // gets values w/o including ghost cells
     DMDAGetGhostCorners(elas_da, NULL,NULL,0, &sizeg_x,&sizeg_y,NULL);


//if(rank==0) printf("sizeg_x[%d] sizeg_y[%d] \n", sizeg_x, sizeg_y);

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//										ELEMENTAL LOOP 


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PetscInt elem=0;
ctr_y=0;
for (ej = sj; ej < sj+ny; ej++) {
ctr_x=0;
    for (ei = si; ei < si+nx; ei++) {
       //printf("here");
        blk[0] = vStart/bSize + elem;
        ierr = ISCreateBlock(PETSC_COMM_SELF, bSize, nBlk, blk, PETSC_COPY_VALUES, &blockIS); CHKERRQ(ierr);
        ierr = ISGetIndices(blockIS, &idx); CHKERRQ(ierr);
        ierr = VecGetValues(dRdx, bSize, idx, dRdxArray); CHKERRQ(ierr);
        
         PetscInt val1, val2;


        val2=(ctr_y+1)*(sizeg_x)*2;
       val1=ctr_y*(sizeg_x)*2;

      psiArray[0]=localPsi[ctr_x*2+val1];
      psiArray[1]=localPsi[ctr_x*2+1+val1]; 
      psiArray[2]=localPsi[val1+(ctr_x+1)*2]; 
      psiArray[3]=localPsi[val1+1+(ctr_x+1)*2];
      psiArray[4]=localPsi[val2+(ctr_x+1)*2];
      psiArray[5]=localPsi[val2+1+(ctr_x+1)*2]; 
      psiArray[6]=localPsi[val2+ctr_x*2]; 
      psiArray[7]=localPsi[val2+1+ctr_x*2];
        //ierr = DMPlexVecGetClosure(optRef.dm_nodes, NULL, localPsi_n, elem, &cSize, &psiArray); CHKERRQ(ierr);
        // map to eigen vectors
        vecMap psi_i(psiArray, 8);
   //if(step==2){ if(rank==0){if(elem==1){std::cout<< ctr_x<<std::endl;std::cout<< ctr_y<<std::endl;std::cout<< step<<std::endl;std::cout<< "***dfdx*****"<<std::endl;std::cout<< dRdx_i_map <<std::endl;std::cout<< "****psi****"<<std::endl;std::cout<< psi_i <<std::endl;}}}
        //if(rank=0) std::cout<<psi_i<<std::endl;

        // df/dx_i = {psi_i}^T * {dR/dx_i}
        dfdxVal[elem] = psi_i.transpose()*dRdx_i_map;//psi_i.dot(dRdx_i_map);
        dfdxIdx[elem] = elem;
        //ierr = DMPlexVecRestoreClosure(optRef.dm_nodes, NULL, localPsi_n, elem, &cSize, &psiArray); CHKERRQ(ierr);
        ierr = ISRestoreIndices(blockIS, &idx); CHKERRQ(ierr);
        ierr = ISDestroy(&blockIS); CHKERRQ(ierr);
       
elem=elem+1;
ctr_x=ctr_x+1;
    }
ctr_y=ctr_y+1;
}
    // assemble into df/dx matrix, adding values
   //PetscMalloc1(nel_p, &gInd);
//for(int i=0;i<nel_p;i++){
//gInd[i]=(vStart/8)+i;

//}
    DMGetLocalToGlobalMapping(prop_da, &l2g);
    VecSetLocalToGlobalMapping(dfdrho, l2g);
    ierr = VecSetValuesLocal(dfdrho, nel_p, dfdxIdx, dfdxVal, ADD_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(dfdrho); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(dfdrho); CHKERRQ(ierr);
    ierr = PetscFree(dfdxVal); CHKERRQ(ierr);
    ierr = PetscFree(dfdxIdx); CHKERRQ(ierr);
    ierr = PetscFree(dRdxArray); CHKERRQ(ierr);
    ierr = PetscFree(blk); CHKERRQ(ierr);
    PetscFree(psiArray);
    ierr = DMRestoreLocalVector(elas_da, &localPsi_n); CHKERRQ(ierr);
	
    PetscFunctionReturn(ierr);
}

//=================================================================================================================================================================================

PetscErrorCode  Force_ext(DM elas_da,  PetscInt rank, Vec &Fext, PetscInt nelx, PetscInt nely, PetscScalar val ){

DM                     cda;
PetscInt               si,sj,nx,ny,i,j;
PetscScalar            *bc_vals;
  PetscInt               M,N;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscInt               nbcs;
  PetscInt               n_dofs;
ISLocalToGlobalMapping ltogm;
PetscInt                ltogmsize;
PetscErrorCode ierr;


  PetscFunctionBeginUser;
  
  ierr = DMGetLocalToGlobalMapping(elas_da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
  ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

for(PetscInt i=0;i<ltogmsize;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,g_idx[i]);// same as just a grid with no
}
   // PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);

  ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
  //ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
 // ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(elas_da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);
for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt                 local_id;
  

    local_id = nx*(j+1);

    bc_global_ids[j] = g_idx[n_dofs*(local_id-1)];
  // printf("rank [%d] local_id[%d] bc_global_id_j[%d] \n",rank,  local_id, bc_global_ids[j]);
   //printf("rank[%d] g_idx[7] [%d]\n", rank, g_idx[6]);
 bc_vals[j] =  2*(val/(nely+1));
if(sj+ny-1==nely && j==ny-1) {bc_vals[j] = val/(nely+1);} 

if (sj==0 && j==0) {bc_vals[j] = val/(nely+1);}
}

ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
   nbcs = 0;
  if (si+(nx-1) == nelx) nbcs = ny;

 


     VecSetValues(Fext,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);
   VecAssemblyBegin(Fext);
     VecAssemblyEnd(Fext);
   
//VecView(Fext, PETSC_VIEWER_STDOUT_WORLD);
ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  //ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
PetscFunctionReturn(0);


}
//====================================================================================================================================================================
 PetscErrorCode FFext(DM elas_da, Vec &Fext, PetscInt nelh_x, PetscInt nelh_y, PetscInt rank, PetscInt nelx, PetscInt nely, PetscScalar Force_amt, PetscInt &des_dof_local)
{
DM                     cda;
PetscInt               si,sj,nx,ny;
PetscScalar            *bc_vals;
  PetscInt               M,N;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscInt               nbcs;
  PetscInt               n_dofs;
ISLocalToGlobalMapping ltogm;
PetscInt                ltogmsize;
PetscErrorCode ierr;


  PetscFunctionBeginUser;
  
  ierr = DMGetLocalToGlobalMapping(elas_da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
  ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

for(PetscInt i=0;i<ltogmsize;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,g_idx[i]);// same as just a grid with no
}
   // PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);

  ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
  //ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
 // ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(elas_da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);
for (PetscInt i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

//PetscInt des_dof_local =0;
// go over all the local points
PetscInt local_id =0; 
for(PetscInt j=0;j<ny;j++){
	for(PetscInt i=0;i<nx;i++){
		if(si+i==nelx){//printf("rank [%d] local_id[%d] \n",rank,  local_id);
		bc_global_ids[j] = g_idx[(local_id+1)*2-2];
		bc_vals[j] =  2*(Force_amt/(nely-nelh_y+1));
		if(sj+ny-1==nely && j==ny-1) {bc_vals[j] = Force_amt/(nely-nelh_y+1);} 

		if (sj+j==nelh_y) {bc_vals[j] = Force_amt/(nely-nelh_y+1);}

		if (sj+j<nelh_y) {bc_vals[j] = 0.0;}

		if(sj+j==nelh_y){des_dof_local =g_idx[(local_id+1)*2-1];}	
		}	

local_id =local_id+1;

	}

}

//printf("rank [%d] des_dof[%d] \n",rank,  des_dof);



/*
for (PetscInt i=0;i<nx;i++){
if (si+i==nelx){
  for (PetscInt j = 0; j < ny; j++) {
    PetscInt                 local_id;
  

    local_id = nx*(j+1);

    bc_global_ids[j] = g_idx[n_dofs*(local_id-1)];
  // printf("rank [%d] local_id[%d] bc_global_id_j[%d] \n",rank,  local_id, bc_global_ids[j]);
   //printf("rank[%d] g_idx[7] [%d]\n", rank, g_idx[6]);
 bc_vals[j] =  2*(Force_amt/(nely-nelh_y+1));
if(sj+ny-1==nely && j==ny-1) {printf("rank [%d] local_id[%d] \n",rank,  local_id);bc_vals[j] = Force_amt/(nely-nelh_y+1);} 

if (sj==0 && j==nelh_y) {printf("rank [%d] local_id[%d] \n",rank,  local_id);bc_vals[j] = Force_amt/(nely-nelh_y+1);}

if (sj==0 && j<nelh_y) {bc_vals[j] = 0.0;printf("rank [%d] local_id[%d] \n",rank,  local_id);}
}
}
}

*/

ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
   nbcs = 0;
 // if (si+(nx-1) == nelx) nbcs = ny;

 nbcs = ny;

VecSetOption(Fext, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
     VecSetValues(Fext,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);
   VecAssemblyBegin(Fext);
     VecAssemblyEnd(Fext);
   
//VecView(Fext, PETSC_VIEWER_STDOUT_WORLD);
ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  //ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);



PetscFunctionReturn(0);
}





