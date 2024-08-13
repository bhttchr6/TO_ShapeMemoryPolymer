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
#include<fstream>
#include<sstream>
//#include"MMA.h"
using namespace Eigen;

using Eigen::MatrixXd;
using Eigen::VectorXd; 

typedef Eigen::Map<MatrixXd,0,Eigen::Stride<1,2> > MatMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, Dynamic, RowMajor> > matMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, 1> > vecMap;
#define ngp 2
#define ndof 2
#define ndim 2
//#define Tmax 355
//#define Tmin 325
//#define Tg 343
#define Tmax 350
#define Tmin 330
#define Tg 340
#define plane_strain 1
#define plane_stress 0

#define GAUSS_POINTS   4
#define NSD   2


int main(int argc, char *argv[]){

PetscErrorCode ierr;

 ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

PetscInt 	nelx, nely, nelh_x, nelh_y, a, stw, gxs, gys, gxm, gym, cpu_x, cpu_y,*lx=NULL, *ly=NULL,size_rho, rank,size, nx, ny, sizeg_x,sizeg_y, size_x,size_y,si_wg,sj_wg, si_g, sj_g, si, sj;
PetscScalar 	time_spec, volfrac;
PetscScalar 	Obj_val, *gx, gx_I, gxval;
Vec 		dfdrho, x, xPhys, Hs, hole_elem, xPhys_red, x_red, dfdrho_red, dgdrho_red;
Mat             H;
DM              elas_da, prop_da;
DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
DMDAStencilType  stype = DMDA_STENCIL_BOX;
PetscReal         dx, dy;
Vec               x_bar;
IS               is_red;
VecScatter       scat_red;


PetscViewer    viewer_x;
PetscViewer    viewer_xo1;
PetscViewer    viewer_xo2;
PetscViewer    viewer_xmin;
PetscViewer    viewer_xmax;
PetscViewer    viewer_L;
PetscViewer    viewer_U;
PetscViewer    viewer_text;




MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
MPI_Comm_size(MPI_COMM_WORLD, &size);

// specify the nelx and nely
nelx = 120;
nely = 60;

// Specify the filter radius
PetscScalar rmin;

// Resoource Upper Limit 
PetscScalar Vlim=0.7;  // smp volume fraction limit

// specify the global(MATLAB ver.) d.o.f at which objective is desired
a = (nelx+1)*(nely+1)*ndof; 

// specify the time at which objective is desired
time_spec=45.00;

// initialize objective function
Obj_val=0.0;  

// initialize mixture ratio
volfrac=0.3;   // volume fraction of aux material

// define the number of constraints
//PetscInt m=1;
PetscInt itr=0;
MMA *mma;

// Set-up initial mesh and DMs
stw=1;
PetscScalar l=  100.0;   //100 mm
PetscScalar w=  50.0;   // 50 mm

// No of elements for hole
nelh_x = ceil(0.25*nelx);
nelh_y = ceil(0.25*nely);

PetscPrintf(PETSC_COMM_WORLD, " nelx %d nely %d l %f w %f nelh_x %d nelh_y %d \n", nelx, nely, l, w, nelh_x, nelh_y);
// set rmin
PetscScalar rmin_x = l/nelx;
PetscScalar rmin_y = w/nely;

rmin = PetscMin(rmin_x, rmin_y);
rmin = 2.0*rmin;

ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nelx+1,nely+1,PETSC_DECIDE,PETSC_DECIDE,ndof,stw,NULL,NULL,&elas_da);CHKERRQ(ierr);
  DMSetMatType(elas_da,MATAIJ);  
  ierr = DMSetUp(elas_da);CHKERRQ(ierr);
  DMDASetUniformCoordinates(elas_da, 0.0, l, 0.0, w, 0.0, 1.0);
  DMDAGetGhostCorners(elas_da, &gxs, &gys, NULL, &gxm,&gym,NULL);// gxs and gys are local indices including ghost gxm and gym are no. of grid points in each direction on each processor

//==================================================================================================================================================
							// For checking and debigging meshes 

  //DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // ghosted corners
  //DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of elements
  //DMDAGetCorners(elas_da, &si_wg, &sj_wg,0,&size_x,&size_y,NULL); // gets no of nodal points for each processor values w/o ghost cells
  //DMDAGetGhostCorners(elas_da, &si_g, &sj_g,0, &sizeg_x,&sizeg_y,NULL);  // ghost corners

 

//printf("rank [%d]  nx[%d]  ny[%d]  \n", rank, size_x, size_y);
//printf("rank [%d]  nx[%d]  ny[%d]  \n", rank, si_wg, sj_wg);
//=====================================================================================================================================================
// SET DMDA FOR DENSITY PROPERTIES
// each element needs to have a particular density value //

  DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);
  DMDAGetElementOwnershipRanges2d(elas_da,&lx,&ly);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nelx,nely,cpu_x,cpu_y,1,0,lx,ly,&prop_da);CHKERRQ(ierr);
  ierr = DMSetUp(prop_da);CHKERRQ(ierr);

//====================================================================================================================================================
							// For checking and debugging meshes

//DMDAGetElementsCorners(prop_da,&si,&sj,0);                   // ghosted corners
//DMDAGetElementsSizes(prop_da, &nx,&ny,0);                    // gets the non-overlapping no. of elements for each processor
 // DMDAGetCorners(prop_da, &si_wg, &sj_wg,0,&size_x,&size_y,NULL); // gets the distribution of nodal points (mesh nodes: size_x, size_y) across processors values w/o ghost cells
 // DMDAGetGhostCorners(prop_da, &si_g, &sj_g,0, &sizeg_x,&sizeg_y,NULL);  // ghost corners

//printf("rank [%d]  size_x[%d]  size_y[%d]  \n", rank, size_x, size_y);


//const PetscInt         *g_idx;
 // ISLocalToGlobalMapping ltogm;
 // PetscInt                ltogmsize;

//ierr = DMGetLocalToGlobalMapping(prop_da,&ltogm);CHKERRQ(ierr);
 // ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
 // ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

//ISLocalToGlobalMappingView(ltogm,PETSC_VIEWER_STDOUT_WORLD);

//for(PetscInt i=0;i<ltogmsize;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,g_idx[i]);// same as just a grid with no
//}
  //  PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);


//=======================================================================================================================================================








  PetscFree(lx);
  PetscFree(ly);

 dx   = l/((PetscReal)(nelx));
 dy   = w/((PetscReal)(nely));

  DMDASetUniformCoordinates(prop_da,0.0+0.5*dx,l-0.5*dx,0.0+0.5*dy,w-0.5*dy,0.0,1.0); 
 
 
  DMCreateGlobalVector(prop_da, &xPhys);                           // define global elemental density vector
  VecGetSize(xPhys, &size_rho);
  VecDuplicate(xPhys,&x);

// Define the vector representing hole or solid elements
DMCreateGlobalVector(prop_da, &hole_elem);
/*

						//////Load density values from restart files
			//------------------------------------------------------------------------------------------(1)
			
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/DesPt_499.dat", FILE_MODE_READ, &viewer_x);
			VecLoad(x, viewer_x);
			//VecView(x, PETSC_VIEWER_STDOUT_WORLD);
			PetscViewerDestroy(&viewer_x);
			
*/
  VecSet(x, volfrac);
  VecSet(xPhys,volfrac);

  VecSet(hole_elem, 0.0);

//************************************************ Get appropiate indexes for placing lower density elements ********************************************
 DM                     cpa;
const PetscInt         *g_idx;
 ISLocalToGlobalMapping ltogm;
 PetscInt                ltogmsize;
PetscInt               *bc_global_ids;
PetscScalar             dens_val;

ierr = DMGetLocalToGlobalMapping(prop_da,&ltogm);CHKERRQ(ierr);
ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);
ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

ierr = DMGetCoordinateDM(prop_da,&cpa);CHKERRQ(ierr); 
ierr = DMDAGetGhostCorners(cpa,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

ierr = PetscMalloc1(ltogmsize,&bc_global_ids);CHKERRQ(ierr);
//printf("rank [%d]  si [%d] sj[%d] nx[%d]  ny[%d]  \n",  rank, si, sj, nx, ny);
PetscInt index = 0;
PetscInt bchs = ltogmsize;
PetscScalar *value;
ierr = PetscMalloc1(ltogmsize,&value);CHKERRQ(ierr);
dens_val =1.0;
for(PetscInt j=0;j<ny;j++){   
	for(PetscInt i=0; i<nx;i++){
		PetscInt local_id_x = si+i;
		PetscInt local_id_y = sj+j;
bc_global_ids[index] = -1;

		if((local_id_y< nelh_y) && (local_id_x>=nelx-nelh_x) ){ //bchs = bchs+1;
			
			 bc_global_ids[index] = g_idx[index];
			//printf("rank [%d]  id_x[%d]  id_y[%d] index[%d] \n", rank, local_id_x, local_id_y,  bc_global_ids[bchs-1]);
//printf("rank [%d]  bchs[%d] \n", rank, bchs);
			//VecSetValue(x, row, value, INSERT_VALUES);
			//VecAssemblyBegin(x); VecAssemblyEnd(x);


			}
index = index+1;
	}
} 

//printf("rank [%d]   bchs[%d] \n", rank,  bchs);
for(PetscInt i=0;i<ltogmsize;i++){
value[i]=dens_val;
//printf("rank [%d]   index[%d] \n", rank,  bc_global_ids[i]);
}
VecSetOption(hole_elem, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
VecSetValues(hole_elem,bchs,bc_global_ids,value,INSERT_VALUES);
	
VecAssemblyBegin(hole_elem);
 VecAssemblyEnd(hole_elem);

//VecSetOption(xPhys, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
//VecSetValues(xPhys, bchs, bc_global_ids, value, INSERT_VALUES);
//VecAssemblyBegin(xPhys);
//VecAssemblyEnd(xPhys);





ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
PetscFree(value);
PetscFree(bc_global_ids);		
//VecView(x, PETSC_VIEWER_STDOUT_WORLD);			 
//VecView(xPhys, PETSC_VIEWER_STDOUT_WORLD);



Reduced_dens(prop_da, hole_elem, is_red, xPhys, x, &xPhys_red, &x_red);

//VecView(xPhys_red, PETSC_VIEWER_STDOUT_WORLD);





//**********************************************************************************************************************************************************


// Calculate Filter matrix
//================================================== filter ==============================================================================================================
		DM da_elem;

		PetscInt M,N,P,md,nd,pd; 
		
		
		Vec lcoor;
		DMGetCoordinatesLocal(elas_da,&lcoor);
		PetscScalar *lcoorp;
		VecGetArray(lcoor,&lcoorp);
		ierr = DMDAGetInfo(elas_da,NULL,&M,&N,NULL,&md,&nd,NULL,NULL,NULL,&bx,&by,NULL,&stype); CHKERRQ(ierr);
		PetscInt ElemConn;
		// Check dx,dy,dz and find max conn for a given rmin
		ElemConn = (PetscInt)PetscMax(ceil(rmin/dx),ceil(rmin/dy));
		ElemConn = PetscMin(ElemConn,PetscMin((M-1)/2,(N-1)/2));

		// The following is needed due to roundoff errors 
		PetscInt tmp;
		MPI_Allreduce(&ElemConn, &tmp, 1,MPIU_INT, MPI_MAX,PETSC_COMM_WORLD );
		ElemConn = tmp;


		
		// Print to screen: mesh overlap!
		PetscPrintf(PETSC_COMM_WORLD,"# Filter radius rmin = %f results in a stencil of %i elements \n",rmin,ElemConn);

		// Find the geometric partitioning of the nodal mesh, so the element mesh will coincide 
		 PetscInt *Lx=new PetscInt[md];
		 PetscInt *Ly=new PetscInt[nd];
		

		// get number of nodes for each partition
		 const PetscInt *LxCorrect, *LyCorrect;
		DMDAGetOwnershipRanges(elas_da, &LxCorrect, &LyCorrect, NULL); 

		//DMDAGetElementOwnershipRanges2d(elas_da,&LxCorrect,&LyCorrect);

		// subtract one from the lower left corner.
		for (int i=0; i<md; i++){
			Lx[i] = LxCorrect[i];
			if (i==0){Lx[i] = Lx[i]-1;}
		}
		for (int i=0; i<nd; i++){
			Ly[i] = LyCorrect[i];
			if (i==0){Ly[i] = Ly[i]-1;}
		}


		for (int i=0; i<md; i++){
			Lx[i] = LxCorrect[i];
			if (i==0){Lx[i] = Lx[i]-1;}
		}
                 
		// Create the element grid:
		DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M-1,N-1,md,nd,1,ElemConn,Lx,Ly,&da_elem);

		// Initialize
                DMSetFromOptions(da_elem);
                DMSetUp(da_elem);
                 
		// Set the coordinates: from 0+dx/2 to xmax-dx/2 and so on
		PetscScalar xmaxi = (M-1)*dx;
		PetscScalar ymaxi = (N-1)*dy;
		
		DMDASetUniformCoordinates(da_elem , dx/2.0,xmaxi-dx/2.0, dy/2.0,ymaxi-dy/2.0, 0,0);
                 
		// Allocate and assemble
		DMCreateMatrix(da_elem,&H);
		DMCreateGlobalVector(da_elem,&Hs);

		// Set the filter matrix and vector
		DMGetCoordinatesLocal(da_elem,&lcoor);
		VecGetArray(lcoor,&lcoorp);
		DMDALocalInfo info;
		DMDAGetLocalInfo(da_elem,&info);

		// The variables from info that are used are described below:
		// -------------------------------------------------------------------------
		// sw = Stencil width
		// mx, my, mz = Global number of "elements" in each direction 
		// xs, ys, zs = Starting point of this processor, excluding ghosts
		// xm, ym, zm = Number of grid points on this processor, excluding ghosts
		// gxs, gys, gzs = Starting point of this processor, including ghosts
		// gxm, gym, gzm = Number of grid points on this processor, including ghosts
		// -------------------------------------------------------------------------
		
		// Outer loop is local part = find row
		// What is done here, is:
		// 
		// 1. Run through all elements in the mesh - should not include ghosts
		
			for (PetscInt j=info.ys; j<info.ys+info.ym; j++) {
				for (PetscInt i=info.xs; i<info.xs+info.xm; i++) {
					// The row number of the element we are considering:
					PetscInt row = (i-info.gxs) + (j-info.gys)*(info.gxm);
					//
					// 2. Loop over nodes (including ghosts) within a cubic domain with center at (i,j,k)
					//    For each element, run through all elements in a box of size stencilWidth * stencilWidth * stencilWidth 
					//    Remark, we want to make sure we are not running "out of the domain", 
					//    therefore k2 etc. are limited to the max global index (info.mz-1 etc.)
					
						for (PetscInt j2=PetscMax(j-info.sw,0);j2<=PetscMin(j+info.sw,info.my-1);j2++){
							for (PetscInt i2=PetscMax(i-info.sw,0);i2<=PetscMin(i+info.sw,info.mx-1);i2++){
								PetscInt col = (i2-info.gxs) + (j2-info.gys)*(info.gxm) ;
								PetscScalar dist = 0.0;
								// Compute the distance from the "col"-element to the "row"-element
								for(PetscInt kk=0; kk<2; kk++){
									dist = dist + PetscPowScalar(lcoorp[2*row+kk]-lcoorp[2*col+kk],2.0);
								}
								dist = PetscSqrtScalar(dist);
								
								if (dist<rmin){
									// Longer distances should have less weight
									dist = rmin-dist;
									MatSetValuesLocal(H, 1, &row, 1, &col, &dist, INSERT_VALUES); 
								}
							}
						
					
				}
			}
		}
		// Assemble H:
		MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

		//MatView(H, PETSC_VIEWER_STDOUT_WORLD);
		// Compute the Hs, i.e. sum the rows
		Vec dummy;
		VecDuplicate(Hs,&dummy);
		VecSet(dummy,1.0);
		MatMult(H,dummy,Hs);

		VecRestoreArray(lcoor,&lcoorp);
		VecDestroy(&dummy);
		delete [] Lx;
		delete [] Ly;
		//MatDestroy(&H);
		//VecDestroy(&xPhys);
		//VecDestroy(&x);
                DMDestroy(&da_elem);



//=========================================================================================================================================================================
//filter(x, elas_da, prop_da, rmin, H, Hs);

ierr = MatMult(H,x,xPhys); CHKERRQ(ierr);
VecPointwiseDivide(xPhys,xPhys,Hs);

// initialize MMA & Optimization Parameters
Vec xo1, xo2, U, L, xold, xmin, xmax;
//OPtimization Loop
PetscInt m=1;
PetscInt maxItr =1000;
PetscScalar Xmin=0.0;   // minimum value of design variable
PetscScalar Xmax=1.0;	// maximum value of design variable
PetscScalar movlim=0.2;
PetscScalar ch=1.0;
PetscScalar normInf=1;
PetscScalar norm2;
// Set MMA parameters (for multiple load cases)
PetscScalar aMMA[m];
PetscScalar cMMA[m];
PetscScalar dMMA[m];
	for (PetscInt i=0;i<m;i++){
	    aMMA[i]=0.0;
	    dMMA[i]=0.0;
	    cMMA[i]=1000.0;
	}

PetscInt nGlobalDesignVar;
VecGetSize(x_red,&nGlobalDesignVar); // ASSUMES THAT SIZE IS ALWAYS MATCHED TO CURRENT MESH




ierr = VecDuplicate(x_red,&xo1); CHKERRQ(ierr);
ierr = VecDuplicate(x_red,&xo2); CHKERRQ(ierr);
ierr = VecDuplicate(x_red,&U); CHKERRQ(ierr);
ierr = VecDuplicate(x_red,&L); CHKERRQ(ierr);
ierr = VecDuplicate(x_red, &xold);
ierr = VecDuplicate(x_red, &xmin);
ierr = VecDuplicate(x_red, &xmax);

VecCopy(x_red, xold);
VecCopy(x_red,xo1);
VecCopy(x_red,xo2);
VecCopy(x_red, L);
VecCopy(x_red, U);
/*

					//--------------------------------------------------------------(2)
			
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/xmin_499.dat", FILE_MODE_READ, &viewer_xmin);
			VecLoad(xmin, viewer_xmin);
			//VecView(xmin, PETSC_VIEWER_STDOUT_WORLD);


			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/xmax_499.dat", FILE_MODE_READ, &viewer_xmax);
			VecLoad(xmax, viewer_xmax);
			//VecView(xmax, PETSC_VIEWER_STDOUT_WORLD);
			
			
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/U_499.dat", FILE_MODE_READ, &viewer_U);
			VecLoad(U, viewer_U);
			//VecView(U, PETSC_VIEWER_STDOUT_WORLD);

			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/L_499.dat", FILE_MODE_READ, &viewer_L);
			VecLoad(L, viewer_L);
			//VecView(L, PETSC_VIEWER_STDOUT_WORLD);

			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/xo2_499.dat", FILE_MODE_READ, &viewer_xo2);
			VecLoad(xo2, viewer_xo2);
			//VecView(xo2, PETSC_VIEWER_STDOUT_WORLD);



			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "HS_36/xo1_499.dat", FILE_MODE_READ, &viewer_xo1);
			VecLoad(xo1, viewer_xo1);
			//VecView(xo1, PETSC_VIEWER_STDOUT_WORLD);
			PetscViewerDestroy(&viewer_xmin);
			PetscViewerDestroy(&viewer_xmax);
			PetscViewerDestroy(&viewer_U);
			PetscViewerDestroy(&viewer_L);
			PetscViewerDestroy(&viewer_xo2);
			PetscViewerDestroy(&viewer_xo1);
			

*/

// initialize MMA with restart
itr = 0;
mma= new MMA(nGlobalDesignVar,m,itr,xo1,xo2,U,L,aMMA,cMMA,dMMA);   //------------------------------(3)

// initialize MMA
//AllocateMMAwithRestart(m, x_red, xPhys_red, &itr, &mma);

// Initialize constraint and sensitivities

Vec gx_i;             // vec of constraint value for each element

Vec dgdrho;           // vec for sensitivity of constarint function
Vec *dgdx;             // sensitivity of constraints

Vec *dgdx_red;



// set up constaints for MMA
PetscCalloc1(m, &gx);
PetscCalloc1(m, &dgdx);
PetscCalloc1(m, &dgdx_red);



// Total volume of structure
Vec Vtotal_vec;
PetscScalar a_i=dx*dy;
VecDuplicate(x, &Vtotal_vec);
VecSet(Vtotal_vec, a_i);
PetscScalar Vtotal;
VecSum(Vtotal_vec, &Vtotal);





//Vec xmin, xmax, xold;
//VecDuplicate(x_red, &xmin);
//VecDuplicate(x_red, &xmax);
//VecDuplicate(x_red, &xold);
//VecCopy( x_red, xold);




PetscScalar fscale;

Vec df;
VecDuplicate(x, &df);

PetscInt optit=0;




double t1, t2;
while (itr< maxItr && ch>0.01){

itr++;

optit = itr;
t1=MPI_Wtime();

// FEA and sesnitivity of objective function
physics(xPhys, hole_elem, nelx,nelh_x, nely,nelh_y, time_spec, a, Obj_val,dfdrho, elas_da, prop_da, optit) ;
//VecView(dfdrho, PETSC_VIEWER_STDOUT_WORLD);

// Compute
Obj_val = Obj_val;

// Constraints and their sensitivities
func_cons(nelx, nely, l, w, volfrac, dfdrho, elas_da, prop_da, Vlim, x, gx_i, dgdrho);


VecCopy(dfdrho, df);
VecSum(gx_i, &gx_I);

// Constraint value
gxval=(gx_I/Vtotal-Vlim);


gx[0] = gxval;  // #1 constraint on the amount of SMP material 1
dgdx[0] = dgdrho; // #2 constraint sensitivity



// change sign of the  objective sensitivities 
VecScale(dfdrho, 1);


// Filter the sensitivities
Vec xtmp;
VecDuplicate(x,&xtmp);

//dfdx
//VecView(dfdrho, PETSC_VIEWER_STDOUT_WORLD);
MatMult(H, dfdrho, xtmp);
VecPointwiseDivide(dfdrho, xtmp,Hs);



//dgdx
//MatMult(H, dgdx[0], xtmp);
//VecPointwiseDivide(dgdx[0], xtmp,Hs);




VecDestroy(&xtmp);


// Set the outer limits

mma->SetOuterMovelimit(Xmin, Xmax, movlim, x_red, xmin, xmax);



// reduce the full sized sensitivity vectors
ISDestroy(&is_red);
Reduced_dens(prop_da, hole_elem, is_red, dfdrho, dgdrho, &dfdrho_red, &dgdrho_red);
dgdx_red[0] = dgdrho_red;

mma->Update(x_red, dfdrho_red, gx, dgdx_red, xmin, xmax);  // generates and solves the sub problem

//VecView(x, PETSC_VIEWER_STDOUT_WORLD);
mma->KKTresidual(x_red, dfdrho_red, &Obj_val, dgdx_red, xmin, xmax, &norm2, &normInf);

// Calculate design change
ch=mma->DesignChange(x_red, xold);

// Update x 
 VecScatterCreate(x_red,NULL,x,is_red,&scat_red);

ierr = VecScatterBegin(scat_red,x_red,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
 ierr = VecScatterEnd(scat_red,x_red,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
VecScatterDestroy(&scat_red);



// Filter the new design 
ierr = MatMult(H,x,xPhys); CHKERRQ(ierr);
VecPointwiseDivide(xPhys,xPhys,Hs);


//VecView(xPhys, PETSC_VIEWER_STDOUT_WORLD);

t2=MPI_Wtime();
PetscPrintf(PETSC_COMM_WORLD, " It.: %i , Obj.: %f, g[0]: %f ,ch.%f, norm2:, %g, normInf:, %g, time: %f \n", itr, Obj_val, gx[0], ch, norm2, normInf, t2-t1); 

//Get vectors for writing restart files
mma->Restart(xo1,xo2,U,L);


// Write design points to text file

PetscViewerASCIIOpen( PETSC_COMM_WORLD, "xPhys.txt",&viewer_text);
PetscViewerPushFormat(viewer_text, PETSC_VIEWER_ASCII_MATLAB);
VecView(xPhys, viewer_text);
PetscViewerPopFormat(viewer_text);


//======================== write to  binary file =======================================
// Write individual iterations to binary file
MPI_Barrier(PETSC_COMM_WORLD);
char buf[256], bufxmin[256], bufxmax[256], bufU[256], bufL[256], bufxo2[256], bufxo1[256] ;

//PetscViewerCreate(PETSC_COMM_WORLD, &viewer_end);
//PetscViewerSetType(viewer_end, PETSCVIEWERBINARY);
PetscSNPrintf(buf, 256, "HS_0/DesPt_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,buf, FILE_MODE_WRITE, &viewer_x);
//PetscViewerFileSetMode(viewer_end, FILE_MODE_WRITE);
//PetscViewerFileSetName(viewer_end, buf);
VecView(x, viewer_x);

// write xmin point

PetscSNPrintf(bufxmin, 256, "HS_0/xmin_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufxmin, FILE_MODE_WRITE, &viewer_xmin);
VecView(xmin, viewer_xmin);

// write xmax
PetscSNPrintf(bufxmax, 256, "HS_0/xmax_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufxmax, FILE_MODE_WRITE, &viewer_xmax);
VecView(xmax, viewer_xmax);

//write U
PetscSNPrintf(bufU, 256, "HS_0/U_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufU, FILE_MODE_WRITE, &viewer_U);
VecView(U, viewer_U);

//write L
PetscSNPrintf(bufL, 256, "HS_0/L_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufL, FILE_MODE_WRITE, &viewer_L);
VecView(L, viewer_L);

//write xo1
PetscSNPrintf(bufxo1, 256, "HS_0/xo1_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufxo1, FILE_MODE_WRITE, &viewer_xo1);
VecView(xo1, viewer_xo1);


//write xo2
PetscSNPrintf(bufxo2, 256, "HS_0/xo2_%d.dat", optit);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,bufxo2, FILE_MODE_WRITE, &viewer_xo2);
VecView(xo2, viewer_xo2);



//====== Clean-up optimization iterations =======
VecDestroy(&dfdrho);
VecDestroy(&dfdrho_red);


VecDestroy(&dgdrho);
VecDestroy(&dgdrho_red);

VecDestroy(&gx_i);
ISDestroy(&is_red);
PetscViewerDestroy(&viewer_xo1);
PetscViewerDestroy(&viewer_text);
PetscViewerDestroy(&viewer_x);
PetscViewerDestroy(&viewer_xmin);
PetscViewerDestroy(&viewer_xmax);
PetscViewerDestroy(&viewer_xo2);
PetscViewerDestroy(&viewer_U);
PetscViewerDestroy(&viewer_L);
}

//VecView(xPhys, PETSC_VIEWER_STDOUT_WORLD);

//PetscViewerBinaryOpen( PETSC_COMM_WORLD, "Restart_8.dat", FILE_MODE_WRITE, &viewer);
//PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//VecView(xPhys, viewer);
//VecView(x, viewer);
//PetscViewerPopFormat(viewer);


// clean up

VecDestroy(&xPhys);
VecDestroy(&xPhys_red);

VecDestroy(&x);
VecDestroy(&x_red);
VecDestroy(&xo1);
VecDestroy(&xo2);
VecDestroy(&L);
VecDestroy(&U);
VecDestroy(&hole_elem);

//VecDestroy(&dfdrho);
//VecDestroy(&dfdrho_red);


//VecDestroy(&dgdrho);
//VecDestroy(&dgdrho_red);

//VecDestroy(&gx_i);

DMDestroy(&elas_da);
DMDestroy(&prop_da);

MatDestroy(&H);
VecDestroy(&Hs);

VecDestroy(&xmin);
VecDestroy(&xmax);
VecDestroy(&xold);

VecDestroy(&df);

VecDestroy(&Vtotal_vec);

PetscFree(gx);

PetscFree(dgdx);
PetscFree(dgdx_red);

ISDestroy(&is_red);

delete mma;

//PetscViewerDestroy(&viewer);


PetscFinalize();

return 0;


}

//================================================================================================================================================

PetscErrorCode  func_cons(PetscInt nelx, PetscInt nely, PetscScalar l, PetscScalar w, PetscScalar volfrac, Vec &dfdrho, DM elas_da, DM prop_da, PetscScalar Vlim, Vec &x, Vec &gx_i, Vec &dgdrho){

PetscErrorCode ierr;
PetscInt         si, sj, ei, ej, nx, ny, nel_p, nen,size_x, sizeg_x, sizeg_y, size_y,ctr_x,ctr_y;
 const PetscInt   *el_indices;
 ISLocalToGlobalMapping l2g;


VecDuplicate(dfdrho, &dgdrho);
VecDuplicate(dfdrho, &gx_i);

PetscScalar dx, dy;

dx   = l/((PetscReal)(nelx));
dy   = w/((PetscReal)(nely));

PetscScalar a_i=dx*dy;

	Vec localrho;

    	ierr = DMGetLocalVector(prop_da, &localrho); CHKERRQ(ierr);
    	ierr = DMGlobalToLocalBegin(prop_da, x, INSERT_VALUES, localrho); CHKERRQ(ierr);
    	ierr = DMGlobalToLocalEnd(prop_da, x, INSERT_VALUES, localrho); CHKERRQ(ierr);

	PetscScalar *localrho_array;
    	VecGetArray(localrho, &localrho_array);

	DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // w/o the ghost cells
     	DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of nodes
     	DMDAGetCorners(elas_da, NULL, NULL,0,&size_x,&size_y,NULL); // gets values w/o including ghost cells
     	DMDAGetGhostCorners(elas_da, NULL,NULL,0, &sizeg_x,&sizeg_y,NULL);


	PetscInt *dfdxIdx = NULL;
    	PetscScalar *dfdxVal = NULL;
	PetscScalar *gx_val= NULL;
    	// allocate arrays for df/dx_i values and indices
    	DMDAGetElements(elas_da,&nel_p,&nen,&el_indices);

    	ierr = PetscCalloc1(nel_p, &dfdxIdx); CHKERRQ(ierr);
    	ierr = PetscCalloc1(nel_p, &dfdxVal); CHKERRQ(ierr);
	ierr = PetscCalloc1(nel_p, &gx_val); CHKERRQ(ierr);

PetscInt elem=0;
ctr_y=0;
for (ej = sj; ej < sj+ny; ej++) {
ctr_x=0;
    for (ei = si; ei < si+nx; ei++) {
	PetscScalar val=localrho_array[elem];
	gx_val[elem]=a_i*(1-val);
 	dfdxVal[elem] = -a_i;
        dfdxIdx[elem] = elem;

elem=elem+1;
ctr_x=ctr_x+1;
    }
ctr_y=ctr_y+1;
}

	DMGetLocalToGlobalMapping(prop_da, &l2g);
    	VecSetLocalToGlobalMapping(dgdrho, l2g);
    	ierr = VecSetValuesLocal(dgdrho, nel_p, dfdxIdx, dfdxVal, INSERT_VALUES); CHKERRQ(ierr);
    	ierr = VecAssemblyBegin(dgdrho); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(dgdrho); CHKERRQ(ierr);

	ierr = VecSetValuesLocal(gx_i, nel_p, dfdxIdx, gx_val, INSERT_VALUES); CHKERRQ(ierr);
    	ierr = VecAssemblyBegin(gx_i); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(gx_i); CHKERRQ(ierr);
// clean up
PetscFree(dfdxIdx);
PetscFree(dfdxVal);
PetscFree(gx_val);
VecRestoreArray(localrho, &localrho_array);
ierr = DMRestoreLocalVector(prop_da, &localrho); 
PetscFunctionReturn(ierr);
}

//===================================================================================================================================================================

PetscErrorCode  AllocateMMAwithRestart(PetscInt m, Vec &x, Vec &xPhys, PetscInt *itr, MMA **mma)  {

		

	PetscErrorCode ierr = 0;

	Vec xo1, xo2, U, L;

	// Set MMA parameters (for multiple load cases)
	PetscScalar aMMA[m];
	PetscScalar cMMA[m];
	PetscScalar dMMA[m];
	for (PetscInt i=0;i<m;i++){
	    aMMA[i]=0.0;
	    dMMA[i]=0.0;
	    cMMA[i]=1000.0;
	}

	// Check if restart is desired
	PetscBool restart = PETSC_TRUE; // DEFAULT USES RESTART
	PetscBool flip = PETSC_TRUE;     // BOOL to ensure that two dump streams are kept
	PetscBool onlyLoadDesign = PETSC_FALSE; // Default restarts everything

	// Get inputs
	PetscBool flg;
	std::string filename00, filename00Itr, filename01, filename01Itr;
	char filenameChar[PETSC_MAX_PATH_LEN];
	PetscOptionsGetBool(NULL,NULL,"-restart",&restart,&flg);
	PetscOptionsGetBool(NULL,NULL,"-onlyLoadDesign",&onlyLoadDesign,&flg);

	if (restart) {
	  ierr = VecDuplicate(x,&xo1); CHKERRQ(ierr);
	  ierr = VecDuplicate(x,&xo2); CHKERRQ(ierr);
	  ierr = VecDuplicate(x,&U); CHKERRQ(ierr);
	  ierr = VecDuplicate(x,&L); CHKERRQ(ierr);
	}
	
	// Determine the right place to write the new restart files
	std::string filenameWorkdir = "./";
	PetscOptionsGetString(NULL,NULL,"-workdir",filenameChar,sizeof(filenameChar),&flg);
	if (flg){
		filenameWorkdir = "";
		filenameWorkdir.append(filenameChar);
	}
	filename00 = filenameWorkdir;
	filename00Itr = filenameWorkdir;
	filename01 = filenameWorkdir;
	filename01Itr = filenameWorkdir;

	filename00.append("/Restart00.dat");
	filename00Itr.append("/Restart00_itr_f0.dat");
	filename01.append("/Restart01.dat");
	filename01Itr.append("/Restart01_itr_f0.dat");

	// Where to read the restart point from
	std::string restartFileVec = ""; // NO RESTART FILE !!!!!
	std::string restartFileItr = ""; // NO RESTART FILE !!!!!

	PetscOptionsGetString(NULL,NULL,"-restartFileVec",filenameChar,sizeof(filenameChar),&flg);
	if (flg) {
	   restartFileVec.append(filenameChar);
	}
	PetscOptionsGetString(NULL,NULL,"-restartFileItr",filenameChar,sizeof(filenameChar),&flg);
	if (flg) {
		restartFileItr.append(filenameChar);
	}

	// Which solution to use for restarting
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Continue from previous iteration (-restart): %i \n",restart);
	PetscPrintf(PETSC_COMM_WORLD,"# Restart file (-restartFileVec): %s \n",restartFileVec.c_str());
	PetscPrintf(PETSC_COMM_WORLD,"# Restart file (-restartFileItr): %s \n",restartFileItr.c_str());
	PetscPrintf(PETSC_COMM_WORLD,"# New restart files are written to (-workdir): %s (Restart0x.dat and Restart0x_itr_f0.dat) \n",filenameWorkdir.c_str());

	// Check if files exist:
	
	PetscBool vecFile = fexists(restartFileVec);
	if (!vecFile) { PetscPrintf(PETSC_COMM_WORLD,"File: %s NOT FOUND \n",restartFileVec.c_str()); }
	PetscBool itrFile = fexists(restartFileItr);
	if (!itrFile) { PetscPrintf(PETSC_COMM_WORLD,"File: %s NOT FOUND \n",restartFileItr.c_str()); }
	
	// Read from restart point
	
	PetscInt nGlobalDesignVar;
	VecGetSize(x,&nGlobalDesignVar); // ASSUMES THAT SIZE IS ALWAYS MATCHED TO CURRENT MESH

	if (restart && vecFile && itrFile){
		
		PetscViewer view;
		// Open the data files 
		ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,restartFileVec.c_str(),FILE_MODE_READ,&view);	
				
		VecLoad(x,view);
		VecLoad(xPhys,view);
		VecLoad(xo1,view);
		VecLoad(xo2,view);
		VecLoad(U,view);
		VecLoad(L,view);
		PetscViewerDestroy(&view);
		
		// Read iteration and fscale
		PetscScalar fscale;
		std::fstream itrfile(restartFileItr.c_str(), std::ios_base::in);
		itrfile >> itr[0];
		itrfile >> fscale;
		
		
		// Choose if restart is full or just an initial design guess
		if (onlyLoadDesign){
			PetscPrintf(PETSC_COMM_WORLD,"# Loading design from file: %s \n",restartFileVec.c_str());
			*mma= new MMA(nGlobalDesignVar,m,x, aMMA, cMMA, dMMA);
		}
		else {
			PetscPrintf(PETSC_COMM_WORLD,"# Continue optimization from file: %s \n",restartFileVec.c_str());
			*mma= new MMA(nGlobalDesignVar,m,*itr,xo1,xo2,U,L,aMMA,cMMA,dMMA);
		}

		PetscPrintf(PETSC_COMM_WORLD,"# Successful restart from file: %s and %s \n",restartFileVec.c_str(),restartFileItr.c_str());

	}
	else {

		*mma=new MMA(nGlobalDesignVar,m,x,aMMA,cMMA,dMMA);
		
	}  

	

// clean up
VecDestroy(&xo1);
VecDestroy(&xo2);
VecDestroy(&U);
VecDestroy(&L);

return ierr;
} 

//==========================================================================================================================================================
inline PetscBool fexists(const std::string& filename) {
	      std::ifstream ifile(filename.c_str());
	      if (ifile) {
		return PETSC_TRUE;
	      }
	      return PETSC_FALSE;
}

//============================================================================================================================================================
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

/*

 PetscErrorCode physics( PetscInt nelx, PetscInt nely, PetscScalar volfrac, PetscScalar time_spec, PetscInt a, PetscScalar &Obj_val, Vec &dfdrho) 
{

  PetscMPIInt      rank,size;
  PetscInt         nel, stw, nel_p, nen, i, mxl, myl, cpu_x, cpu_y, *lx=NULL, *ly=NULL, prop_dof, prop_stencil_width, M, N, gxs, gys,gxm, gym , neq, nnodes, e, iter;
  PetscInt         nnodesx,nnodesy, sex, sey,lower_node_x, lower_node_y,size_x,size_y,ein,ltogmsize,len_f, size_rho,size_test=16,size_dfdx,len_p,set_dof;
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE;
  DM               elas_da,prop_da;
  DM               cdm,vel_cda;
  DMDACoor2d       **local_coords;
  Vec              local,global;
  PetscScalar      value, one=1.0,penal=3.0;
  const PetscInt   *el_indices=NULL,*g_idx=NULL,*idx=NULL,*local_indices=NULL;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  PetscReal         dx, dy;
  Vec               coords,U,Fint,X,local_F,vec_indices,Fext_petsc,Residual_f,delta_U_f,U_f,Fext_f,Fext,Fint_f;
  Mat               KTAN, KFF,*KFF_dup_pt=NULL, KFF_dup, dFint_du, KTAN_dup;
  IS                is;
  MatNullSpace      matnull;
  ElasticityDOF     **ff;
  KSP                ksp_E;
  ISLocalToGlobalMapping ltogm;
  PetscReal      norm,norm_0, *disp_val=NULL;
  //Vec             U_hist[10];
   Vec             *U_hist=NULL, xPhys, x, dFint_dx,dfint_dx_dup;

// Define the variables
 
 nel=nelx*nely;
 stw=1;
 neq=(nelx+1)*(nely+1)*ndof;
 iter=0;

 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  VecScatter     scat;



// Setup mesh

 ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nelx+1,nely+1,PETSC_DECIDE,PETSC_DECIDE,ndof,stw,NULL,NULL,&elas_da);CHKERRQ(ierr);
  DMSetMatType(elas_da,MATAIJ);  
  ierr = DMSetUp(elas_da);CHKERRQ(ierr);
  DMDASetUniformCoordinates(elas_da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  DMDAGetGhostCorners(elas_da, &gxs, &gys, NULL, &gxm,&gym,NULL);// gxs and gys are local indices including ghost gxm and gym are no. of grid points in each direction on each processor


// SET DMDA FOR DENSITY PROPERTIES
// each element needs to have a particular density value //

  DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);
  DMDAGetElementOwnershipRanges2d(elas_da,&lx,&ly);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nelx,nely,cpu_x,cpu_y,1,0,lx,ly,&prop_da);CHKERRQ(ierr);
  ierr = DMSetUp(prop_da);CHKERRQ(ierr);
  PetscFree(lx);
  PetscFree(ly);

 dx   = 1.0/((PetscReal)(nelx));
 dy   = 1.0/((PetscReal)(nely));

  DMDASetUniformCoordinates(prop_da,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dy,1.0-0.5*dy,0.0,1.0); 
 
 
  DMCreateGlobalVector(prop_da, &xPhys);                           // define global elemental density vector
  VecGetSize(xPhys, &size_rho);
  VecDuplicate(xPhys,&x);
  VecSet(x, volfrac);
  VecSet(xPhys,volfrac);

//DMCreateGlobalVector(prop_da, &rho_global);
//for(PetscInt i=0;i<size_rho;i++){
//PetscInt set_at=i;
//PetscScalar set_val=i; 
//VecSetValues(rho, 1, &set_at, &set_val, INSERT_VALUES);
//}
//VecAssemblyBegin(rho);
//VecAssemblyEnd(rho);
 //ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
//VecView(rho,PETSC_VIEWER_STDOUT_WORLD);

//DMDANaturalToGlobalBegin(prop_da, rho, INSERT_VALUES, rho_global);
//DMDANaturalToGlobalEnd(prop_da, rho, INSERT_VALUES, rho_global);
//VecView(rho_global,PETSC_VIEWER_STDOUT_WORLD);




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


//for(PetscMPIInt i=0;i<neq;i++){
//PetscScalar value_set=i;
//PetscMPIInt st_at=i;
//VecSetValues(U, 1, &st_at, &value_set, INSERT_VALUES);
//}
//VecAssemblyBegin(U);
//VecAssemblyEnd(U);




//VecView(U,PETSC_VIEWER_STDOUT_WORLD);
//assembly(elas_da, rank, iter, T, T_hist, Fint, KTAN, U, U_hist);         //initilize with assembly function


//        create KSP solver
KSPCreate(PETSC_COMM_WORLD,&ksp_E);
KSPSetTolerances(ksp_E,1.e-10,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT);
KSPSetOptionsPrefix(ksp_E,"elas_");  

// enforce bc's
DMGetLocalToGlobalMapping(elas_da,&ltogm);
ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);
ISLocalToGlobalMappingGetSize(ltogm,&ltogmsize);

// Apply boundary conditions
//ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs
//MatCreateVecs(KFF,NULL,&Fext_f);
//VecZeroEntries(Fext_f);

//VecDuplicate(Fint_f,&Residual_f);
//VecDuplicate(Fint_f,&U_f);
//VecZeroEntries(U_f);
//Delta U_f vector
//VecDuplicate(Fint_f,&delta_U_f);

//ISGetSize(is,&len_f);



// SMP cycle properties
//PetscScalar time_spec=45;
PetscReal C_R=-0.25;
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

PetscScalar tot_its=time_spec/delta_t; // total number of iterations

// define and declare history variables
PetscMalloc1(tot_its, &U_hist);
for(i=0;i<tot_its;i++){
MatCreateVecs(KTAN, NULL,&U_hist[i]);
}



PetscMalloc1(tot_its, &T_hist);
PetscMalloc1(tot_its, &KT_step); // define the number of storage cells needed for KT
PetscMalloc1(tot_its, &dFdx_step); // define the number of storage cells needed for dfint_dx
PetscMalloc1(tot_its, &KTAN_step); // define the number of storage cells needed for KTAN_step


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


//111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
//******************************************************* N-R iterations *******************************************************
// Cooling with deformations
if(time_spec>t_loading_start)
{
   while(t+delta_t<=PetscMin(t_loading_fin,time_spec))
    {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time

VecCopy(U,U_hist[iter-1]);                       // initial value of u history
T_hist[iter-1]=T;                               // store history values
T=T+C_R*delta_t;                                // update temperature

//PetscPrintf(PETSC_COMM_WORLD,"current Temp %f T_hist %f \n",T, T_hist[iter-1]);
//VecView(U_hist[iter-1],PETSC_VIEWER_STDOUT_WORLD);

assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function

// Apply boundary conditions
ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs



MatCreateVecs(KFF,NULL,&Fext_f);
VecZeroEntries(Fext_f);

VecDuplicate(Fint_f,&Residual_f);


VecDuplicate(Fint_f,&U_f);
VecZeroEntries(U_f);


//Delta U_f vector
VecDuplicate(Fint_f,&delta_U_f);
ISGetSize(is,&len_f);

 len_p=(nelx+1)*(nely+1)*2-len_f;
 //set_dof=(nelx+1)*(nely+1)*2-len_p-1; // the free dof
set_dof=a-len_p-1; // the free dof
// define Fext vector
//PetscScalar Fext_val=0.2*iter;
PetscScalar Fext_val=0.05;
VecSetValues(Fext_f, 1, &set_dof, &Fext_val, INSERT_VALUES);

VecAssemblyBegin(Fext_f);
VecAssemblyEnd(Fext_f);

//PetscPrintf(PETSC_COMM_WORLD,"nelx [%d] nely [%d] load at [%d] \n", nelx, nely, set_dof);

//calculate residual
VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

//calculate norm value
VecNorm(Residual_f,NORM_2,&norm);
 

while (norm>1e-06)
{

KSPSetOperators(ksp_E,KFF,KFF);

		
		KSPSolve(ksp_E,Residual_f,delta_U_f);

                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);
                //calculate KTAN and Fint
            
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function

		VecDestroy(&Fint_f);
		MatDestroy(&KFF);
		ISDestroy(&is);

                // apply BC FORM KFF AND Fintf
                ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs
		
              
               //Calculate residual
                VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

                VecNorm(Residual_f,NORM_2,&norm);
              

}

// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Fint_f);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
VecDestroy(&delta_U_f);
VecDestroy(&Fext_f);
ISDestroy(&is);
  }
}



//2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
PetscScalar t_relax_start=t_loading_fin;
PetscScalar relax_duration=10;
PetscScalar t_relax_fin=t_relax_start+relax_duration;


// Relaxation cycle
if(time_spec>t_relax_start)
{
   while(t+delta_t<=PetscMin(t_relax_fin,time_spec))
    {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time

VecCopy(U,U_hist[iter-1]);                       // initial value of u history
T_hist[iter-1]=T;                               // store history values
//T=T+C_R*delta_t;                                // update temperature
T=T;
//PetscPrintf(PETSC_COMM_WORLD,"current Temp %f T_hist %f \n",T, T_hist[iter-1]);
//VecView(U_hist[iter-1],PETSC_VIEWER_STDOUT_WORLD);

assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function

// Apply boundary conditions
ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs



MatCreateVecs(KFF,NULL,&Fext_f);
VecZeroEntries(Fext_f);

VecDuplicate(Fint_f,&Residual_f);


VecDuplicate(Fint_f,&U_f);
VecZeroEntries(U_f);


//Delta U_f vector
VecDuplicate(Fint_f,&delta_U_f);
ISGetSize(is,&len_f);

 len_p=(nelx+1)*(nely+1)*2-len_f;
 //set_dof=(nelx+1)*(nely+1)*2-len_p-1; // the free dof
set_dof=a-len_p-1; // the free dof
// define Fext vector
//PetscScalar Fext_val=0.2*iter;
PetscScalar Fext_val=0.00;
VecSetValues(Fext_f, 1, &set_dof, &Fext_val, INSERT_VALUES);

VecAssemblyBegin(Fext_f);
VecAssemblyEnd(Fext_f);

//PetscPrintf(PETSC_COMM_WORLD,"nelx [%d] nely [%d] load at [%d] \n", nelx, nely, set_dof);

//calculate residual
VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

//calculate norm value
VecNorm(Residual_f,NORM_2,&norm);
 

while (norm>1e-06)
{

KSPSetOperators(ksp_E,KFF,KFF);

		
		KSPSolve(ksp_E,Residual_f,delta_U_f);

                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);
                //calculate KTAN and Fint
            
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function

		VecDestroy(&Fint_f);
		MatDestroy(&KFF);
		ISDestroy(&is);

                // apply BC FORM KFF AND Fintf
                ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs
		
              
               //Calculate residual
                VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

                VecNorm(Residual_f,NORM_2,&norm);
              

}

// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Fint_f);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
VecDestroy(&delta_U_f);
VecDestroy(&Fext_f);
ISDestroy(&is);
  }
}



//33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
// Heating Phase
PetscInt iter_heat=iter+1;
PetscReal H_R=0.25;
PetscScalar t_heat_start=t_relax_fin;
PetscScalar t_heat_fin=t_heat_start+((Tmax-Tmin)/PetscAbsReal(H_R));

if(time_spec>t_heat_start)
{
   while(t+delta_t<=PetscMin(t_heat_fin,time_spec))
    {

iter=iter+1;                                     // update iteration number
t=t+delta_t;                                     //update time

VecCopy(U,U_hist[iter-1]);                       // initial value of u history
T_hist[iter-1]=T;                               // store history values
T=T+H_R*delta_t;                                // update temperature


//PetscPrintf(PETSC_COMM_WORLD,"current Temp %f T_hist %f \n",T, T_hist[iter-1]);
//VecView(U_hist[iter-1],PETSC_VIEWER_STDOUT_WORLD);

assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);         //initilize with assembly function
//PetscPrintf(PETSC_COMM_WORLD,"current Temp %f T_hist %f \n",T, T_hist[iter-1]);
//VecView(Fint,PETSC_VIEWER_STDOUT_WORLD);

// Apply boundary conditions
ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs



MatCreateVecs(KFF,NULL,&Fext_f);
VecZeroEntries(Fext_f);

VecDuplicate(Fint_f,&Residual_f);


VecDuplicate(Fint_f,&U_f);
VecZeroEntries(U_f);


//Delta U_f vector
VecDuplicate(Fint_f,&delta_U_f);
ISGetSize(is,&len_f);

 len_p=(nelx+1)*(nely+1)*2-len_f;
 //set_dof=(nelx+1)*(nely+1)*2-len_p-1; // the free dof
set_dof=a-len_p-1; // the free dof
// define Fext vector
//PetscScalar Fext_val=0.2*iter;
PetscScalar Fext_val=0.00;
VecSetValues(Fext_f, 1, &set_dof, &Fext_val, INSERT_VALUES);

VecAssemblyBegin(Fext_f);
VecAssemblyEnd(Fext_f);

//PetscPrintf(PETSC_COMM_WORLD,"nelx [%d] nely [%d] load at [%d] \n", nelx, nely, set_dof);

//calculate residual
VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

//calculate norm value
VecNorm(Residual_f,NORM_2,&norm);
 

while (norm>1e-06)
{

KSPSetOperators(ksp_E,KFF,KFF);

		
		KSPSolve(ksp_E,Residual_f,delta_U_f);

                VecAXPY(U_f,1,delta_U_f);
		
                DMDABCApplyCompression(elas_da,rank,NULL,U);

                VecScatterCreate(U_f,NULL,U,is,&scat);

  		ierr = VecScatterBegin(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  		ierr = VecScatterEnd(scat,U_f,U,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
               VecScatterDestroy(&scat);
                //calculate KTAN and Fint
            
                assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,0);  // assembly function

		VecDestroy(&Fint_f);
		MatDestroy(&KFF);
		ISDestroy(&is);

                // apply BC FORM KFF AND Fintf
                ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs
		
              
               //Calculate residual
                VecWAXPY(Residual_f,-1,Fint_f,Fext_f);

                VecNorm(Residual_f,NORM_2,&norm);
              

}

// calculate dfint_dx for sensitivity analysis
 assembly(elas_da, prop_da, rank, iter, delta_t, xPhys, penal, T, T_hist, Fint, dFint_dx, KTAN, U, U_hist,1);  // assembly function

// store values for all the iterations
 MatDuplicate(KFF,MAT_COPY_VALUES, &KFF_dup);
MatDuplicate(KTAN, MAT_COPY_VALUES, &KTAN_dup);
 VecDuplicate(dFint_dx, &dfint_dx_dup);  
KT_step[iter-1]=KFF_dup;
KTAN_step[iter-1]=KTAN_dup;
VecCopy(dFint_dx, dfint_dx_dup);
dFdx_step[iter-1]=dfint_dx_dup;


PetscPrintf(PETSC_COMM_WORLD,"iter # %d ||Norm of error %g||time %f\n",iter,(double)norm,t);

MatDestroy(&KFF);
VecDestroy(&Fint_f);
VecDestroy(&Residual_f);
VecDestroy(&U_f);
VecDestroy(&delta_U_f);
VecDestroy(&Fext_f);
ISDestroy(&is);

  }
}




//VecView(U, PETSC_VIEWER_STDOUT_WORLD);
PetscPrintf(PETSC_COMM_WORLD,"%%%%%%%%  FEA COMPLETED %%%%%%%%%%\n");

PetscInt low_proc, high_proc;
PetscScalar leaf_val;
VecGetOwnershipRange(U, &low_proc,&high_proc); 
PetscInt get_from =a-1;
if((get_from>=low_proc)&&(get_from<high_proc)){VecGetValues(U, 1, &get_from, &leaf_val);}



//printf("obj_val[%f] \n", leaf_val);
MPI_Barrier(PETSC_COMM_WORLD);
MPI_Reduce(&leaf_val, &Obj_val,1, MPIU_SCALAR, MPI_SUM, 0, PETSC_COMM_WORLD); 

// view final displacement vector
//VecView(U, PETSC_VIEWER_STDOUT_WORLD);

//VecDestroy(&dfint_dx_dup);


for(i=0;i<tot_its;i++){
VecDestroy(&U_hist[i]);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//										SENSITIVITY ANALYSIS 
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//DECLARE DFDRHO


ApplyBC(elas_da,rank, KTAN, Fint, &is,&KFF, &Fint_f); //is gives the free d.o.fs



//Vec dfdrho;
DMCreateGlobalVector(prop_da, &dfdrho); 


Vec lambda_test;
VecDuplicate(Fint_f, &lambda_test);
Vec L_test;
VecDuplicate(Fint_f, &L_test);
VecSet(L_test, 0.0);
PetscInt add_at_index_test=set_dof;
VecSetValues(L_test, 1, &add_at_index_test, &one, INSERT_VALUES);
VecAssemblyBegin(L_test);
VecAssemblyEnd(L_test);

KSP ksp_l;
KSPCreate(PETSC_COMM_WORLD,&ksp_l);

KSPSetOptionsPrefix(ksp_l,"lambda_"); 
KSPSetType(ksp_l, KSPGMRES); 
KSPSetTolerances(ksp_l,1.e-15,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
KSPSetOperators(ksp_l,KFF,KFF);
KSPConvergedReason converged;
KSPSolve(ksp_l,L_test,lambda_test);
KSPReasonView(ksp_l, PETSC_VIEWER_STDOUT_WORLD);
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
PetscMalloc1(iter, &lambda_steps);

//Store the lambda of last step
Vec lambda_m_f;
VecDuplicate(Fint_f, &lambda_m_f);
VecCopy(lambda_test, lambda_m_f);
lambda_iter[iter-1]=lambda_m_f;


//VecView(lambda_iter[iter-1], PETSC_VIEWER_STDOUT_WORLD);

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
KSPSetType(ksp_lambda, KSPGMRES); 
KSPSetTolerances(ksp_lambda,1.e-15,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
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
for(PetscInt i=iter-2;i>-1;i=i-1)
{
VecSet(RHS, 0.0);

VecSet(Y, 0.0);
VecSet(lambda_b,0.0);

//std::cout<< "i "<<i  <<std::endl;

	for(PetscInt k=i+1;k<iter;k++){
		MatZeroEntries(DFDUcoup);

           	df_du(elas_da, prop_da, nelx,nely, delta_t,penal, xPhys, k, i, T, T_hist, iter, iter_heat, DFDUcoup);
		ISDestroy(&is);
		
		 VecDestroy(&Fint_f);
                 ApplyBC(elas_da,rank, DFDUcoup, Fint, &is,&DFDUcoup_f, &Fint_f); //is gives the free d.o.fs
		



       		MatMult(DFDUcoup_f, lambda_iter[k], Y);
          
		VecAXPY(RHS, 1.0, Y);
 		MatDestroy(&DFDUcoup_f);
	
	}// k loop



//VecView(RHS, PETSC_VIEWER_STDOUT_WORLD);

//MatCopy(KT_step[i], KTAN_T, SAME_NONZERO_PATTERN);
////MatCreateTranspose(KTAN_step[i], &KTAN_T); 

VecDuplicate(lambda_b, &lambda_copy);

KSPSetOperators(ksp_lambda,KT_step[i],KT_step[i]);

KSPSolve(ksp_lambda,RHS,lambda_b);
KSPReasonView(ksp_lambda, PETSC_VIEWER_STDOUT_WORLD);
KSPGetConvergedReason(ksp_lambda, &converged_lam);

VecScale(lambda_b, -1.0);

if(converged <0){SETERRQ(PETSC_COMM_WORLD,1,"KSP sens solver did not converge\n");}


	
 lambda_iter[i]=lambda_b;
VecCopy(lambda_b, lambda_copy);
lambda_steps[i]=lambda_copy;

//VecView(lambda_iter[i], PETSC_VIEWER_STDOUT_WORLD);


}// i loop

lambda_steps[iter-1]=lambda_m_f;


//******************************************** lambda calculated for all the steps *******************************************************



// CALCULATE DFDRHO
Vec *psi;
PetscCalloc1(iter, &psi);


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


VecDuplicate(lambda_final, &lambda_final_copy);

//lambda_i=lambda_steps[i];
VecCopy(lambda_steps[i], lambda_i);

VecGetArray(lambda_i, &lambda_local);



	for(int j=0;j<free_p_size;j++){
	PetscInt put_idx=freedofs[j];
	PetscScalar in_val=lambda_local[j];
        //if(rank==0) printf("j [%d] put idx [%d] \n",j, put_idx);
	VecSetValues(lambda_final, 1, &put_idx,&in_val, INSERT_VALUES); 


	}

VecAssemblyBegin(lambda_final);
VecAssemblyEnd(lambda_final);

VecCopy(lambda_final, lambda_final_copy);

psi[i]=lambda_final_copy;
VecRestoreArray(lambda_i, &lambda_local);
}
ISRestoreIndices(is, &freedofs);
VecDestroy(&lambda_i);




// SOLVE FOR FINAL VALUE
Vec psi_n;
VecDuplicate(lambda_final,&psi_n);
VecCopy(lambda_final, psi_n);
VecSet(psi_n, 0.0);


Vec dRdx;
VecDuplicate(dFint_dx, &dRdx);
VecCopy(dFint_dx, dRdx);
VecSet(dRdx, 0.0);

for(int i=0;i<iter;i++){

//psi_n=psi[i];
VecCopy(psi[i], psi_n);
//dRdx=dFdx_step[i];
VecCopy(dFdx_step[i], dRdx);
//VecView(psi_n, PETSC_VIEWER_STDOUT_WORLD);
PetscInt time_step=i;

dfdrho_func(elas_da, prop_da,rank, time_step, dfdrho, psi_n, dRdx);

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
        KSPDestroy(&ksp_E);
	VecDestroy(&xPhys);
        VecDestroy(&x);
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
	MatDestroy(&KFF);
        VecDestroy(&Fint_f);
 
	ISLocalToGlobalMappingDestroy(&l2g);
        KSPDestroy(&ksp_E);
//==========================================       	
	//VecDestroy(&dfdrho);
	VecDestroy(&L_test);
	VecDestroy(&lambda_test);
	
	KSPDestroy(&ksp_l);
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
PetscFree(psi);
VecDestroy(&psi_n);
VecDestroy(&dRdx);
//MatDestroy(&KFF_dup);
//MatDestroy(&KTAN_dup);
//VecDestroy(&dfint_dx_dup);

	//ISRestoreIndices(is, &freedofs);
	ISDestroy(&is);

	DMDestroy(&elas_da);
        DMDestroy(&prop_da);


 PetscFunctionReturn(0);

}

//===========================================================================================================================================================================
static PetscErrorCode assembly(DM elas_da, DM prop_da, PetscMPIInt rank,PetscInt iter, PetscScalar delta_t, Vec& xPhys, PetscScalar penal, PetscScalar T, PetscScalar *T_hist, Vec& Fint, Vec& dFint_dx, Mat& KTAN, Vec& U, Vec* U_hist, PetscInt flag)
{

PetscInt         si, sj, ei, ej, nx, ny,e=0,size_x, sizeg_x, sizeg_y, size_y,ctr_x,ctr_y,xp_size, *blk;
DM               cdm, u_cda;
DMDACoor2d       **local_coords;
Vec              coords,local_F,u_local, *u_local_hist=NULL,xp;
ElasticityDOF    **ff;
MatStencil        s_u[8];
PetscErrorCode   ierr;
PetscScalar     *u_local_array=NULL, *xp_array=NULL,  **u_local_hist_array=NULL;

     



  DMDAGetElementsCorners(elas_da,&si,&sj,0);                   // w/o the ghost cells
  DMDAGetElementsSizes(elas_da, &nx,&ny,0);                    // gets the non-overlapping no. of nodes
  DMDAGetCorners(elas_da, NULL, NULL,0,&size_x,&size_y,NULL); // gets values w/o including ghost cells
  DMDAGetGhostCorners(elas_da, NULL,NULL,0, &sizeg_x,&sizeg_y,NULL);

  
//get local displacements for the current step
   DMGetLocalVector(elas_da,&u_local);

   ierr = DMGlobalToLocalBegin(elas_da, U, INSERT_VALUES, u_local); CHKERRQ(ierr);
   ierr = DMGlobalToLocalEnd(elas_da, U, INSERT_VALUES, u_local); CHKERRQ(ierr);
   VecGetArray(u_local,&u_local_array);
   //DMRestoreLocalVector(elas_da,&u_local);

// get local elental density values
  
  DMCreateLocalVector(prop_da, &xp);

   ierr = DMGlobalToLocalBegin(prop_da, xPhys, INSERT_VALUES, xp); CHKERRQ(ierr);
   ierr = DMGlobalToLocalEnd(prop_da, xPhys, INSERT_VALUES, xp); CHKERRQ(ierr);
   VecGetArray(xp,&xp_array);
   VecGetLocalSize(xp,&xp_size); // get size of local density vector
 

    VectorXd rho(xp_size);

for(PetscInt i=0;i<xp_size;i++){
rho(i)=xp_array[i];

}

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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//										ELEMENTAL LOOP 


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ctr_y=0;
for (ej = sj; ej < sj+ny; ej++) {
ctr_x=0;
    for (ei = si; ei < si+nx; ei++) {
       
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
      gp<<0.5774,-0.5774;// gauss quadrature points
      PetscScalar w=1.0;

     // ue=VectorXd::Zero(8);
      int_pt=0;

       // get elemental displacements
if((sizeg_x-size_x)==1){

// for current step
    PetscInt val1, val2;
    val2=(ctr_y+1)*(size_x+1)*2;
   val1=ctr_y*(size_x+1)*2;
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
   val2=(ctr_y+1)*(size_x+1)*2;
   val1=ctr_y*(size_x+1)*2;
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
   val2=(ctr_y+1)*(size_x+1)*2;
   val1=ctr_y*(size_x+1)*2;
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

//if(rank==0){if((ej==0) && (ei==1)){std::cout<< ue_hist <<std::endl;std::cout<<"******************" <<std::endl;std::cout<< ue <<std::endl;}}


        // get coordinates for the element //
   el_coords[NSD*0+0] = local_coords[ej][ei].x;      el_coords[NSD*0+1] = local_coords[ej][ei].y;
   el_coords[NSD*1+0] = local_coords[ej][ei+1].x;    el_coords[NSD*1+1] = local_coords[ej][ei+1].y;
   el_coords[NSD*2+0] = local_coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = local_coords[ej+1][ei+1].y;
   el_coords[NSD*3+0] = local_coords[ej+1][ei].x;    el_coords[NSD*3+1] = local_coords[ej+1][ei].y;

  MatMap el_coords_eig(el_coords, 4,2);        // convert CCur from scalar to MatrixXd
//if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<ue<<std::endl;}}
   
//==================== guass quadrature	 =========================================
  // get the element level quantities
   for(int i=0;i<ngp;i++)
	{for(int j=0;j<ngp;j++){
			eta=gp(i);psi=gp(j);
			int_pt=int_pt+1;                     // integration point number
			shapel(psi,eta,el_coords_eig,DET,B);      // call shape functions
			epsilon=B*ue;                      // calculate small strain

    for(PetscInt k=0;k<iter;k++){
                      ue_star=ue_hist.row(k);
                      epsilon_hist.row(k)=B*ue_star;
                      epsilon_total.row(k)=B*ue_star;
                      }
                        epsilon_total.row(iter)=epsilon;

// if (flag==1) if((i==0)&& (j==0)){if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<epsilon_total<<std::endl;std::cout<<"*************"<<std::endl;std::cout<<epsilon<<std::endl;}}}
 SMP_cycle(rank, epsilon_total, e, T, T_hist, delta_t, rho, penal,iter, int_pt, B, sigma_vis, Dmat_vis, dfint_dx, flag);
 // if (flag==1) if((i==0)&& (j==0)){if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<iter <<epsilon_total<<std::endl;}}}
		       //stiffness(Dmat);
			//sigma=Dmat*epsilon;               // calculate small stress 
                       Dmat=Dmat_vis;
                       sigma=sigma_vis.row(iter-1);  
                       dfint_dx_row=dfint_dx.row(iter-1);


			elFint+=  B.transpose()*sigma*DET*w;
                        elK+=w*B.transpose()*Dmat*B*DET;
			dFdx+=B.transpose()*dfint_dx_row*DET*w;

          }//j loop
}//i loop
//========================================================================================	
//if(rank==0){if((ctr_y==0) && (ctr_x==0)){std::cout<<elFint<<std::endl;}}
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
for(PetscInt i=0;i<iter;i++){
VecDestroy(&u_local_hist[i]);
}

PetscFree(u_local_hist);
PetscFree(u_local_hist_array);
VecDestroy(&local_F);
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
  // enforce bc's //
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

  // --- //

  ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);

  // init the entries to -1 so VecSetValues will ignore them //
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt                 local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];
   // printf("rank [%d] local_id[%d] bc_global_id_j[%d] \n",rank,  local_id, bc_global_ids[j]);

   
    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
  }

  ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
  nbcs = 0;
  if (si == 0) nbcs = ny;

  
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

//==============================================================================================
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
ierr = BCApply_WEST(elas_da,rank,1,0.0,KTAN,x);CHKERRQ(ierr);//apply on y dof

 // define which dofs are not constrained //
  ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&unconstrained);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArray(x,&_x);CHKERRQ(ierr);
  cnt  = 0;
  for (i = 0; i < m; i+=2) {
    PetscReal val1,val2;

    val1 = PetscRealPart(_x[i]);
    val2 = PetscRealPart(_x[i+1]);
//printf("rank [%d] val1[%f] val2[%f] \n",rank, val1,val2);
    if (PetscAbs(val1) < 0.1 && PetscAbs(val2) < 0.1) {
      unconstrained[cnt] = start + i;
      cnt++;
      unconstrained[cnt] = start + i + 1;
      cnt++;
    }
  }

//for(i=0;i<cnt;i++){
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," i [%d]  glo_in[%d]  \n",i,unconstrained[i]);
//}
//PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
  ierr = VecRestoreArray(x,&_x);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,unconstrained,PETSC_COPY_VALUES,&is);CHKERRQ(ierr); // IS containes all the free dofs
  ierr = PetscFree(unconstrained);CHKERRQ(ierr);
  ierr = ISSetBlockSize(is,2);CHKERRQ(ierr);
//ISView(is,PETSC_VIEWER_STDOUT_WORLD);
 //VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  // define correction for dirichlet in the rhs //
  //ierr = MatMult(KTAN,x,Fint);CHKERRQ(ierr);
 // ierr = VecScale(Fint,-1.0);CHKERRQ(ierr);

  // get new matrix //
  ierr = MatCreateSubMatrix(KTAN,is,is,MAT_INITIAL_MATRIX,KFF);CHKERRQ(ierr);
  // get new vector //
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


        val2=(ctr_y+1)*(size_x+1)*2;
       val1=ctr_y*(size_x+1)*2;
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
*/
