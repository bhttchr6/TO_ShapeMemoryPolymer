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

typedef Eigen::Map<MatrixXd,0,Eigen::Stride<1,2> > MatMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, Dynamic, RowMajor> > matMap;
typedef Eigen::Map<Matrix<PetscScalar,Dynamic, 1> > vecMap;
#define ngp 2
#define ndof 2
#define ndim 2

#define plane_strain 1
#define plane_stress 0

#define GAUSS_POINTS   4
#define NSD   2

PetscErrorCode filter(Vec &x, DM elas_da,DM prop_da, PetscScalar rmin, Mat &H, Vec &Hs)
{

		PetscErrorCode ierr;

		DM da_elem;

		//Vec Hs;

		

		PetscInt M,N,P,md,nd,pd; 
		DMBoundaryType bx, by, bz;
		DMDAStencilType stype;
		ierr = DMDAGetInfo(elas_da,NULL,&M,&N,NULL,&md,&nd,NULL,NULL,NULL,&bx,&by,NULL,&stype); CHKERRQ(ierr);

		

		// Find the element size
		Vec lcoor;
		DMGetCoordinatesLocal(elas_da,&lcoor);
		PetscScalar *lcoorp;
		VecGetArray(lcoor,&lcoorp);

		PetscInt nel, nen;
		const PetscInt *necon;
		DMDAGetElements(elas_da,&nel,&nen,&necon);

		PetscScalar dx,dy;
		// Use the first element to compute the dx, dy, dz
		dx = lcoorp[2*necon[0*nen + 1]+0]-lcoorp[2*necon[0*nen + 0]+0];
		dy = lcoorp[2*necon[0*nen + 2]+1]-lcoorp[2*necon[0*nen + 1]+1];
		
//PetscPrintf(PETSC_COMM_WORLD, "dx[%f] dy[%f] \n", dx, dy);		


		VecRestoreArray(lcoor,&lcoorp);

		// Create the minimum element connectivity shit
		PetscInt ElemConn;
		// Check dx,dy,dz and find max conn for a given rmin
		ElemConn = (PetscInt)PetscMax(ceil(rmin/dx)-1,ceil(rmin/dy)-1);
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
		PetscScalar xmax = (M-1)*dx;
		PetscScalar ymax = (N-1)*dy;
		
		DMDASetUniformCoordinates(da_elem , dx/2.0,xmax-dx/2.0, dy/2.0,ymax-dy/2.0, 0,0);
                 
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
//VecView(Hs, PETSC_VIEWER_STDOUT_WORLD);
		//Vec w;
		//VecDuplicate(xPhys, &w);

		//ierr = MatMult(H,x,xPhys); CHKERRQ(ierr);
//VecView(xPhys, PETSC_VIEWER_STDOUT_WORLD);
		//VecPointwiseDivide(xPhys,xPhys,Hs);
                

		// Clean up
		VecRestoreArray(lcoor,&lcoorp);
		VecDestroy(&dummy);
		delete [] Lx;
		delete [] Ly;
		//MatDestroy(&H);
		//VecDestroy(&xPhys);
		//VecDestroy(&x);
                DMDestroy(&da_elem);

		return ierr;
}

/*
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
*/
