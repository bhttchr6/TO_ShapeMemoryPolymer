/// External Force Vector on Gripper Domain ////////////


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


static PetscErrorCode FFext(DM elas_da, Vec &Fext, PetscInt nelh_x, PetscInt nelh_y, PetscInt rank, PetscInt nelx, PetscInt nely, PetscScalar Force_amt)
{
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
  /*
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
 bc_vals[j] =  2*(Force_amt/(nely+1));
if(sj+ny-1==nely && j==ny-1) {bc_vals[j] = Force_amt/(nely+1);} 

if (sj==0 && j==nelh_y) {bc_vals[j] = Force_amt/(nely+1);}
}

if (sj==0 && j<nelh_y) {bc_vals[j] = 0.0;}
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

*/

PetscFunctionReturn(0);
}
