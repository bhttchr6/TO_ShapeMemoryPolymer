#include <Eigen/Dense>
#include <Eigen/Sparse>
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
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using Eigen::IOFormat;
typedef Eigen::SparseMatrix<double> SpMat;

void shapel(double psi, double eta, MatrixXd C, double& DET, MatrixXd& B, PetscInt int_pt, PetscInt e, PetscInt iter)
{
/*
%********************************************************************
% Compute shape function, derivatives, and determinant of hexahedral element
...SHAPEL gives three things in return,
    %********************************************************************
*/
IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n","[","]","[","]");

 MatrixXd GN(2,4);
 MatrixXd J(2,2);
 MatrixXd Jinv(2,2);
 MatrixXd BB(2,4);
 

GN<< eta-1, 1-eta, 1+eta, -eta-1,
     psi-1, -psi-1,1+psi, 1-psi;
GN=0.25*GN;
     

J=GN*C;

DET=J.determinant();

Jinv=J.inverse();

BB=Jinv*GN;

 double B1x     = BB(0,0);
 double B2x     = BB(0,1);
 double B3x     = BB(0,2);
 double B4x     = BB(0,3);
 double B1y     = BB(1,0);
 double B2y     = BB(1,1);
 double B3y     = BB(1,2);
 double B4y     = BB(1,3);

B<< B1x,  0,   B2x, 0,   B3x, 0,   B4x, 0,
    0,    B1y, 0,   B2y, 0,   B3y, 0,   B4y,
    B1y,  B1x,  B2y,  B2x, B3y,  B3x, B4y, B4x;


//if(int_pt==1){if(e==29){if(iter==5){std::cout<<B.format(HeavyFmt)<<std::endl;} } }

}

