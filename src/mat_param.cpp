#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "analysis.h"
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::IOFormat;
using std::vector;
typedef Eigen::SparseMatrix<double> SpMat;
using std::pow;
#define ndof 2
#define nen 4
#define ndim 2
#define Tmax 350
#define Tmin 330
#define Tg 340
#define plane_strain 1
#define plane_stress 0

void param(int phase, VectorXd rho, VectorXd &holes,double penal, int e, MatrixXd& KEQ, MatrixXd& dKEQ_dx, MatrixXd& KNEQ, MatrixXd& dKNEQ_dx, double& neta, double& dneta_dx, MatrixXd& I, PetscInt int_pt, PetscInt iter)
{

IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n","[","]","[","]");
//SMP2  material
double Eeq_r_smp2 = 0.5;
double Eeq_g_smp2 = 1500;
double Eneq_r_smp2 = 0.04;
double Eneq_g_smp2 = 180;

//smp material
double Eeq_r_smp = 0.39;
double Eeq_g_smp = 1100;
double Eneq_r_smp = 0.02;
double Eneq_g_smp = 150;

// stiffness of mixture
double rho_e=rho(e);
double holes_e =holes(e);

// Stiffness to simulate the hole of gripper
if (holes_e == 1.0){
//SMP2  material
 Eeq_r_smp2 = 0.001;
 Eeq_g_smp2 = 0.001;
 Eneq_r_smp2 = 0.001;
 Eneq_g_smp2 = 0.001;

//smp material
 Eeq_r_smp = 0.001;
 Eeq_g_smp = 0.001;
 Eneq_r_smp = 0.001;
 Eneq_g_smp = 0.001;

}




double Eeq_r = Eeq_r_smp+pow(rho_e,penal)*(Eeq_r_smp2-Eeq_r_smp);
double Eneq_r = Eneq_r_smp+pow(rho_e,penal)*(Eneq_r_smp2-Eneq_r_smp);

double Eeq_g = Eeq_g_smp+pow(rho_e,penal)*(Eeq_g_smp2-Eeq_g_smp);
double Eneq_g = Eneq_g_smp+pow(rho_e,penal)*(Eneq_g_smp2-Eneq_g_smp);





// derivative of stiffness of mixture
double dEeqr_dx = penal*pow(rho_e,penal-1)*(Eeq_r_smp2-Eeq_r_smp);
double dEneqr_dx = penal*pow(rho_e,penal-1)*(Eneq_r_smp2-Eneq_r_smp);

double dEeqg_dx = penal*pow(rho_e,penal-1)*(Eeq_g_smp2-Eeq_g_smp);
double dEneqg_dx = penal*pow(rho_e,penal-1)*(Eneq_g_smp2-Eneq_g_smp);





// poisons ratio of mixture
double nu_r_smp=0.4;double nu_g_smp=0.3; double nu_r_smp2 = 0.4;double nu_g_smp2 = 0.3;

// poissons ratio for gripper hole
if(holes_e == 1.0){ nu_r_smp =0.001; nu_g_smp = 0.001; nu_r_smp2 = 0.001; nu_g_smp2 = 0.001;}

//derivative of poisson's ratio
double nu_r=nu_r_smp+rho_e*(nu_r_smp2-nu_r_smp);
double dnur_dx=(nu_r_smp2-nu_r_smp);

double nu_g=nu_g_smp+rho_e*(nu_g_smp2-nu_g_smp);
double dnug_dx=(nu_g_smp2-nu_g_smp);

// viscoleastic properties of mixture
double neta_r_smp=1;double neta_g_smp=4000;
double neta_r_smp2=1.5;double neta_g_smp2=4500;
double neta_r=neta_r_smp+rho_e*(neta_r_smp2-neta_r_smp);
double dnetar_dx=(neta_r_smp2-neta_r_smp);

// Viscolelastic properties of gripper hole
if (holes_e == 1.0){
neta_r_smp = 0.001; neta_g_smp = 0.001; neta_r_smp2 =0.001; neta_g_smp2 = 0.001;
}

double neta_g=neta_g_smp+rho_e*(neta_g_smp2-neta_g_smp);
double dnetag_dx=(neta_g_smp2-neta_g_smp);

I=MatrixXd::Identity(3,3);
/*
MatrixXd KEQ(3,3);
MatrixXd dKEQ_dx(3,3);
MatrixXd KNEQ(3,3);
MatrixXd dKNEQ_dx(3,3);
*/
// plane strain formulations
if(phase==1){
	if(plane_strain==1){
		int ps=1;
		KEQ<< 1-nu_r, nu_r,   0,
                     nu_r,    1-nu_r, 0,
                     0,       0,      0.5-nu_r;
                KEQ=(Eeq_r/((1+nu_r)*(1-2*nu_r)))*KEQ;
                stiffness(nu_r,Eeq_r,dnur_dx,dEeqr_dx,ps,dKEQ_dx);
		

		KNEQ<< 1-nu_r, nu_r,   0,
                     nu_r,    1-nu_r, 0,
                     0,       0,      0.5-nu_r;
                KNEQ=(Eneq_r/((1+nu_r)*(1-2*nu_r)))*KNEQ;
                
                stiffness(nu_r,Eneq_r,dnur_dx,dEneqr_dx,ps,dKNEQ_dx);
}


	if(plane_stress==1){
		int ps=2;
		KEQ<< 1, nu_r, 0,
                      nu_r, 1, 0,
                      0,    0, 0.5*(1-nu_r);
		KEQ=(Eeq_r/(1-pow(nu_r,2)))*KEQ;
		stiffness(nu_r,Eeq_r,dnur_dx,dEeqr_dx,ps,dKEQ_dx);


		KNEQ<< 1, nu_r, 0,
			nu_r, 1, 0,
			0, 0, 0.5*(1-nu_r);
		KNEQ=(Eneq_r/(1-pow(nu_r,2)))*KNEQ;
  		stiffness(nu_r,Eneq_r,dnur_dx,dEneqr_dx,ps,dKNEQ_dx);
}

	neta=neta_r;
	dneta_dx=dnetar_dx;
}//end of phase 1


if(phase==2){
	if(plane_strain==1){
		int ps=1;
		KEQ<<1-nu_g, nu_g,   0,
		     nu_g,   1-nu_g, 0,
                     0,      0,      0.5-nu_g;
		KEQ=(Eeq_g/((1+nu_g)*(1-2*nu_g)))*KEQ;
		stiffness(nu_g,Eeq_g,dnug_dx,dEeqg_dx,ps,dKEQ_dx);

		KNEQ<<1-nu_g, nu_g, 0,
                      nu_g, 1-nu_g, 0,
                      0,    0,      0.5-nu_g;
		KNEQ=(Eneq_g/((1+nu_g)*(1-2*nu_g)))*KNEQ;
		stiffness(nu_g, Eneq_g,dnug_dx,dEneqg_dx,ps,dKNEQ_dx);
}



if(plane_stress==1){
	int ps=2;
	KEQ<< 1, nu_g, 0,
              nu_g, 1, 0,
              0, 0, 0.5*(1-nu_g);

	KEQ=(Eeq_g/(1-pow(nu_g,2)))*KEQ;
	KNEQ<<1, nu_g, 0,
              nu_g,1,0,
              0, 0, 0.5*(1-nu_g);

        stiffness(nu_g, Eneq_g,dnug_dx,dEneqg_dx,ps,dKNEQ_dx);
}

	neta=neta_g;
	dneta_dx=dnetag_dx;
}//end of phase 2	

//if(phase==1){if(iter==3){if(e==176){ if(int_pt==1){printf("%.12f\n",pow(rho_e,3.0));std::cout<<  "============"   <<std::endl;std::cout<<rho_e<<" "<<  pow(rho_e,penal) <<std::endl;std::cout<<  "============"   <<std::endl;//std::cout<< e_r.format(HeavyFmt)  <<std::endl;std::cout<<  "============"   <<std::endl;std::cout<< sigma_vis.format(HeavyFmt)  <<std::endl;
//}}}}

}
//*******************************************************************************************************************************************************************
void stiffness(double nu, double E, double dnu_dx, double dE_dx, int pole, MatrixXd& dK_dx)
{
if(pole==1){
	double t1=((1+nu)*(1-2*nu)*(dE_dx*(1-nu)-E*dnu_dx)-E*(1-nu)*((1+nu)*(-2*dnu_dx)+(1-2*nu)*dnu_dx))/pow((1+nu)*(1-2*nu),2);
        double t2=((1+nu)*(1-2*nu)*(E*dnu_dx+nu*dE_dx)-E*nu*((1+nu)*(-2*dnu_dx)+(1-2*nu)*dnu_dx))/pow((1+nu)*(1-2*nu),2);
        double t3=((1+nu)*(1-2*nu)*(E*(-dnu_dx)+(0.5-nu)*dE_dx)-E*(0.5-nu)*((1+nu)*(-2*dnu_dx)+(1-2*nu)*dnu_dx))/pow((1+nu)*(1-2*nu),2);
	dK_dx<< t1,t2,0,
                t2,t1,0,
                0,0, t3;
}

if(pole==2){
	double t1=((1-pow(nu,2))*dE_dx-E*(-2*nu*dnu_dx))/pow(1-pow(nu,2),2);
	double t2=((1-pow(nu,2))*(dE_dx*nu+E*dnu_dx)-E*nu*(-2*nu*dnu_dx))/pow(1-pow(nu,2),2);
	double t3=0.5*((1-pow(nu,2))*(dE_dx*(1-nu)+E*(-dnu_dx))-E*(1-nu)*(-2*nu*dnu_dx))/pow(1-pow(nu,2),2);
        dK_dx<< t1,t2,0,
                t2,t1,0,
                0,0,t3;
}
}
//*********************************************************************************************************************************************************************
void paracon(MatrixXd Keq, double neta, double delta_t, MatrixXd Kneq, MatrixXd I,MatrixXd& H, MatrixXd& A, MatrixXd& B)
{

double cons=delta_t/neta;

H=(I+cons*Kneq);
A=(Kneq+Keq)-(cons)*Kneq*(H.inverse()*Kneq);
B=Kneq*H.inverse();

} 
