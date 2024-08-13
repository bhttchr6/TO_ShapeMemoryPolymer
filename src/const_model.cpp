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
 void SMP_cycle(PetscMPIInt rank, MatrixXd epsilon_total, PetscInt e, PetscScalar T, PetscScalar *T_hist, PetscScalar delta_t, VectorXd& rho, VectorXd &holes, PetscScalar penal, PetscInt iter, PetscInt int_pt, MatrixXd B, MatrixXd& sigma_vis, MatrixXd& Dmat_vis, MatrixXd& dfint_dx, PetscInt sens_flag)


{
 
IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n","[","]","[","]");

VectorXd strain(3);
VectorXd temp(iter+1);

double rho_e=rho(e);
double holes_e = holes(e);
//if(rank==0){printf("[%f]  \n", rho_e);}

double eta_i_smp2 = 15000;
double eta_i_smp = 10000;

double eta_i = eta_i_smp+rho_e*(eta_i_smp2-eta_i_smp);
double detai_dx = (eta_i_smp2-eta_i_smp);


// aux and SMP material
double c_1 = 2.76e-05;
double q=3;
double c_2=4;

for(int i=0;i<iter;i++){
temp(i)=T_hist[i];
}

temp(iter)=T;

//if(rank==0) {if(e==0) {if(iter==4) {if(sens_flag==1){ if(int_pt==1){std::cout<<iter<<temp<<std::endl;}}}}}
//if (e==1){if(rank==0){std::cout<< epsilon_total<<std::endl;}}



// initialize history variables

MatrixXd e_ir_hist(iter+1,3);
MatrixXd e_ig_hist(iter+1,3);
MatrixXd e_i_hist(iter+1,3);
MatrixXd e_is_hist(iter+1,3);

MatrixXd e_ir(iter,3);
MatrixXd e_ig(iter,3);
MatrixXd e_i(iter,3);
MatrixXd e_is(iter,3);
MatrixXd e_g(iter,3);

e_ir_hist.row(0)<<0,0,0;
e_ig_hist.row(0)<<0,0,0;
e_i_hist.row(0)<<0,0,0;
e_is_hist.row(0)<<0,0,0;

VectorXd phi_g_total(iter+1);
VectorXd phi_g_1(iter+1);
VectorXd phi_g_2(iter+1);
VectorXd phi_r_1(iter+1);
VectorXd phi_r_2(iter+1);

// Phi_g and Phi_r for SMP_1
phi_g_1(0) = 1-(1/(1+c_1*pow((Tmax-temp(0)),c_2)));
phi_r_1(0) = 1- phi_g_1(0);

// Phi_g and Phi_r for SMP_2
double Tg2 = 345;
phi_g_2(0) = 1-1/(1+exp(-0.66*(temp(0)-Tg2)));
phi_r_2(0) = 1-phi_g_2(0);

// phi_g mixture
phi_g_total(0) = phi_g_1(0)+rho_e*(phi_g_2(0)-phi_g_1(0));

VectorXd phi_r_total(iter+1);

phi_r_total(0) = 1-phi_g_total(0);

VectorXd delta_phi_g(iter);
VectorXd k(iter);
VectorXd K1(iter);
VectorXd e_T_0(3);
VectorXd e_T(3);
e_T_0<<1,1,0;

// Define D steps
 std::vector <Eigen::MatrixXd> D_iter(iter);

// define diff var
        MatrixXd deirn_dx(iter+1,3);
	MatrixXd deign_dx(iter+1,3);
	MatrixXd dein_dx(iter+1,3);
	MatrixXd deisn_dx(iter+1,3);

	 
	



	VectorXd dphig_dx(iter+1);
	
	VectorXd ddeltaphig_dx(iter);
	VectorXd deT_dx(3);

	MatrixXd dC_dx(iter,3);
      

	VectorXd dK1_dx(iter);
	VectorXd dk_dx(iter);

	VectorXd dphir_dx(iter+1);

	MatrixXd dD_dx(3,3);
        dD_dx=MatrixXd::Zero(3,3);

        MatrixXd der_dx(iter,3);
        der_dx=MatrixXd::Zero(iter,3);

	MatrixXd deir_dx(iter,3);
        deir_dx=MatrixXd::Zero(iter,3);

	MatrixXd dei_dx(iter,3);
	dei_dx=MatrixXd::Zero(iter,3);

	MatrixXd deg_dx(iter,3);
	deg_dx=MatrixXd::Zero(iter,3);

	MatrixXd deig_dx(iter,3);
	deig_dx=MatrixXd::Zero(iter,3);

	MatrixXd deis_dx(iter,3);
	deis_dx=MatrixXd::Zero(iter,3);

// initialize history

	//deirn_dx.row(0)<< 0,0,0;
	//deign_dx.row(0)<< 0,0,0;
	//dein_dx.row(0)<< 0,0,0;
	//deisn_dx.row(0)<< 0,0,0;


VectorXd deirn_dx_row(3);
VectorXd deign_dx_row(3);
VectorXd dein_dx_row(3);
VectorXd deisn_dx_row(3);
VectorXd dC_dx_row(3);

//
VectorXd deir_dx_row(3);
VectorXd deig_dx_row(3);
VectorXd dei_dx_row(3);
VectorXd deis_dx_row(3);

dphig_dx(0) = phi_g_2(0)-phi_g_1(0);
// define other variables
MatrixXd H_r(3,3);
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
MatrixXd dHg_dx(3,3);

MatrixXd A_g(3,3);
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
MatrixXd D(3,3);
MatrixXd e_r(iter,3);


MatrixXd alpha(3,3);
VectorXd C_row(3); 

// Thermal parameters
double a_1_smp=-3.14e-4;double a_2_smp=0.7e-06;
double a_1_smp2=-3.14e-6;double a_2_smp2=0.7e-10;


        VectorXd e_ir_hist_row(3);
        VectorXd e_ir_row(3);
        VectorXd e_ig_hist_row(3);
 	VectorXd e_ig_row(3);
        VectorXd epsilon_total_row(3);
        VectorXd e_i_hist_row(3);
        VectorXd e_i_row(3);
        VectorXd e_is_hist_row(3);
	VectorXd e_is_row(3);
        VectorXd e_r_row(3);
        VectorXd sigma_row(3);
	VectorXd e_g_row(3);
	VectorXd der_dx_row(3);
	VectorXd deg_dx_row(3);
// iter loops
for(int i=0;i<iter;i++){

        // for SMP1
        phi_g_1(i+1) = 1-(1/(1+c_1*pow(Tmax-temp(i+1),c_2)));
        phi_r_1(i+1) = 1-phi_g_1(i+1);

       // for SMP2
       phi_g_2(i+1) = 1-1/(1+exp(-0.66*(temp(i+1)-Tg2)));
	phi_r_2(i+1) = 1-phi_g_2(i+1);


	// Phi_g of mixture
	phi_g_total(i+1) = phi_g_1(i+1) +rho_e*(phi_g_2(i+1)-phi_g_1(i+1));
	phi_r_total(i+1)= 1-phi_g_total(i+1);

	// delta phi_g of mixture
        delta_phi_g(i) = phi_g_total(i+1)-phi_g_total(i);
        k(i)=1;
	
        if(temp(i+1)>temp(i)){
	K1(i)=delta_phi_g(i)/phi_g_total(i+1);
	k(i)=1/(1-K1(i));}
	
       

 
	//thermal strain
	
	double a_1 = a_1_smp+rho_e*(a_1_smp2-a_1_smp);
	double a_2 = a_2_smp+rho_e*(a_2_smp2-a_2_smp);
        e_T=(a_1*(temp(i+1)-Tmax)+a_2*(pow(temp(i+1),2)-pow(Tmax,2)))*e_T_0;// thermal strain

        //rubbery phase properties
        int phase=1;
        param(phase,rho,holes,penal,e,Keq_r,dKeqr_dx,Kneq_r,dKneqr_dx, neta_r,dnetar_dx,I, int_pt, i);
        paracon(Keq_r,neta_r,delta_t, Kneq_r, I, H_r,A_r,B_r);
        
       

        dHr_dx=(delta_t/neta_r)*dKneqr_dx-(delta_t/pow(neta_r,2))*Kneq_r*dnetar_dx;
	dAr_dx=dKneqr_dx+dKeqr_dx-(delta_t/neta_r)*dKneqr_dx*(H_r.inverse()*Kneq_r) \
               -(delta_t/neta_r)*Kneq_r*((-H_r.inverse()*dHr_dx*H_r.inverse())*Kneq_r) \
               -(delta_t/neta_r)*Kneq_r*(H_r.inverse()*dKneqr_dx) \
               +(delta_t/pow(neta_r,2))*dnetar_dx*Kneq_r*(H_r.inverse()*Kneq_r);
	
	
       dBr_dx=(-H_r.inverse()*dHr_dx*H_r.inverse())*Kneq_r+H_r.inverse()*dKneqr_dx;
       
       //glassy phase properties
        phase=2;
        param(phase,rho, holes,penal,e,Keq_g,dKeqg_dx,Kneq_g,dKneqg_dx, neta_g,dnetag_dx,I, int_pt, i);
        paracon(Keq_g,neta_g,delta_t, Kneq_g, I, H_g,A_g,B_g);
        dHg_dx=(delta_t/neta_g)*dKneqg_dx-(delta_t/pow(neta_g,2))*Kneq_g*dnetag_dx;
	dAg_dx=dKneqg_dx+dKeqg_dx-(delta_t/neta_g)*dKneqg_dx*(H_g.inverse()*Kneq_g) \
               -(delta_t/neta_g)*Kneq_g*((-H_g.inverse()*dHr_dx*H_g.inverse())*Kneq_g) \
               -(delta_t/neta_g)*Kneq_g*(H_g.inverse()*dKneqg_dx) \
               +(delta_t/pow(neta_g,2))*dnetag_dx*Kneq_g*(H_g.inverse()*Kneq_g);
	
	
       dBg_dx=(-H_g.inverse()*dHg_dx*H_g.inverse())*Kneq_g+H_g.inverse()*dKneqg_dx;

	        


        epsilon_total_row=epsilon_total.row(i+1);
        e_ir_hist_row =e_ir_hist.row(i);
        e_ig_hist_row=e_ig_hist.row(i);
        e_i_hist_row=e_i_hist.row(i);
        e_is_hist_row=e_is_hist.row(i);
        
        C.row(i)=epsilon_total_row+phi_g_total(i+1)*(A_g.inverse()*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row))-e_i_hist_row+(delta_t/eta_i)*B_r*e_ir_hist_row-e_is_hist_row-e_T;
        if(temp(i+1)>temp(i)){
        C.row(i)=epsilon_total_row+phi_g_total(i+1)*(A_g.inverse()*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row))-e_i_hist_row+(delta_t/eta_i)*B_r*e_ir_hist_row-k(i)*e_is_hist_row-e_T;
        }// for heating

	
	C_row=C.row(i); 
        D=phi_r_total(i+1)*I+phi_g_total(i+1)*(A_g.inverse()*A_r)+(delta_t/eta_i)*A_r+delta_phi_g(i)*I;
	D_iter[i]=D;

        e_r.row(i)=D.inverse()*C_row;
       
        e_r_row=e_r.row(i);

        sigma_vis.row(i)=A_r*e_r_row-B_r*e_ir_hist_row;

        sigma_row=sigma_vis.row(i);

      

        alpha=((delta_t/neta_r)*(H_r.inverse()*Kneq_r));
        Dmat_vis=A_r*(D.inverse()*I);
        e_is.row(i)=e_is_hist_row+delta_phi_g(i)*e_r_row;

//if(i==3){if(e==176){ if(int_pt==1){std::cout<<  i+1  <<std::endl;std::cout<<  "============"   <<std::endl;std::cout<<rho_e<<" "<<  A_r.format(HeavyFmt)  <<std::endl;std::cout<<  "============"   <<std::endl;//std::cout<< e_r.format(HeavyFmt)  <<std::endl;std::cout<<  "============"   <<std::endl;std::cout<< sigma_vis.format(HeavyFmt)  <<std::endl;
//}}}

        if(temp(i+1)>temp(i)){e_is.row(i)=k(i)*e_is_hist_row;}
         e_i.row(i)=e_i_hist_row+(delta_t/eta_i)*sigma_row;
	e_ir.row(i)=H_r.inverse()*e_ir_hist_row +(delta_t/neta_r)*(H_r.inverse()*(Kneq_r*e_r_row));
	e_g.row(i)=A_g.inverse()*(A_r*e_r_row-B_r*e_ir_hist_row+B_g*e_ig_hist_row);
	e_g_row=e_g.row(i);
	e_ig.row(i)=(H_g.inverse()*e_ig_hist_row)+(delta_t/neta_g)*(H_g.inverse()*(Kneq_g*e_g_row));
	
	e_ir_row=e_ir.row(i);
	e_ig_row=e_ig.row(i);
	e_i_row=e_i.row(i);
	e_is_row=e_is.row(i);

	e_ir_hist.row(i+1)=e_ir_row;
	e_ig_hist.row(i+1)=e_ig_row;
	e_i_hist.row(i+1)=e_i_row;
	e_is_hist.row(i+1)=e_is_row;

      


}

//********************************************* SENSITIVITY ANALYSIS PART *****************************************************************************************


//dphig_dx(0) =0;
        

deirn_dx=MatrixXd::Zero(iter+1,3);
deign_dx=MatrixXd::Zero(iter+1,3);
dein_dx=MatrixXd::Zero(iter+1,3);
deisn_dx=MatrixXd::Zero(iter+1,3);
dk_dx=VectorXd::Zero(iter);
 dC_dx=MatrixXd::Zero(iter,3);
MatrixXd D_step(3,3);
if(sens_flag==1){
 for(int i=0;i<iter;i++){
	
			dphig_dx(i+1) = phi_g_2(i+1)-phi_g_1(i+1);
			//dphig_dx(i+1)=(1/pow(1+c_1*pow(Tmax-temp(i+1),c_2),2))*(q*pow(rho_e,q-1)*(c_1_aux-c_1_smp)*pow(Tmax-temp(i+1),c_2));
			
			ddeltaphig_dx(i) = dphig_dx(i+1)-dphig_dx(i);
			
			deT_dx=((a_1_smp2-a_1_smp)*(temp(i+1)-Tmax)+(a_2_smp2-a_2_smp)*(pow(temp(i+1),2)-pow(Tmax,2)))*e_T_0;

			deirn_dx_row=deirn_dx.row(i);
			deign_dx_row=deign_dx.row(i);
			dein_dx_row=dein_dx.row(i);
			deisn_dx_row=deisn_dx.row(i);

			       e_ir_hist_row =e_ir_hist.row(i);
				e_ig_hist_row=e_ig_hist.row(i);
				e_i_hist_row=e_i_hist.row(i);
				e_is_hist_row=e_is_hist.row(i);

				e_r_row=e_r.row(i);
				e_g_row=e_g.row(i);
				

			dC_dx.row(i)=dphig_dx(i+1)*(A_g.inverse()*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row)) \
				     +phi_g_total(i+1)*((-A_g.inverse()*dAg_dx*A_g.inverse())*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row) \
				     +A_g.inverse()*(-dBr_dx*e_ir_hist_row-B_r*deirn_dx_row+dBg_dx*e_ig_hist_row+B_g*deign_dx_row))\
				     +(delta_t/eta_i)*(dBr_dx*e_ir_hist_row+B_r*deirn_dx_row)\
				     -(delta_t/pow(eta_i,2))*B_r*e_ir_hist_row*detai_dx \
				     -dein_dx_row-deisn_dx_row-deT_dx;





			if(temp(i+1)>temp(i)){


			dK1_dx(i)=(phi_g_total(i+1)*ddeltaphig_dx(i)-delta_phi_g(i)*dphig_dx(i+1))/pow(phi_g_total(i+1),2);

			dk_dx(i)=dK1_dx(i)/pow((1-K1(i)),2);

			dC_dx.row(i)=dphig_dx(i+1)*(A_g.inverse()*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row))\
				     +phi_g_total(i+1)*((-A_g.inverse()*dAg_dx*A_g.inverse())*(-B_r*e_ir_hist_row+B_g*e_ig_hist_row) \
				     +A_g.inverse()*(-dBr_dx*e_ir_hist_row-B_r*deirn_dx_row+dBg_dx*e_ig_hist_row+B_g*deign_dx_row))\
				     +(delta_t/eta_i)*(dBr_dx*e_ir_hist_row+B_r*deirn_dx_row)\
				     -(delta_t/pow(eta_i,2))*B_r*e_ir_hist_row*detai_dx \
				     -dein_dx_row-k(i)*deisn_dx_row-dk_dx(i)*e_is_hist_row-deT_dx;


		

}
			dC_dx_row=dC_dx.row(i);
			C_row=C.row(i);
			dphir_dx(i+1)=-dphig_dx(i+1);

			dD_dx=(dphir_dx(i+1)+ddeltaphig_dx(i))*I+phi_g_total(i+1)*((-A_g.inverse()*dAg_dx*A_g.inverse())*A_r+A_g.inverse()*dAr_dx)\
				+(delta_t/eta_i)*dAr_dx\
				+dphig_dx(i+1)*(A_g.inverse()*A_r)\
				-(delta_t/pow(eta_i,2))*A_r*detai_dx;

			D_step=D_iter[i];

			der_dx.row(i)=(-D_step.inverse()*dD_dx*D_step.inverse())*C_row+D_step.inverse()*dC_dx_row;



			der_dx_row=der_dx.row(i);

			deir_dx.row(i)=(-H_r.inverse()*dHr_dx*H_r.inverse())*e_ir_hist_row \
					+H_r.inverse()*deirn_dx_row \
					+(delta_t/neta_r)*((-H_r.inverse()*dHr_dx*H_r.inverse())*Kneq_r)*e_r_row\
					+(delta_t/neta_r)*(H_r.inverse()*dKneqr_dx)*e_r_row \
					+(delta_t/neta_r)*(H_r.inverse()*Kneq_r)*der_dx_row\
					-(delta_t/pow(neta_r,2))*(H_r.inverse()*Kneq_r*e_r_row)*dnetar_dx;

			dei_dx.row(i)=dein_dx_row+(delta_t/eta_i)*(A_r*der_dx_row\
                                 	+dAr_dx*e_r_row \
					-dBr_dx*e_ir_hist_row-B_r*deirn_dx_row)\
					-(delta_t/pow(eta_i,2))*detai_dx*(A_r*e_r_row-B_r*e_ir_hist_row);

			deg_dx.row(i)=A_g.inverse()*(dAr_dx*e_r_row \
						     +A_r*der_dx_row\
						     -dBr_dx*e_ir_hist_row-B_r*deirn_dx_row\
						     +dBg_dx*e_ig_hist_row+B_g*deign_dx_row)\
						     +(-A_g.inverse()*dAg_dx*A_g.inverse())*(A_r*e_r_row\
						                                    -B_r*e_ir_hist_row\
										    +B_g*e_ig_hist_row);


			deg_dx_row=deg_dx.row(i);

			deig_dx.row(i)=(-H_g.inverse()*dHg_dx*H_g.inverse())*e_ig_hist_row\
                			+H_g.inverse()*deign_dx_row\
               				+(delta_t/neta_g)*(-H_g.inverse()*dHg_dx*H_g.inverse())*(Kneq_g*e_g_row)\
                			+(delta_t/neta_g)*(H_g.inverse()*(dKneqg_dx*e_g_row))\
                			+(delta_t/neta_g)*(H_g.inverse()*(Kneq_g*deg_dx_row))\
                			-(delta_t/pow(neta_g,2))*(H_g.inverse()*(Kneq_g*e_g_row))*dnetag_dx;

			deis_dx.row(i)=deisn_dx_row\
                			+delta_phi_g(i)*der_dx_row\
               				 + ddeltaphig_dx(i)*e_r_row;
			
            	if (temp(i+1)>temp(i)){
            		deis_dx.row(i)=k(i)*deisn_dx_row\
                			+dk_dx(i)*e_is_hist_row;

				}



			dfint_dx.row(i)=dAr_dx*e_r_row-dBr_dx*e_ir_hist_row-B_r*deirn_dx_row+A_r*der_dx_row;



			deir_dx_row=deir_dx.row(i);
			deig_dx_row=deig_dx.row(i);
			dei_dx_row=dei_dx.row(i);
			deis_dx_row=deis_dx.row(i);

			deirn_dx.row(i+1)=deir_dx_row;
			deign_dx.row(i+1)=deig_dx_row;
			dein_dx.row(i+1)=dei_dx_row;
			deisn_dx.row(i+1)=deis_dx_row;


//if(e==1){if(rank==0){if(iter==3){if(i==0){if(int_pt==1){std::cout<<e<<std::endl;std::cout<<iter<<std::endl;std::cout<<"***************"<<std::endl;std::cout<<dphig_dx <<std::endl;}}}}}



}//iter

}// if sens==1



}//func end
