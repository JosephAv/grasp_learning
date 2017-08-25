/*

Software License Agreement (BSD License)

Copyright (c) 2016--, Liana Bertoni (liana.bertoni@gmail.com)
  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder(s) nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Contact GitHub API Training Shop Blog About
*/


#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames_io.hpp>

#include <boost/scoped_ptr.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>

#include <math.h>
#include <stdio.h>
#include <ctime>
#include <time.h>

#include "PGR_4.h"
#include "quality.h"
#include "normal_component_box_surface.h"


using namespace std;
using namespace Eigen;
using namespace KDL;




//////////////////////////////////	BOX

Eigen::VectorXd box(3);
//////////////////////////////////////////////


////////////////////////////////////  POSE GRASP NOMINAL

Eigen::VectorXd pose_grasp_nominal(7);
/////////////////////////////////////////////


////////////////////////////////////  POSE GRASP SUCCESSFULL

Eigen::VectorXd pose_grasp_success(7);
/////////////////////////////////////////////



////////////////////////////////////  POSE GRASP SUCCESSFULL

Eigen::VectorXd joints(33);
/////////////////////////////////////////////



/////////////////////////////////// CONTACT POINTS

Eigen::VectorXd contact_points(20*3);
Eigen::MatrixXd cp(20,3);
///////////////////////////////////////


////////////////////////////////////  CONTACT FORCE

Eigen::VectorXd contact_wrenches(20*6);
Eigen::MatrixXd cf(20,6);
/////////////////////////////////////////////


////////////////////////////////////  ALL CONTACT FLAG

Eigen::VectorXd contact_flag(20);
/////////////////////////////////////////////


////////////////////////////////////  CONTACT

std::vector<int> contact_id;
/////////////////////////////////////////////



// take it from yaml file
int n_rows;
int n_cols;
int quality_index;


string file_name;
string relative_path_file;


string frame_name_finger[5];
string frame_name_root;
//std::string root_name = "right_hand_softhand_base";
//std::string root_name = "right_hand_palm_link";
//std::string root_name = "world";
int n_fingers;
double joint_stiffness = 0;
double contact_stiffness = 0;
int n_q = 39;

double quality_i = 0;
double quality_i_A = 0;
double quality_i_B = 0;


double mu = 0.03;
double f_i_max = 1;







int main (int argc, char **argv)
{

	ros::init(argc, argv, "Grasp_quality");	// ROS node
	ros::NodeHandle nh;



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	nh.param<std::string>("file_name", relative_path_file, "/db/box_db_2.csv" );
	nh.param<std::string>("frame_name_thumb", frame_name_finger[0], "right_hand_index_distal_link" );
	nh.param<std::string>("frame_name_index", frame_name_finger[1], "right_hand_little_distal_link" );
	nh.param<std::string>("frame_name_middle", frame_name_finger[2], "right_hand_middle_distal_link");
	nh.param<std::string>("frame_name_ring", frame_name_finger[3], "right_hand_ring_distal_link");
	nh.param<std::string>("frame_name_little", frame_name_finger[4], "right_hand_thumb_distal_link");
	nh.param<std::string>("frame_name_root", frame_name_root, "world");
	nh.param<double>("joint_stiffness",joint_stiffness,0);
	nh.param<double>("contact_stiffness",contact_stiffness,0);
	nh.param<double>("mu", mu, 0.03);
	nh.param<double>("f_i_max", f_i_max, 1);
	nh.param<int>("quality_index",quality_index,0);
	nh.param<int>("number_of_fingers", n_fingers, 5);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////




	ofstream file_output; //output file
    file_output.open("box_db_quality.txt", ofstream::app);








	///////////////////// load the data_base ////////////////////////////////////
	std::string path = ros::package::getPath("grasp_learning");
	file_name = path + relative_path_file;
	ifstream file(file_name); 

	std::cout << "file: " << file_name.c_str() << " is " << (file.is_open() == true ? "already" : "not") << " open" << std::endl;
	if(!file.is_open())
	return 0;



	
	//////////////////////////////// laod the urdf model //////////////////////////////////
	KDL::Tree hand_tree;
	KDL::Chain chains_hand_finger[n_fingers];
	KDL::Jacobian hand_jacob[n_fingers];
	KDL::JntArray q_finger[n_fingers];
	boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver[n_fingers];



	std::string robot_desc_string;
	nh.param("robot_description", robot_desc_string, string());  // robot description is the name in the launch file 
	if (!kdl_parser::treeFromString(robot_desc_string, hand_tree))
		{ ROS_ERROR("Failed to construct kdl tree"); return false;}
  

	for(int i=0; i < n_fingers; i++) // get chain for each fingers
	{	
		hand_tree.getChain(frame_name_root, frame_name_finger[i], chains_hand_finger[i]);      
		q_finger[i] = JntArray(chains_hand_finger[i].getNrOfJoints());
		jnt_to_jac_solver[i].reset(new KDL::ChainJntToJacSolver(chains_hand_finger[i]));
		q_finger[i].resize(chains_hand_finger[i].getNrOfJoints());
		hand_jacob[i].resize(chains_hand_finger[i].getNrOfJoints());

		// cout << "JntArray " << i << " : " << chains_hand_finger[i].getNrOfJoints() << endl;
	}







	///////////////////////////////// get values from file for each line //////////////////////////////////
	for(std::string line; getline( file, line, '\n' ); ) // for each line
	{
		std::vector<double> values_inline;
    	std::istringstream iss_line(line);	
    	for(std::string value; getline(iss_line, value, ',' ); )
    		values_inline.push_back(stod(value));





    	//////////////////////////////////////////////////////////////////////////////		IT CREATES THE DATA STRUCTURE   	
    	box(0) = values_inline[0];
    	box(1) = values_inline[1];
    	box(2) = values_inline[2];

    	pose_grasp_nominal(0) = values_inline[3];
    	pose_grasp_nominal(1) = values_inline[4];
    	pose_grasp_nominal(2) = values_inline[5];
    	pose_grasp_nominal(3) = values_inline[6];
    	pose_grasp_nominal(4) = values_inline[7];
    	pose_grasp_nominal(5) = values_inline[8];
    	pose_grasp_nominal(6) = values_inline[9];

    	pose_grasp_success(0) = values_inline[10];
    	pose_grasp_success(1) = values_inline[11];
    	pose_grasp_success(2) = values_inline[12];
    	pose_grasp_success(3) = values_inline[13];
    	pose_grasp_success(4) = values_inline[14];
    	pose_grasp_success(5) = values_inline[15];
    	pose_grasp_success(6) = values_inline[16];


    	for(int i = 0 ; i < 33 ; i++)
    		joints(i) = values_inline[17 + i];

    	for(int i = 0 ; i < 60 ; i++)
    		contact_points(i) = values_inline[50 + i];

   		for(int i = 0 ; i < 120 ; i++)
   			contact_wrenches(i) = values_inline[110 + i];

   		for(int i = 0 ; i < 20 ; i++) // 0 : no contact
   			contact_flag(i) = 0;	
   		
   		int k = 0;
   		for(int i = 0 ; i < 20 ; i++)
   			for(int j=0 ; j < 3 ; j++)
			{	cp(i,j) = values_inline[50 + k]; if( !std::isnan(cp(i,j))) contact_flag(i) = 1; k++; } cout << "cp : " << endl << cp << endl;
		
   		int h = 0;
   		for(int i = 0 ; i < 20 ; i++)
   			for(int j=0 ; j < 6; j++)
			{	cf(i,j) = values_inline[110 + h]; h++; }	cout << "cf : " << endl << cf << endl;


		int s = 0;
		for(int i = 0 ; i < 20 ; i++)
		{
			Eigen::MatrixXd c_Rotation_o(3,3);
			Eigen::MatrixXd c_Rotation_o_6(6,6);
			normal_component(c_Rotation_o, box(0)/2, box(1)/2, box(2)/2 , cp(i,0), cp(i,1), cp(i,2));

			c_Rotation_o_6.block<3,3>(0,0) = c_Rotation_o;
			c_Rotation_o_6.block<3,3>(3,3) = c_Rotation_o;
			c_Rotation_o_6.block<3,3>(3,0) = MatrixXd::Identity(3,3);

			Eigen::VectorXd fo(6);
			fo << -cf(i,0), -cf(i,1), -cf(i,2), -cf(i,3), -cf(i,4), -cf(i,5) ;
			Eigen::VectorXd fc(6);

			fc = c_Rotation_o_6*fo;

			Eigen::VectorXd cf(3);
			cf << fc(0), fc(1), fc(2);

			if(cf.norm() >= 1e-3 && cf(2) >= 0)
				contact_flag(i) = 1;
			else
				contact_flag(i) = 0;


			s += 6;
		}



		contact_id.resize(0);

		int n_c = 0;

		for(int i = 0 ; i < 20 ; i++)
		{	
			if( contact_flag(i) )
			{	cout << " contact_flag : " << i << endl;
				contact_id.push_back(i); 
				n_c++; 
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////////
    	cout << "contact_id.size() : " << endl << contact_id.size() << endl;

		for(int i = 0 ; i < contact_id.size() ; i++)
			cout << " contact_id : " << endl << contact_id[i] << endl;


		cout << "n_c : " << endl << n_c << endl;
		

		if(n_c > 0)
		{

		////////////////////////////////////////////////////////////////////////////////////////		GRASP MATRIX
		//for each contact point 
		Eigen::MatrixXd Grasp_Matrix_b  = MatrixXd::Zero(6,6*n_c);		// G
		Eigen::MatrixXd Grasp_Matrix_c  = MatrixXd::Zero(6,6*n_c);  	// G


		int step = 0 ;
		for(int i=0 ; i < contact_id.size() ; i++)
		{
			Eigen::MatrixXd Grasp_Matrix(6,6);  
 			Eigen::MatrixXd Skew_Matrix(3,3);
  			Eigen::MatrixXd Rotation(3,3);
  			Eigen::MatrixXd b_Rotation_c(3,3);

  					
	 		//check if the values ​​of the skew matrix are expressed in the correct reference system
    	  	Rotation = MatrixXd::Identity(3,3);

			Skew_Matrix(0,0) = Skew_Matrix(1,1) = Skew_Matrix(2,2) = 0;
     		Skew_Matrix(0,1) = - cp(contact_id[i],2); // -rz    
     		Skew_Matrix(0,2) = cp(contact_id[i],1);   // ry
        	Skew_Matrix(1,0) = cp(contact_id[i],2);   // rz
        	Skew_Matrix(2,0) = - cp(contact_id[i],1); // -ry
        	Skew_Matrix(1,2) = - cp(contact_id[i],0); // -rx
       		Skew_Matrix(2,1) = cp(contact_id[i],0);   // rx

        	Grasp_Matrix.block<3,3>(0,0) = Rotation;
     		Grasp_Matrix.block<3,3>(3,3) = Rotation;
     		Grasp_Matrix.block<3,3>(3,0) = Skew_Matrix * Rotation;
        	Grasp_Matrix.block<3,3>(0,3) = MatrixXd::Zero(3,3);

	      	Grasp_Matrix_b.block<6,6>(0,step) = Grasp_Matrix;


	      	//check if the values ​​of the skew matrix are expressed in the correct reference system
  			normal_component(b_Rotation_c, box(0)/2, box(1)/2, box(2)/2,cp(contact_id[i],0),cp(contact_id[i],1),cp(contact_id[i],2));
	      			
	      	Skew_Matrix(0,0) = Skew_Matrix(1,1) = Skew_Matrix(2,2) = 0;
     		Skew_Matrix(0,1) = - cp(contact_id[i],2); // -rz    
     		Skew_Matrix(0,2) = cp(contact_id[i],1);   // ry
        	Skew_Matrix(1,0) = cp(contact_id[i],2);   // rz
        	Skew_Matrix(2,0) = - cp(contact_id[i],1); // -ry
        	Skew_Matrix(1,2) = - cp(contact_id[i],0); // -rx
       		Skew_Matrix(2,1) = cp(contact_id[i],0);   // rx

        	Grasp_Matrix.block<3,3>(0,0) = b_Rotation_c;
     		Grasp_Matrix.block<3,3>(3,3) = b_Rotation_c;
     		Grasp_Matrix.block<3,3>(3,0) = Skew_Matrix * b_Rotation_c;
        	Grasp_Matrix.block<3,3>(0,3) = MatrixXd::Zero(3,3);

        	Grasp_Matrix_c.block<6,6>(0,step) = Grasp_Matrix;

        	step += 6;
        }			
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
    











		////////////////////////////////////////////////////////////////////////////////////	JACOBIAN MATRIX
    	Eigen::MatrixXd Hand_Jacobian_  = MatrixXd::Zero(6*n_c, n_q);	// J
    	Eigen::MatrixXd R_contact_hand_object_ = MatrixXd::Zero(3*n_c,3*n_c);


    	/////////////////////////////////////////////////////////////////////////////////////////// init values of the floating base
		KDL::Vector trasl_w_T_o(0,0, (box(2)/2)); // i m not sure if it is corrected
   		KDL::Vector trasl_o_T_p(pose_grasp_nominal(0),pose_grasp_nominal(1),pose_grasp_nominal(2));
    	KDL::Rotation R_o_T_p = Rotation::Quaternion(pose_grasp_nominal(3),pose_grasp_nominal(4),pose_grasp_nominal(5),pose_grasp_nominal(6));
   	
    	KDL::Frame w_T_o(trasl_w_T_o);
    	KDL::Frame o_T_p(R_o_T_p,trasl_o_T_p);
    	KDL::Frame p_T_h(chains_hand_finger[0].getSegment(5).getFrameToTip());

		KDL::Frame w_T_h = w_T_o * o_T_p * p_T_h ;
    	KDL::Vector trasl_w_T_h = w_T_h.p;
    	KDL::Rotation R_w_T_h = w_T_h.M;

    	double roll , pitch , yaw ;

    	R_w_T_h.GetRPY(roll,pitch,yaw);
    	///////////////////////////////////////////////////////////////////////////////////////////////////////
    

    	int first_element_joint_array = 17;
		int number_of_joints = 33;
		int first_element_cp_array = first_element_joint_array + number_of_joints;


    	int k_ = 0;
		for(int i = 0; i < n_fingers; i++) //joint values
    	{	
    		q_finger[i](0) = trasl_w_T_h.x();
    		q_finger[i](1) = trasl_w_T_h.y();
    		q_finger[i](2) = trasl_w_T_h.z();
    		q_finger[i](3) = roll;
    		q_finger[i](4) = pitch;
    		q_finger[i](5) = yaw;


    		if(i == 4) // thumb
    		{
    			q_finger[i](6) = joints(k_); // joint around z-axis knuckle

				q_finger[i](7) = joints(k_+1) ; // joint around y-axis knuckle
				q_finger[i](8) = joints(k_+2) ; // joint around y-axis proximal

				q_finger[i](9) = joints(k_+3) ; // joint around y-axis proximal
				q_finger[i](10) = joints(k_+4) ; // joint around y-axis distal

				k_+=5;
			}
			else
			{
				q_finger[i](6) = joints(k_); // joint around z-axis knuckle

				q_finger[i](7) = joints(k_+1) ; // joint around y-axis knuckle
				q_finger[i](8) = joints(k_+2) ; // joint around y-axis proximal

				q_finger[i](9) = joints(k_+3) ; // joint around y-axis proximal
				q_finger[i](10) = joints(k_+4) ;  // joint around y-axis middle

				q_finger[i](11) = joints(k_+5) ;  // joint around y-axis middle
				q_finger[i](12) = joints(k_+6) ;  // joint around y-axis distal

				k_+=7;
			}			
		}


		int s_ = 0;
    	int step_ = 0;
		for(int i=0 ; i < contact_id.size() ; i++)
		{
			for(int n = 0 ; n < n_fingers ; n++)//calc jacobian for each finger
       					jnt_to_jac_solver[n]->JntToJac(q_finger[n], hand_jacob[n], -1);

    		int which_finger = 0;
      		int which_phalanx = 0;

	        
	        if( contact_id[i]>= 0 && contact_id[i]<= 4 ) which_finger = 0; // index
	        if( contact_id[i]>= 5 && contact_id[i]<= 8 ) which_finger = 1; // little
	        if( contact_id[i]>= 9 && contact_id[i]<= 12 ) which_finger = 2; // middle
	        if( contact_id[i]>= 13 && contact_id[i]<= 16 ) which_finger = 3; // ring
	        if( contact_id[i]>= 17 && contact_id[i]<= 19 ) which_finger = 4; // thumb

	        if( contact_id[i]==0 ) which_phalanx = 5; //palm
	        if( contact_id[i]==1 || contact_id[i]==5 || contact_id[i]==9  || contact_id[i]==13 || contact_id[i]==17)  which_phalanx = 7;
	        if( contact_id[i]==2 || contact_id[i]==6 || contact_id[i]==10 || contact_id[i]==14 || contact_id[i]==18)  which_phalanx = 9;
	       	if( contact_id[i]==3 || contact_id[i]==7 || contact_id[i]==11 || contact_id[i]==15 || contact_id[i]==19)  which_phalanx = 11;
	       	if( contact_id[i]==4 || contact_id[i]==8 || contact_id[i]==12 || contact_id[i]==16 ) which_phalanx = 13;

	       	///////////////////////////////////////////////////////////////////				Ks = RtKsR
			Eigen::MatrixXd h_T_o = MatrixXd::Identity(4,4);

			Eigen::MatrixXd R_k(3,3);
			double qx = pose_grasp_success(3);
  			double qy = pose_grasp_success(4);
  			double qz = pose_grasp_success(5);
  			double qw = pose_grasp_success(6);
			double q_k[] = {qw, qx, qy, qz};
		
			R_k(0,0)	= q_k[0]*q_k[0] + q_k[1]*q_k[1] - q_k[2]*q_k[2] - q_k[3]*q_k[3];
			R_k(0,1)	= 2*q_k[1]*q_k[2] - 2*q_k[0]*q_k[3];
			R_k(0,2)	= 2*q_k[1]*q_k[3] + 2*q_k[0]*q_k[2];
			R_k(1,0)	= 2*q_k[1]*q_k[2] + 2*q_k[0]*q_k[3];
			R_k(1,1)	= q_k[0]*q_k[0] + q_k[2]*q_k[2] - q_k[1]*q_k[1] - q_k[3]*q_k[3];
			R_k(1,2)	= 2*q_k[2]*q_k[3] - 2*q_k[0]*q_k[1];
			R_k(2,0)	= 2*q_k[1]*q_k[3] - 2*q_k[0]*q_k[2];
			R_k(2,1)	= 2*q_k[2]*q_k[3] + 2*q_k[0]*q_k[1];
			R_k(2,2)	= q_k[0]*q_k[0] + q_k[3]*q_k[3] - q_k[2]*q_k[2] - q_k[1]*q_k[1];

			h_T_o.block<3,3>(0,0) = R_k;
			h_T_o(0,3) = pose_grasp_success(0);
			h_T_o(1,3) = pose_grasp_success(1);
			h_T_o(2,3) = pose_grasp_success(2);


			Eigen::MatrixXd o_T_c = MatrixXd::Identity(4,4);
			Eigen::MatrixXd b_Rotation_c(3,3);
			normal_component(b_Rotation_c, box(0)/2, box(1)/2, box(2)/2,cp(contact_id[i],0),cp(contact_id[i],1),cp(contact_id[i],2));
	      

         	o_T_c.block<3,3>(0,0) = b_Rotation_c;
  			o_T_c(0,3) = cp(contact_id[i],0);
  			o_T_c(1,3) = cp(contact_id[i],1);
  			o_T_c(2,3) = cp(contact_id[i],2);

			Eigen::MatrixXd h_T_c = h_T_o * o_T_c ;
			Eigen::MatrixXd h_T_e = MatrixXd::Identity(4,4);
  					

			KDL::Frame h_T_e_step ;
  			for ( int i = 0 ; i < which_phalanx ; i++)
  				h_T_e_step = h_T_e_step * chains_hand_finger[which_finger].getSegment(i).getFrameToTip();


  			h_T_e(0,0) = h_T_e_step.M.data[0];
	        h_T_e(0,1) = h_T_e_step.M.data[1];
	        h_T_e(0,2) = h_T_e_step.M.data[2];
	        h_T_e(1,0) = h_T_e_step.M.data[3];
	        h_T_e(1,1) = h_T_e_step.M.data[4];
	        h_T_e(1,2) = h_T_e_step.M.data[5];
	        h_T_e(2,0) = h_T_e_step.M.data[6];
	        h_T_e(2,1) = h_T_e_step.M.data[7];
	        h_T_e(2,2) = h_T_e_step.M.data[8];
	        h_T_e(0,3) = h_T_e_step.p.x();
	        h_T_e(1,3) = h_T_e_step.p.y();
	        h_T_e(2,3) = h_T_e_step.p.z();

			Eigen::MatrixXd c_T_h = h_T_c.inverse();
			Eigen::MatrixXd c_T_e = c_T_h * h_T_e;



			///////////////////////////////////////////////////////////////////////////////////////////////////				 Test 
			Eigen::MatrixXd h_T_c_mano = h_T_e * c_T_e.inverse();
			Eigen::MatrixXd R_hand_contact(3,3);
			R_hand_contact = h_T_c_mano.block<3,3>(0,0);

			Eigen::MatrixXd R_object_contact(3,3);
			R_object_contact = h_T_c.block<3,3>(0,0);

			Eigen::MatrixXd R_contact_hand_object = R_hand_contact.transpose() * R_object_contact;
			// cout << " R_contact_hand_object " << endl << R_contact_hand_object << endl; // it is identity 

	        R_contact_hand_object_.block<3,3>(s_,s_) = R_contact_hand_object; // for PGR
			// cout << " R_contact_hand_object (3*n_c,3*n_c)" << endl << R_contact_hand_object_ << endl;

			KDL::Chain chain_contact;	           		
			KDL::Frame frame_relative_contact;
			string name_which_phalanx = chains_hand_finger[which_finger].getSegment(which_phalanx).getName();

			KDL::Rotation c_R_e(c_T_e(0,0),c_T_e(0,1),c_T_e(0,2),c_T_e(1,0),c_T_e(1,1),c_T_e(1,2),c_T_e(2,0),c_T_e(2,1),c_T_e(2,2));
			KDL::Vector p_(c_T_e(0,3), c_T_e(1,3), c_T_e(2,3));

			frame_relative_contact.M = c_R_e;
			frame_relative_contact.p = p_;

			hand_tree.getChain(frame_name_root, name_which_phalanx, chain_contact);      
			chain_contact.addSegment(Segment(Joint(Joint::None), frame_relative_contact));
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



			boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_relative_contact;
			jnt_to_jac_solver_relative_contact.reset(new KDL::ChainJntToJacSolver(chain_contact));

			jnt_to_jac_solver_relative_contact->JntToJac(q_finger[which_finger],hand_jacob[which_finger]);
	  		Hand_Jacobian_.block<6,6>(step_,0) = hand_jacob[0].data.topLeftCorner(6,6);	// 6 
			Hand_Jacobian_.block<6,7>(step_,6) = hand_jacob[0].data.topRightCorner(6,7); // 7
			Hand_Jacobian_.block<6,7>(step_,13)= hand_jacob[1].data.topRightCorner(6,7); // 7
			Hand_Jacobian_.block<6,7>(step_,20)= hand_jacob[2].data.topRightCorner(6,7); // 7
			Hand_Jacobian_.block<6,7>(step_,27)= hand_jacob[3].data.topRightCorner(6,7); // 7
			Hand_Jacobian_.block<6,5>(step_,34)= hand_jacob[4].data.topRightCorner(6,5); // 5

			s_ += 3;
			step_ += 6;
	  			
        }
	    ///////////////////////////////////////////////////////////////////////////////////////////////////////


	    if(quality_index < 6) quality_i = quality(quality_index, Grasp_Matrix_b, Hand_Jacobian_); 		

		if(quality_index == 6 ) // PCR PGR 
    	{
			Eigen::MatrixXd G_c(6,6*n_c);
			Eigen::MatrixXd J_c(6*n_c, n_q);
			Eigen::MatrixXd R_c = MatrixXd::Identity(3*n_c, 3*n_c);
			Eigen::VectorXd f_c(6*n_c);


			G_c = Grasp_Matrix_c;
			J_c = Hand_Jacobian_ ;
			R_c = R_contact_hand_object_;


			int s = 0;
			for(int i = 0 ; i < n_c ; i++)
			{
				Eigen::MatrixXd c_Rotation_o(3,3);
				Eigen::MatrixXd c_Rotation_o_6(6,6);
				normal_component(c_Rotation_o, box(0)/2, box(1)/2, box(2)/2 , cp(contact_id[i],0), cp(contact_id[i],1), cp(contact_id[i],2));

				c_Rotation_o_6.block<3,3>(0,0) = c_Rotation_o;
				c_Rotation_o_6.block<3,3>(3,3) = c_Rotation_o;
				c_Rotation_o_6.block<3,3>(3,0) = MatrixXd::Identity(3,3);

				Eigen::VectorXd fo(6);
				fo << -cf(contact_id[i],0), -cf(contact_id[i],1), -cf(contact_id[i],2), -cf(contact_id[i],3), -cf(contact_id[i],4), -cf(contact_id[i],5) ;
				Eigen::VectorXd fc(6);

				fc = c_Rotation_o_6*fo;

				f_c(s+0) = fc(0);
				f_c(s+1) = fc(1);
				f_c(s+2) = fc(2);
				f_c(s+3) = fc(3);
				f_c(s+4) = fc(4);
				f_c(s+5) = fc(5);

				s += 6;
			}



			// cout << " G_c : " << endl << G_c << endl;
			// cout << " R_c : " << endl << R_c << endl;
			// cout << " f_c : " << endl << f_c << endl;	

    		Eigen::MatrixXd Contact_Stiffness_Matrix = MatrixXd::Zero(3,3);		// Kis
    		for(int i = 0 ; i < Contact_Stiffness_Matrix.rows() ; i++) //Kis
    			Contact_Stiffness_Matrix(i,i) = contact_stiffness;


    		Eigen::MatrixXd Joint_Stiffness_Matrix = MatrixXd::Zero(n_q,n_q);    // Kp    				
    		for(int j = 0 ; j < Joint_Stiffness_Matrix.rows() ; j++) //Kp
    			Joint_Stiffness_Matrix(j,j) = joint_stiffness;

    			
    		quality_i = quality_pcr_pgr_4(f_c, G_c, J_c, R_c, Contact_Stiffness_Matrix, Joint_Stiffness_Matrix, mu, f_i_max);
    	}
    	else
    		quality_i = -50; // no valid qualiti index




    	}
    	else
    		quality_i = -60; // no contact
    




	    cout << " THEFINALCOUNTDOWN:::: " << quality_i << endl;
	    file_output << quality_i ;
		
		for(int i = 0 ; i < 10 ; i++)
	    	file_output << ' ' << i+1 << ":" << values_inline[i] ;
      	file_output << ' ' << endl;

    	quality_i = 0;	
    }// end file

	cout << endl;
	cout << "quality_index : " << quality_index << endl;
	
	cout << " YEAH ENJOY " << endl;
	cout << "   fine   " << endl;

	ros::spinOnce();
	return 0;

}