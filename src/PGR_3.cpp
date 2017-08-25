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

#include "PGR_3.h"
#include "quality.h"
#include "normal_component_box_surface.h"


using namespace std;
using namespace Eigen;
using namespace KDL;


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
int type_of_contact = 0;
int n_q = 39;
int n_c_max = 20;

double quality_i = 0;
double quality_i_A = 0;
double quality_i_B = 0;


double mu = 0.03;
double f_i_max = 1;
int set_PCR_PGR = 0;


std::vector<double> synergie;



int main (int argc, char **argv)
{

	ros::init(argc, argv, "Grasp_quality");	// ROS node
	ros::NodeHandle nh;





 	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	nh.param<int>("n_rows_file",n_rows,108);
	nh.param<int>("n_cols_file",n_cols,86);
	nh.param<int>("quality_index",quality_index,0);
	nh.param<int>("number_of_fingers", n_fingers, 5);
	nh.param<std::string>("frame_name_root", frame_name_root, "world");
	nh.param<std::string>("frame_name_thumb", frame_name_finger[0], "right_hand_index_distal_link" );
	nh.param<std::string>("frame_name_index", frame_name_finger[1], "right_hand_little_distal_link" );
	nh.param<std::string>("frame_name_middle", frame_name_finger[2], "right_hand_middle_distal_link");
	nh.param<std::string>("frame_name_ring", frame_name_finger[3], "right_hand_ring_distal_link");
	nh.param<std::string>("frame_name_little", frame_name_finger[4], "right_hand_thumb_distal_link");
	nh.param<std::string>("file_name", relative_path_file, "/db/box_db_2.csv" );
	nh.param<int>("type_of_contact", type_of_contact, 0); 
	nh.param<double>("joint_stiffness",joint_stiffness,0);
	nh.param<double>("contact_stiffness",contact_stiffness,0);
	nh.param<std::vector<double>>("synergie", synergie, std::vector<double>{1});
	nh.param<double>("mu", mu, 0.03);
	nh.param<double>("f_i_max", f_i_max, 1);
	nh.param<int>("set_PCR_PGR", set_PCR_PGR, 0);
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

	int first_element_joint_array = 17;
	int number_of_joints = 33;
	int first_element_cp_array = first_element_joint_array + number_of_joints;

	int count = 0; 
	int count_contact = 0;
	int count_line = 0;

	
	for(std::string line; getline( file, line, '\n' ); ) // for each line
	{
		std::vector<double> values_inline;
    	std::istringstream iss_line(line);	
    	for(std::string value; getline(iss_line, value, ',' ); )
    		values_inline.push_back(stod(value));

    	int quante_colonne = 0;


    	cout << "-----------------------" << endl;
    	cout << " line : " <<  count_line << endl;
    	cout << "-----------------------" << endl;




 /*   	for(int i = 0 ; i < values_inline.size(); i++)
    	{	cout << " line " << count_line << " : colonna : " << quante_colonne << " = " << values_inline[i] << endl; quante_colonne++;}
    		
    	cout << " numero di colonne : " << quante_colonne << endl;
 */   	
    	KDL::Vector trasl_w_T_o(0,0, (values_inline[2]/2)); // i m not sure if it is corrected
   		KDL::Vector trasl_o_T_p(values_inline[3],values_inline[4],values_inline[5]);
    	KDL::Rotation R_o_T_p = Rotation::Quaternion(values_inline[6],values_inline[7],values_inline[8],values_inline[9]);
   	
    	KDL::Frame w_T_o(trasl_w_T_o);
    	KDL::Frame o_T_p(R_o_T_p,trasl_o_T_p);
    	KDL::Frame p_T_h(chains_hand_finger[0].getSegment(5).getFrameToTip());

		KDL::Frame w_T_h = w_T_o * o_T_p * p_T_h ;
    	KDL::Vector trasl_w_T_h = w_T_h.p;
    	KDL::Rotation R_w_T_h = w_T_h.M;

    	double roll , pitch , yaw ;

    	R_w_T_h.GetRPY(roll,pitch,yaw);


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
    			q_finger[i](6) = values_inline[first_element_joint_array]; // joint around z-axis knuckle

				q_finger[i](7) = values_inline[first_element_joint_array+1] ; // joint around y-axis knuckle
				q_finger[i](8) = values_inline[first_element_joint_array+2] ; // joint around y-axis proximal

				q_finger[i](9) = values_inline[first_element_joint_array+3] ; // joint around y-axis proximal
				q_finger[i](10) = values_inline[first_element_joint_array+4] ; // joint around y-axis distal

				k_+=5;
			}
			else
			{
				q_finger[i](6) = values_inline[first_element_joint_array+k_]; // joint around z-axis knuckle

				q_finger[i](7) = values_inline[first_element_joint_array+k_+1] ; // joint around y-axis knuckle
				q_finger[i](8) = values_inline[first_element_joint_array+k_+2] ; // joint around y-axis proximal

				q_finger[i](9) = values_inline[first_element_joint_array+k_+3] ; // joint around y-axis proximal
				q_finger[i](10) = values_inline[first_element_joint_array+k_+4] ;  // joint around y-axis middle

				q_finger[i](11) = values_inline[first_element_joint_array+k_+5] ;  // joint around y-axis middle
				q_finger[i](12) = values_inline[first_element_joint_array+k_+6] ;  // joint around y-axis distal

				k_+=7;
			}			
		}
/*		for ( int i = 0 ; i < n_fingers ; i++)
		{	cout << " finger " << i << endl;
			cout << q_finger[i].data << endl; }
*/
		Eigen::MatrixXd Contacts(n_c_max,3);
		int gap_coordinate = 0;

		for(int i = 0; i < n_c_max; i++)
		{						
			for(int j = 0; j < 3; j++ )
				Contacts(i,j) = values_inline[first_element_cp_array + gap_coordinate + j];  // check if it is correct
			gap_coordinate += 3;
		}

		cout << " Box : " << endl ;
		cout <<  " x : " << values_inline[0] << " y : " << values_inline[1] << " z : " << values_inline[2] << endl;
 

		cout << "Contacts : " << endl;
		cout << Contacts << endl;


		int n_c = 0;
		for(int i = 0; i < n_c_max; i++)
			if(!std::isnan(Contacts(i,0)))	
				n_c++;
	



		if(n_c > 0)
		{
    		//for each contact point 
			Eigen::MatrixXd Grasp_Matrix_b  = MatrixXd::Zero(6,6*n_c);		// G
			Eigen::MatrixXd Grasp_Matrix_c  = MatrixXd::Zero(6,6*n_c);  	// G
    		Eigen::MatrixXd Hand_Jacobian_  = MatrixXd::Zero(6*n_c, n_q);	// J
    		Eigen::MatrixXd R_contact_hand_object_ = MatrixXd::Zero(3*n_c,3*n_c);

   			int k = 1;
   			int step = 0;	
   			int s_ = 0;	
			for(int i = 0 ; i < n_c_max ; i++) //calc the grasp_matrix and hand_jacobian for each contact point
    		{
    			if(!std::isnan(Contacts(i,0)))	
    			{
        			Eigen::MatrixXd Grasp_Matrix(6,6);  
 					Eigen::MatrixXd Skew_Matrix(3,3);
  					Eigen::MatrixXd Rotation(3,3);
  					Eigen::MatrixXd b_Rotation_c(3,3);

  					Rotation = MatrixXd::Identity(3,3);

	 				//check if the values ​​of the skew matrix are expressed in the correct reference system
    	  			Skew_Matrix(0,0) = Skew_Matrix(1,1) = Skew_Matrix(2,2) = 0;
     				Skew_Matrix(0,1) = - Contacts(i,2); // -rz    
     				Skew_Matrix(0,2) = Contacts(i,1);   // ry
        			Skew_Matrix(1,0) = Contacts(i,2);   // rz
        			Skew_Matrix(2,0) = - Contacts(i,1); // -ry
        			Skew_Matrix(1,2) = - Contacts(i,0); // -rx
       		 		Skew_Matrix(2,1) = Contacts(i,0);   // rx

        			Grasp_Matrix.block<3,3>(0,0) = Rotation;
     		   		Grasp_Matrix.block<3,3>(3,3) = Rotation;
     		   		Grasp_Matrix.block<3,3>(3,0) = Skew_Matrix * Rotation;
        			Grasp_Matrix.block<3,3>(0,3) = MatrixXd::Zero(3,3);

	      			Grasp_Matrix_b.block<6,6>(0,step) = Grasp_Matrix;

  					normal_component(b_Rotation_c, values_inline[0]/2, values_inline[1]/2, values_inline[2]/2,Contacts(i,0),Contacts(i,1),Contacts(i,2));


  					

	      			//check if the values ​​of the skew matrix are expressed in the correct reference system
    	  			Skew_Matrix(0,0) = Skew_Matrix(1,1) = Skew_Matrix(2,2) = 0;
     				Skew_Matrix(0,1) = - Contacts(i,2); // -rz    
     				Skew_Matrix(0,2) = Contacts(i,1);   // ry
        			Skew_Matrix(1,0) = Contacts(i,2);   // rz
        			Skew_Matrix(2,0) = - Contacts(i,1); // -ry
        			Skew_Matrix(1,2) = - Contacts(i,0); // -rx
       		 		Skew_Matrix(2,1) = Contacts(i,0);   // rx

        			Grasp_Matrix.block<3,3>(0,0) = b_Rotation_c;
     		   		Grasp_Matrix.block<3,3>(3,3) = b_Rotation_c;
     		   		Grasp_Matrix.block<3,3>(3,0) = Skew_Matrix * b_Rotation_c;
        			Grasp_Matrix.block<3,3>(0,3) = MatrixXd::Zero(3,3);

        			Grasp_Matrix_c.block<6,6>(0,step) = Grasp_Matrix;





       				for(int n = 0 ; n < n_fingers ; n++)//calc jacobian for each finger
       					jnt_to_jac_solver[n]->JntToJac(q_finger[n], hand_jacob[n], -1);

    				int which_finger = 0;
      				int which_phalanx = 0;

	           		if( i==0 ) which_phalanx = 5;
	           		if( i==1 || i==5 || i==9  || i==13 || i==17 )  which_phalanx = 7;
	           		if( i==2 || i==6 || i==10 || i==14 || i==18)  which_phalanx = 9;
	           		if( i==3 || i==7 || i==11 || i==15 || i==19)  which_phalanx = 11;
	           		if( i==4 || i==8 || i==12 || i==16 ) which_phalanx = 13;

	           		if( i >= 0 && i <= 4 ) which_finger = 0;
	           		if( i >= 5 && i <= 8 ) which_finger = 1;
	           		if( i >= 9 && i <= 12 ) which_finger = 2;
	           		if( i >= 13 && i <= 16 ) which_finger = 3;
	           		if( i >= 17 && i <= 19 ) which_finger = 4;









///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////	Ks = RtKsR				///////////////////
//// contact //////////////////////////////////////////////////////////////////////////////////////
					Eigen::MatrixXd h_T_o = MatrixXd::Identity(4,4);

					Eigen::MatrixXd R_k(3,3);
					double qx = values_inline[13];
  					double qy = values_inline[14];
  					double qz = values_inline[15];
  					double qw = values_inline[16];
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
					h_T_o(0,3) = values_inline[10];
					h_T_o(1,3) = values_inline[11];
					h_T_o(2,3) = values_inline[12];




					Eigen::MatrixXd o_T_c = MatrixXd::Identity(4,4);

         			o_T_c.block<3,3>(0,0) = b_Rotation_c;
  					o_T_c(0,3) = Contacts(i,0);
  					o_T_c(1,3) = Contacts(i,1);
  					o_T_c(2,3) = Contacts(i,2);




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


	           		// cout << " Mano contact : " << endl << h_T_c_mano << endl;
	           		// cout << " Object contact : " << endl << h_T_c << endl;

	           		Eigen::MatrixXd R_hand_contact(3,3);
	           		R_hand_contact = h_T_c_mano.block<3,3>(0,0);


	           		Eigen::MatrixXd R_object_contact(3,3);
	           		R_object_contact = h_T_c.block<3,3>(0,0);


	           		Eigen::MatrixXd R_contact_hand_object = R_hand_contact.transpose() * R_object_contact;
	           		cout << " R_contact_hand_object " << endl << R_contact_hand_object << endl; // it is identity 


	           		R_contact_hand_object_.block<3,3>(s_,s_) = R_contact_hand_object; // for PGR
			 

	           		cout << " R_contact_hand_object (3*n_c,3*n_c)" << endl << R_contact_hand_object_ << endl;


	           		KDL::Chain chain_contact;	           		
	           		KDL::Frame frame_relative_contact;
	           		string name_which_phalanx = chains_hand_finger[which_finger].getSegment(which_phalanx).getName();


	           		KDL::Rotation c_R_e(c_T_e(0,0),c_T_e(0,1),c_T_e(0,2),c_T_e(1,0),c_T_e(1,1),c_T_e(1,2),c_T_e(2,0),c_T_e(2,1),c_T_e(2,2));
	           		KDL::Vector p_(c_T_e(0,3), c_T_e(1,3), c_T_e(2,3));

	           		// cout << "c_T_e" << endl << c_T_e << endl;

	           		frame_relative_contact.M = c_R_e;
					frame_relative_contact.p = p_;


	           		hand_tree.getChain(frame_name_root, name_which_phalanx, chain_contact);      
           		
					chain_contact.addSegment(Segment(Joint(Joint::None), frame_relative_contact));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					////////////			Jacobian   mod



	           		boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_relative_contact;
					jnt_to_jac_solver_relative_contact.reset(new KDL::ChainJntToJacSolver(chain_contact));


	           
	        //  		cout << "which_finger : " << which_finger << endl;
	        // 			cout << "which_phalanx : " << which_phalanx << endl;

	           		jnt_to_jac_solver_relative_contact->JntToJac(q_finger[which_finger],hand_jacob[which_finger]);
	  		// 		cout << " Jacobian thumb "  << endl;
			//		cout << hand_jacob[4].data  << endl;
  			// 		cout << " Jacobian index "  << endl;
	  		// 		cout << hand_jacob[0].data  << endl;
			// 		cout << " Jacobian middle " << endl;
	  		// 		cout << hand_jacob[2].data  << endl;
					// cout << " Jacobian ring "   << endl;
					// cout << hand_jacob[3].data  << endl;
					// cout << " Jacobian little " << endl;
					// cout << hand_jacob[1].data  << endl;
					// cout << " _______________ " << endl;

					Hand_Jacobian_.block<6,6>(step,0) = hand_jacob[0].data.topLeftCorner(6,6);	// 6 
					Hand_Jacobian_.block<6,7>(step,6) = hand_jacob[0].data.topRightCorner(6,7); // 7
					Hand_Jacobian_.block<6,7>(step,13)= hand_jacob[1].data.topRightCorner(6,7); // 7
					Hand_Jacobian_.block<6,7>(step,20)= hand_jacob[2].data.topRightCorner(6,7); // 7
					Hand_Jacobian_.block<6,7>(step,27)= hand_jacob[3].data.topRightCorner(6,7); // 7
	  				Hand_Jacobian_.block<6,5>(step,34)= hand_jacob[4].data.topRightCorner(6,5); // 5

	  				s_ += 3;
	  				step += 6;
	  			} // if i have contact
        	} // end for(n_max) each possible contact point	
        	// cout << "................................................................." << endl;
        	// cout << "FINAL GRASP MATRIX  " << endl;
        	// cout << Grasp_Matrix_ << endl;
        	// cout << "................................................................." << endl;

    
//////////////////////////	  now i have a 
/////////////////////////////   G          grasp matrix and 
/////////////////////////////   J          hand jacobian for 6D
/////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	        if(quality_index < 6) quality_i = quality(quality_index, Grasp_Matrix_b, Hand_Jacobian_); 		

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    		if(quality_index == 6 ) // PCR PGR 
    		{
	    		int first_element_contact_force = 110;
	    		int number_force_contact = 120;

   				Eigen::VectorXd contact_force_ = VectorXd::Zero(6*n_c_max);
   				Eigen::VectorXd contact_force_b  = VectorXd::Zero(6*n_c);
   				Eigen::VectorXd contact_force_c_h = VectorXd::Zero(3*n_c);
   				Eigen::VectorXd contact_force_c = VectorXd::Zero(3*n_c);


   				int  j= 0;
    			for(int i = 0 ; i < number_force_contact ; i++)
    				if(!std::isnan(values_inline[first_element_contact_force + i]))
    				{	contact_force_(j) = (values_inline[first_element_contact_force + i]); j++; }

    			for(int i = 0 ; i < (6*n_c) ; i++)
    				contact_force_b(i) = contact_force_(i);

				Eigen::MatrixXd c_R_b = MatrixXd::Zero(3*n_c, 3*n_c);
				Eigen::MatrixXd c_R_b_6 = MatrixXd::Zero(6*n_c, 6*n_c);

				int step = 0;
				int step_ = 0;
				for(int i = 0 ; i < n_c ; i++)
				{
					Eigen::MatrixXd b_Rotation_c  = Grasp_Matrix_c.block<3,3>(0,step_);
					c_R_b.block<3,3>(step, step) = b_Rotation_c;

					c_R_b_6.block<3,3>(step, step) = b_Rotation_c;
					c_R_b_6.block<3,3>(step+3, step+3) = MatrixXd::Identity(3,3);
					

					step += 3;
					step_ += 6;
				}


				contact_force_c = c_R_b_6 * contact_force_b;

				cout << " ----------------------------- " << endl;
				cout << " contact_force B : " << endl << contact_force_b << endl;
				cout << " ----------------------------- " << endl;

				cout << " ----------------------------- " << endl;
				cout << " contact_force C : " << endl << contact_force_c << endl;
				cout << " ----------------------------- " << endl;


///////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////


				
				int stepp = 0;
				int n_c_ = 0;

				for (int i = 0 ; i < n_c ; i++)
				{
					Eigen::VectorXd cf(3);

					cf(0) = contact_force_c(stepp+0);
					cf(1) = contact_force_c(stepp+1);
					cf(2) = contact_force_c(stepp+2);

					if(cf.norm() >= 1e-3 && cf(2) >= 0)
						n_c_++;


					stepp +=6 ;

				}


				cout << " qui 1" << endl;



				Eigen::MatrixXd G_c(6,6*n_c_);
				Eigen::VectorXd f_c(6*n_c_);
				Eigen::MatrixXd R_c = MatrixXd::Identity(3*n_c_, 3*n_c_);
				Eigen::MatrixXd J_c(6*n_c_, n_q);


				cout << " qui 2 " << endl;



				int stepp_ = 0;
				int stepp_3 = 0;
				int stepp_now= 0;
				int stepp_now_3 = 0;
				for(int i = 0 ; i < n_c ; i++)
				{

					Eigen::VectorXd cf(3);

					cf(0) = contact_force_c(stepp_now+0);
					cf(1) = contact_force_c(stepp_now+1);
					cf(2) = contact_force_c(stepp_now+2);

					cout << " qui 4" << endl;
					cout << " --------------- --" << endl;
					cout << " c_f : " << endl << cf << endl;
					cout << "-------------------" << endl;

					cout << " cf.norm : " << cf.norm() << endl;

					cout << " cf(2) : " << cf(2) << endl;

					if(cf.norm() >= 1e-3 && cf(2) >= 0)
					{
						 cout << " qui 5 " << endl ;

						G_c.block<6,6>(0,stepp_) = Grasp_Matrix_c.block<6,6>(0,stepp_now);
						R_c.block<3,3>(stepp_3,stepp_3) = R_contact_hand_object_.block<3,3>(stepp_now_3,stepp_now_3);
						J_c.block(stepp_,0,6,n_q) = Hand_Jacobian_.block(stepp_now, 0, 6,n_q);
						
						for(int j = 0 ; j < 6 ; j++)
						{	f_c(stepp_+j) = contact_force_c(stepp_now+j);
							// cout << " qui riempio " << endl;	
						}
						stepp_3 += 3;
						stepp_ += 6;
					}
					// cout << "qui 7" << endl;
					stepp_now_3 += 3 ;
					stepp_now += 6 ;
				}

				cout << " G_c : " << endl << G_c << endl;
				cout << " R_c : " << endl << R_c << endl;
				cout << " f_c : " << endl << f_c << endl;
				
				cout << " n_c_ : " << endl << n_c_ << endl;

    			Eigen::MatrixXd Contact_Stiffness_Matrix = MatrixXd::Zero(3,3);		// Kis
    			for(int i = 0 ; i < Contact_Stiffness_Matrix.rows() ; i++) //Kis
    				Contact_Stiffness_Matrix(i,i) = contact_stiffness;


    			Eigen::MatrixXd Joint_Stiffness_Matrix = MatrixXd::Zero(n_q,n_q);    // Kp    				
    			for(int j = 0 ; j < Joint_Stiffness_Matrix.rows() ; j++) //Kp
    				Joint_Stiffness_Matrix(j,j) = joint_stiffness;

    			double quality_final = 0;
    			//double quality_final = quality_pcr_pgr_3(contact_force_c, Grasp_Matrix_c, Hand_Jacobian_, R_contact_hand_object_, Contact_Stiffness_Matrix, Joint_Stiffness_Matrix, mu, f_i_max);
    			

    			if(n_c_> 0)
    				quality_final = quality_pcr_pgr_3(f_c, G_c, J_c, R_c, Contact_Stiffness_Matrix, Joint_Stiffness_Matrix, mu, f_i_max);
    			else
    				quality_final = -44;

    			quality_i = quality_final;
    		}
    		else
    			quality_i = 50; // no contact
    	}// end if ( n_c > 0)





	    cout << " THEFINALCOUNTDOWN:::: " << quality_i << endl;

	    file_output << quality_i ;
		for(int i = 0 ; i < 10 ; i++)
	    	file_output << ' ' << i+1 << ":" << values_inline[i] ;
      	file_output << ' ' << endl;

    	count_line++;
    	quality_i = 0;	
	} // end for each line	

	file_output.close();

	cout << endl;
	cout << "Count rows : " << count_line << endl;
	cout << "quality_index : " << quality_index << endl;
	
	cout << " YEAH ENJOY " << endl;
	cout << "   fine   " << endl;

	ros::spinOnce();
	return 0;

}