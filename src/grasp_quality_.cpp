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


#include <cmath>
#include <ctime>
#include <time.h>

#include "pseudo_inverse.h"




using namespace std;
using namespace Eigen;
using namespace KDL;







// take it from yaml file
int n_rows;
int n_cols;
int quality_index;
string relative_path_file;
string file_name;
std::vector<double> Quality;
int n_c_max = 19; // number of contact point, assumed that 1 contact point for each phalanx 





int main (int argc, char **argv)
{

	ros::init(argc, argv, "Hand_Jacobian");	// ROS node
	ros::NodeHandle nh;


  


  nh.param<int>("n_rows_file",n_rows,108);
  nh.param<int>("n_cols_file",n_cols,86);
  nh.param<int>("quality_index",quality_index,0);
  nh.param<std::string>("file_name", relative_path_file, "/db/box_db_2.csv" );

  std::string path = ros::package::getPath("grasp_learning");
  file_name = path + relative_path_file ;
  ifstream file(file_name); 


  std::cout << "file: " << file_name.c_str() << " is " << (file.is_open() == true ? "already" : "not") << " open" << std::endl;
  if(!file.is_open())
  return 0;



  ////////////////////////////////////////////////////////////////////////////////////
	 /////////////////////////////////////////////////////////////////////////

  //	TAKE DATA_BASE : load a file .csv and put the values into Eigen::MatrixXd

  ////////////////////////////////////////////////////////////////////////////////////
	 /////////////////////////////////////////////////////////////////////////



  int count_row = 0;
  int count_col = 0;

  bool count_cols_only_one = true;


  for(std::string line; getline ( file, line, '\n' ); ) // ciclo sulla riga
  {
    count_row++;
    std::istringstream iss_line(line);  

    if(count_cols_only_one)
    { 

      for(std::string value; getline(iss_line, value, ',' ); )
          count_col++;
      
      count_cols_only_one = false;
    } 
  }

  cout << "ROWS : " << count_row << endl; // 40
  cout << "COLS : " << count_col << endl; // 189



  Eigen::MatrixXd data_set(count_row,count_col);


	int i = 0;
	int j = 0;



	for(std::string line; getline( file, line, '\n' ); ) // ciclo sulla riga
	{

    std::istringstream iss_line(line);	
    for(std::string value; getline(iss_line, value, ',' ); )
    {
    	data_set(i,j) = stod(value);
    	j++;
    }

    //cout << endl;
    j=0;
    i++;
	}


	file.close();





//	cout << " DATA_SET" << data_set << endl;



///////////////////////////// 		END		///////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////

		



////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////

//								TAKE SOFT_HAND

////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////



  KDL::Tree hand_tree;

  std::string robot_desc_string;
  nh.param("robot_description", robot_desc_string, string());  // robot description is the name in the launch file 
  if (!kdl_parser::treeFromString(robot_desc_string, hand_tree))
  { ROS_ERROR("Failed to construct kdl tree"); return false;}


  KDL::Chain chains_hand_finger[5];
  KDL::Jacobian hand_jacob[5];

  
  //std::string root_name = "right_hand_softhand_base";
  //std::string root_name = "right_hand_palm_link";
  std::string root_name = "world";
  std::string end_chain_name[5];
  end_chain_name[0] = "right_hand_thumb_distal_link";
  end_chain_name[1] = "right_hand_index_distal_link";
  end_chain_name[2] = "right_hand_middle_distal_link";
  end_chain_name[3] = "right_hand_ring_distal_link";
  end_chain_name[4] = "right_hand_little_distal_link";


  hand_tree.getChain(root_name, end_chain_name[0], chains_hand_finger[0]);      //thumb
  hand_tree.getChain(root_name, end_chain_name[1], chains_hand_finger[1]);      //index
  hand_tree.getChain(root_name, end_chain_name[2], chains_hand_finger[2]);      //middle
  hand_tree.getChain(root_name, end_chain_name[3], chains_hand_finger[3]);      //ring
  hand_tree.getChain(root_name, end_chain_name[4], chains_hand_finger[4]);      //little


  int nq_hand = hand_tree.getNrOfJoints();  // 34
  int ns_hand = hand_tree.getNrOfSegments(); // 39


  cout << "number_of_joint_in_hand : " << nq_hand << endl;
  cout << "number_of_segment_in_hand : " << ns_hand << endl;




///////////////////////////// 		END		  ///////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////

		
		
////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////

//			CALCULATE THE GRASP_MATRIX AND HAND_JACOBIAN FOR EACH ROW
//	I supposed to have 19 joint variables and a single point of contact for each phalanx

////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////



	unsigned int nj_0 = chains_hand_finger[0].getNrOfJoints();  // 5
  unsigned int nj_1 = chains_hand_finger[1].getNrOfJoints();  // 7
  unsigned int nj_2 = chains_hand_finger[2].getNrOfJoints();  // 7
  unsigned int nj_3 = chains_hand_finger[3].getNrOfJoints();  // 7
  unsigned int nj_4 = chains_hand_finger[4].getNrOfJoints();  // 7 



  KDL::JntArray q_thumb = JntArray(nj_0);    // thumb
  KDL::JntArray q_index = JntArray(nj_1);    // forefinger
  KDL::JntArray q_middle = JntArray(nj_2);   // middlefinger
  KDL::JntArray q_ring = JntArray(nj_3);     // ringfinger
  KDL::JntArray q_little = JntArray(nj_4);   // littlefinger


  cout << "nj_0 : " << nj_0 << endl;
  cout << "nj_1 : " << nj_1 << endl;
  cout << "nj_2 : " << nj_2 << endl;
  cout << "nj_3 : " << nj_3 << endl;
  cout << "nj_4 : " << nj_4 << endl;





  boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_0;
  boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_1;
  boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_2;
  boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_3;
  boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_4;


  // constructs the kdl solvers in non-realtime
  jnt_to_jac_solver_0.reset(new KDL::ChainJntToJacSolver(chains_hand_finger[0]));
  jnt_to_jac_solver_1.reset(new KDL::ChainJntToJacSolver(chains_hand_finger[1]));
  jnt_to_jac_solver_2.reset(new KDL::ChainJntToJacSolver(chains_hand_finger[2]));
  jnt_to_jac_solver_3.reset(new KDL::ChainJntToJacSolver(chains_hand_finger[3]));
  jnt_to_jac_solver_4.reset(new KDL::ChainJntToJacSolver(chains_hand_finger[4]));





   		


  // resizes the joint state vectors in non-realtime
  q_thumb.resize(chains_hand_finger[0].getNrOfJoints());
  q_index.resize(chains_hand_finger[1].getNrOfJoints());
  q_middle.resize(chains_hand_finger[2].getNrOfJoints());
  q_ring.resize(chains_hand_finger[3].getNrOfJoints());
  q_little.resize(chains_hand_finger[4].getNrOfJoints());

  hand_jacob[0].resize(chains_hand_finger[0].getNrOfJoints());
  hand_jacob[1].resize(chains_hand_finger[1].getNrOfJoints());
  hand_jacob[2].resize(chains_hand_finger[2].getNrOfJoints());
 	hand_jacob[3].resize(chains_hand_finger[3].getNrOfJoints());
  hand_jacob[4].resize(chains_hand_finger[4].getNrOfJoints());




	for(int r = 0; r < n_rows ; r++)  // for each row calculation the grasp matrix and hand jacobian
	{
		//take the value of joints from data		
		// initialize jntarray then be able to calculate the Jacobian to that point of contact


		q_thumb(0) = data_set(r,10);

		q_thumb(1) = data_set(r,11) / 2;
		q_thumb(2) = data_set(r,11) / 2;

		q_thumb(3) = data_set(r,12) / 2;
		q_thumb(4) = data_set(r,12) / 2;



		q_index(0) = data_set(r,13);

		q_index(1) = data_set(r,14) / 2;
		q_index(2) = data_set(r,14) / 2;

		q_index(3) = data_set(r,15) / 2;
		q_index(4) = data_set(r,15) / 2;

		q_index(5) = data_set(r,16) / 2;
		q_index(6) = data_set(r,16) / 2;



		q_middle(0) = data_set(r,17);

		q_middle(1) = data_set(r,18) / 2;
		q_middle(2) = data_set(r,18) / 2;

		q_middle(3) = data_set(r,19) / 2;
		q_middle(4) = data_set(r,19) / 2;

		q_middle(5) = data_set(r,20) / 2;
		q_middle(6) = data_set(r,20) / 2;



		q_ring(0) = data_set(r,21);

		q_ring(1) = data_set(r,22) / 2;
		q_ring(2) = data_set(r,22) / 2;

		q_ring(3) = data_set(r,23) / 2;
		q_ring(4) = data_set(r,23) / 2;

		q_ring(5) = data_set(r,24) / 2;
		q_ring(6) = data_set(r,24) / 2;





		q_little(0) = data_set(r,25);

		q_little(1) = data_set(r,26) / 2;
		q_little(2) = data_set(r,26) / 2;

		q_little(3) = data_set(r,27) / 2;
		q_little(4) = data_set(r,27) / 2;

		q_little(5) = data_set(r,28) / 2;
		q_little(6) = data_set(r,28) / 2;






		Eigen::MatrixXd Contacts(n_c_max,3); // I build a matrix of contact points for convenience only
										 // each row is a contact point
										 // row 0, 1, 2 for thumb
										 // row 3, 4, 5, 6 for index
										 // row 7, 8, 9, 10 for middle
										 // row 11, 12, 13, 14 for ring
										 // row 15, 16, 17, 18 for little
										 // n_c = 19 ( number of contact)



		
	 int start = 29; // the first value for the position of the joint variables
	 int gap_point_coordinate = 0;

		for(int i = 0; i < n_c_max; i++)
		{	
			for(int j = 0; j < 3; j++ )
				Contacts(i,j) = data_set(r, start + gap_point_coordinate + j);  // check if it is correct

			gap_point_coordinate += 3;
		}


    cout << "Contacts : " << Contacts << endl;






		
    //for each contact point 
		std::vector<Eigen::MatrixXd> Grasp_Matrix_ ;  
    std::vector<Eigen::MatrixXd> Hand_Jacobian_ ;


    int k = 1;


    for(int i = 0 ; i < n_c_max ; i++) //calc the grasp_matrix and hand_jacobian for each contact point
    {

      if(Contacts(i,0) != 9999) //9999 similar to NaN in dataset
      {	

        Eigen::MatrixXd Grasp_Matrix(6,6);  
 				Eigen::MatrixXd Skew_Matrix(3,3);
  			Eigen::MatrixXd Rotation(3,3);

  			Rotation <<  MatrixXd::Identity(3,3); 


  				
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
        Grasp_Matrix.block<3,3>(0,3) = Skew_Matrix * Rotation;
        Grasp_Matrix.block<3,3>(3,0) = MatrixXd::Zero(3,3);


        cout << "Grasp_Matrix" << endl;
        cout << Grasp_Matrix << endl;


        Grasp_Matrix_.push_back(Grasp_Matrix);



        jnt_to_jac_solver_0->JntToJac(q_thumb, hand_jacob[0], 0);
      	jnt_to_jac_solver_1->JntToJac(q_index, hand_jacob[1], 0);
      	jnt_to_jac_solver_2->JntToJac(q_middle, hand_jacob[2], 0);
      	jnt_to_jac_solver_3->JntToJac(q_ring, hand_jacob[3], 0);
   			jnt_to_jac_solver_4->JntToJac(q_little, hand_jacob[4], 0);

   			cout << "Hand_Jacobian_ THUMB" << endl;
      	cout <<  hand_jacob[0].data << endl;
      	cout << "Hand_Jacobian_ INDEX" << endl;
    		cout <<  hand_jacob[1].data << endl;
  			cout << "Hand_Jacobian_ MIDDLE"<< endl;
   			cout <<  hand_jacob[2].data << endl;
      	cout << "Hand_Jacobian_ RING"  << endl;
    		cout <<  hand_jacob[3].data << endl;
  			cout << "Hand_Jacobian_ LITTLE"<< endl;
  			cout <<  hand_jacob[4].data << endl;



      	int which_finger = 0;
        int which_falange = 0;
            	


        if((0 <= i) && (i <= 2)){ which_finger = 0; which_falange = i+k; k++; if(i==2) k = 1;} // control flag which_falange there may be some error
        if((3 <= i) && (i <= 6)){ which_finger = 1; which_falange = i-3+k; k++; if(i==6) k = 1;}
        if((7 <= i) && (i <= 10)){ which_finger = 2; which_falange = i-7+k; k++; if(i==10) k = 1;}
        if((11 <= i) && (i <= 14)){ which_finger = 3; which_falange = i-11+k; k++; if(i==14) k = 1;}
        if((15 <= i) && (i <= 18)){ which_finger = 4; which_falange = i-15+k; k++; if(i==18) k = 1;}



        cout << "i : " << i << endl;
        cout << "which_finger : " << which_finger << endl;
        cout << "which_falange : " << which_falange << endl;


        switch(which_finger)
        {
          case 0: // thumb
      						jnt_to_jac_solver_0->JntToJac(q_thumb, hand_jacob[0], which_falange);
      						break;

	  			case 1: // index
      						jnt_to_jac_solver_1->JntToJac(q_index, hand_jacob[1], which_falange);
      						break;

	  			case 2: // middle
							jnt_to_jac_solver_2->JntToJac(q_middle, hand_jacob[2], which_falange);
      						break;

	  			case 3: // ring
      						jnt_to_jac_solver_3->JntToJac(q_ring, hand_jacob[3], which_falange);
	  						break;

	  			case 4: // little
      						jnt_to_jac_solver_4->JntToJac(q_little, hand_jacob[4], which_falange);
	  						break;
	  		}// end switch

	  		cout << " Jacobian thumb " << hand_jacob[0].data << endl ;
	  		cout << " Jacobian index " << hand_jacob[1].data << endl ;
	  	  cout << " Jacobian middle " << hand_jacob[2].data << endl ;
			  cout << " Jacobian ring " << hand_jacob[3].data << endl ;
			  cout << " Jacobian little " << hand_jacob[4].data << endl ;





	  		Hand_Jacobian_.push_back(hand_jacob[0].data); // 5
	  		Hand_Jacobian_.push_back(hand_jacob[1].data); // 7
	 		  Hand_Jacobian_.push_back(hand_jacob[2].data); // 7
	  		Hand_Jacobian_.push_back(hand_jacob[3].data); // 7
	  		Hand_Jacobian_.push_back(hand_jacob[4].data); // 7




      }// end if  contact
        	
		


      }	// end for contact


      cout << " Dim of the vectors of the matrices grasp " << Grasp_Matrix_.size() << endl;
      cout << " Dim of the hand jacobian " << Hand_Jacobian_.size() << endl;


      int n_c_eff = Grasp_Matrix_.size();

      
        
      Eigen::MatrixXd Hand_Jacobian_Contact(6*n_c_eff,nq_hand-1);
      Eigen::MatrixXd Grasp_Matrix_Contact(6,6*n_c_eff);



      int s = 0; // dimension 
      int s_ = 0;


      for(int i = 0; i < n_c_eff; i++)
      {

        Grasp_Matrix_Contact.block<6,6>(0,s) = Grasp_Matrix_[i];
        	
        cout << "i : " << i << endl;

      	Hand_Jacobian_Contact.block<6,5>(s,0) = Hand_Jacobian_[s_];
      	Hand_Jacobian_Contact.block<6,7>(s,5) = Hand_Jacobian_[s_+1];
      	Hand_Jacobian_Contact.block<6,7>(s,12) = Hand_Jacobian_[s_+2];
      	Hand_Jacobian_Contact.block<6,7>(s,19) = Hand_Jacobian_[s_+3];
      	Hand_Jacobian_Contact.block<6,7>(s,26) = Hand_Jacobian_[s_+4];
			
  			s_+=5;
       	s+=6;
      }

/*
        file_output << "Grasp_Matrix_Contact : " << endl;
        file_output <<  Grasp_Matrix_Contact << endl;
        file_output << "__________________________________________________" << endl;
        file_output << "Hand_Jacobian_Contact : " << endl;
        file_output << Hand_Jacobian_Contact << endl;
*/

        // GRASP JACOBIAN


      Eigen::MatrixXd GRASP_Jacobian(6,nq_hand-1);
      Eigen::MatrixXd Grasp_Matrix_pseudo(Grasp_Matrix_Contact.rows(),Grasp_Matrix_Contact.cols()) ;
      pseudo_inverse(Grasp_Matrix_Contact,Grasp_Matrix_pseudo);

      GRASP_Jacobian = Grasp_Matrix_pseudo.transpose() * Hand_Jacobian_Contact;


      cout << "GRASP_Jacobian : " << endl;
      cout << GRASP_Jacobian << endl;


      int quality_ = 9999;
      Eigen::VectorXd Singular;
      double sigma_min ;
		  double sigma_max ;



      switch(quality_index) 
      {
        
        case 0: // "minimum_singular_value_of_G"  Q = sigma_min(G)
				      {
        	   		JacobiSVD<MatrixXd> svd0(Grasp_Matrix_Contact, ComputeThinU | ComputeThinV);  
        				Singular = svd0.singularValues();

				    		Quality.push_back(Singular[Singular.size()-1]);
				      } 
        		  break;


        case 1: // "Volume of the ellipsoid in the wrench space"  Q = K sqrt(det(GG.t)) = k ( sigma_0 **** sigma_d)
        		  {
                Eigen::MatrixXd G_G_t = Grasp_Matrix_Contact * Grasp_Matrix_Contact.transpose();
					      JacobiSVD<MatrixXd> svd1(G_G_t, ComputeThinU | ComputeThinV);  
					      Singular = svd1.singularValues();

				        for(int i = 0 ; i < Singular.size() ; i++)
						        quality_ *= Singular[i];

					     Quality.push_back(quality_);
				      }
        		  break;


        case 2: // "Grasp isotropy index" Q = sigma_min(G) / sigma_max(G) 
        		 {
        			  JacobiSVD<MatrixXd> svd2(Grasp_Matrix_Contact, ComputeThinU | ComputeThinV);  
					      Singular = svd2.singularValues();
                sigma_min = Singular[Singular.size()-1];
					      sigma_max = Singular[0];

					      Quality.push_back(sigma_min/sigma_max);
				      }
				      break;



			case 3: // "Distance to singular configuration" Q = sigma_min(H) H = G.pseudo_inverse.transpose * J
				    {
					    JacobiSVD<MatrixXd> svd3(GRASP_Jacobian, ComputeThinU | ComputeThinV);  
					    Singular = svd3.singularValues();
            
              Quality.push_back(Singular[Singular.size()-1]);
				    }
				    break;



			case 4: // "Volume of manipulability ellipsoid" 
				{
					Eigen::MatrixXd H_H_t = GRASP_Jacobian * GRASP_Jacobian.transpose();

					JacobiSVD<MatrixXd> svd4(H_H_t, ComputeThinU | ComputeThinV);  

        		
					Singular = svd4.singularValues();

					
					for(int i = 0 ; i < Singular.size() ; i++)
						quality_ *= Singular[i];


					Quality.push_back(quality_);
				}
				break;


			case 5: // "Uniformity of transformations"
				{
					JacobiSVD<MatrixXd> svd5(GRASP_Jacobian, ComputeThinU | ComputeThinV);  

        			
					Singular = svd5.singularValues();

					sigma_min = Singular[Singular.size()-1];
					sigma_max = Singular[0];

					Quality.push_back(sigma_min/sigma_max);
				}
				break;
      }//end switch

        
    cout << "Quality : " << endl;


    for(int i = 0; i < Quality.size(); i++)
			cout << Quality[i] << ' '; 
    cout << endl;
	} 
  // end for each row


  

  cout << " YEAH ENJOY " << endl;
  cout << "   fine   " << endl;


	ros::spin();
	return 0;
}