#include "xc_aubo_arm_control/interfacetest.h"
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <fstream>


#define SERVER_HOST "192.168.1.5"
#define SERVER_PORT 8899

void Example_0::demo()
{
    ServiceInterface robotService;

    int ret = aubo_robot_namespace::InterfaceCallSuccCode;

    /** Interface call: login ***/
    ret = robotService.robotServiceLogin(SERVER_HOST, SERVER_PORT, "aubo", "123456");
    if(ret == aubo_robot_namespace::InterfaceCallSuccCode)
    {
        std::cout<<"login successful."<<std::endl;
    }
    else
    {
        std::cerr<<"login failed."<<std::endl;
    }


    /** If the real robot arm is connected, the arm needs to be initialized.　**/
    aubo_robot_namespace::ROBOT_SERVICE_STATE result;

    //Tool dynamics parameter
    aubo_robot_namespace::ToolDynamicsParam toolDynamicsParam;
    memset(&toolDynamicsParam, 0, sizeof(toolDynamicsParam));

    ret = robotService.rootServiceRobotStartup(toolDynamicsParam/**Tool dynamics parameter**/,
                                               6        /*Collision level*/,
                                               true     /*Whether to allow reading poses defaults to true*/,
                                               true,    /*Leave the default to true */
                                               1000,    /*Leave the default to 1000 */
                                               result); /*Robot arm initialization*/
    if(ret == aubo_robot_namespace::InterfaceCallSuccCode)
    {
        std::cout<<"Robot arm initialization succeeded."<<std::endl;
    }
    else
    {
        std::cerr<<"Robot arm initialization failed."<<std::endl;
    }

    /** Simulation business **/
    // std::cout<<"Simulation business..."<<std::endl;
    // sleep(5);
    aubo_robot_namespace::wayPoint_S intermediate_point;
    intermediate_point.cartPos.position.x = 0.14;
    intermediate_point.cartPos.position.y = 0.3;
    intermediate_point.cartPos.position.z = 0.38;
    intermediate_point.orientation.w = 0.001;
    intermediate_point.orientation.x = 0.001;
    intermediate_point.orientation.y = 0.707;
    intermediate_point.orientation.z = 0.707;
    intermediate_point.jointpos[0]={-17.6/180.0*M_PI};
    intermediate_point.jointpos[1]={31.2/180.0*M_PI};
    intermediate_point.jointpos[2]={103.9/180.0*M_PI};
    intermediate_point.jointpos[3]={61.9/180.0*M_PI};
    intermediate_point.jointpos[4]={0/180.0*M_PI};
    intermediate_point.jointpos[5]={0/180.0*M_PI};


    ret = robotService.robotServiceJointMove(intermediate_point, true);

    /** Robotic arm shutdown **/
    robotService.robotServiceRobotShutdown();

    /** Interface call: logout　**/
    robotService.robotServiceLogout();
}

