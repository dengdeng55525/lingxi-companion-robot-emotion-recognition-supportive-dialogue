#include <iostream>
#include <string>

#include "geometry_msgs/Twist.h"
#include "ros/ros.h"
#include "std_msgs/String.h"

geometry_msgs::Twist twist;
ros::Publisher cmd_vel_pub;
std_msgs::String string_msg;
ros::Publisher tts_text_pub;
int cmd_type = 0;

void iattextCallback(const std_msgs::String::ConstPtr& msg)
{
    const char* text;
    const std::string dataString = msg->data;

    if (dataString.find("前进") != std::string::npos ||
        dataString.find("向前") != std::string::npos)
    {
        text = "开始前进";
        std::cout << text << std::endl;
        cmd_type = 1;
    }
    else if (dataString.find("后退") != std::string::npos ||
             dataString.find("向后") != std::string::npos)
    {
        text = "开始后退";
        std::cout << text << std::endl;
        cmd_type = 2;
    }
    else if (dataString.find("左转") != std::string::npos ||
             dataString.find("向左") != std::string::npos)
    {
        text = "开始左转";
        std::cout << text << std::endl;
        cmd_type = 3;
    }
    else if (dataString.find("右转") != std::string::npos ||
             dataString.find("向右") != std::string::npos)
    {
        text = "开始右转";
        std::cout << text << std::endl;
        cmd_type = 4;
    }
    else if (dataString.find("停止") != std::string::npos ||
             dataString.find("停车") != std::string::npos)
    {
        text = "停止";
        std::cout << text << std::endl;
        cmd_type = 5;
    }
    else
    {
        ROS_WARN("unrecognized command: %s", msg->data.c_str());
        text = msg->data.c_str();
    }

    string_msg.data = text;
    tts_text_pub.publish(string_msg);
}

int main(int argc, char* argv[])
{
    double control_speed = 0.0;
    double control_turn = 0.0;
    double linear_speed = 0.1;
    double angular_speed = 0.5;

    ros::init(argc, argv, "voice_control");
    ros::NodeHandle n;
    ros::NodeHandle private_nh("~");
    private_nh.param("linear_speed", linear_speed, linear_speed);
    private_nh.param("angular_speed", angular_speed, angular_speed);

    ros::Subscriber iat_text_sub = n.subscribe("iat_text", 1000, iattextCallback);
    tts_text_pub = n.advertise<std_msgs::String>("tts_text", 1000);
    cmd_vel_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1);

    ROS_INFO("Wait Command...");
    ros::Rate loop_rate(50);
    while (ros::ok())
    {
        switch (cmd_type)
        {
        case 1:
            control_speed = linear_speed;
            control_turn = 0.0;
            break;
        case 2:
            control_speed = -linear_speed;
            control_turn = 0.0;
            break;
        case 3:
            control_speed = linear_speed;
            control_turn = angular_speed;
            break;
        case 4:
            control_speed = linear_speed;
            control_turn = -angular_speed;
            break;
        case 5:
            control_speed = 0.0;
            control_turn = 0.0;
            break;
        default:
            break;
        }

        twist.linear.x = control_speed;
        twist.linear.y = 0.0;
        twist.linear.z = 0.0;
        twist.angular.x = 0.0;
        twist.angular.y = 0.0;
        twist.angular.z = control_turn;
        cmd_vel_pub.publish(twist);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
