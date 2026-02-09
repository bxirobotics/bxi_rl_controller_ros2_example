#include <iostream>
#include <communication/msg/motion_commands.hpp>
#include <linux/joystick.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using namespace std;

#if 0  //PS4 JS
#define JS_VELX_AXIS 4  // index of joystick data, used for axis contro, vel x
#define JS_VELX_AXIS_DIR -1  // if it's reverse
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6    // index of joystick data, used for rotation contro, vel of rotation
#define JS_VELR_AXIS_DIR -1

#define JS_STOP_BT 10    // index of joystick data, used for function contro: stop
#define JS_GAIT_STAND_BT 0
#define JS_GAIT_WALK_BT 2
#define JS_HEIGHT_UPPER_BT  1
#define JS_HEIGHT_LOWER_BT  3
#define JS_MODE_BT          5
#define JS_START_BT 9

#else //XBOX JS
#define JS_VELX_AXIS 3
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1

#define JS_STOP_BT          11      // 终止程序                     (button for stop programe)
#define JS_START_BT         14      // 启动程序                     (button for start programe)
#define JS_LB_BT            6       // LB功能                       (define LB button)
#define JS_RB_BT            7       // RB功能                       (define RB button)

#define JS_SWITCH_X         3       // 暂停或继续跳舞                 (stop or continue danceing)
                                    // 组合LB：跳舞                   (conbine with LB for change to dance)
                                    // 组合RB：正常模式，走路站立跑步    (conbine with RB for change to normal: walk stand and run)
#define JS_SWITCH_A         0       // 组合LB：HOST起身               (conbine with LB for change to dance)
                                    // 组合RB：零力                   (conbine with LB for change to dance)
#define JS_SWITCH_B         1       //
                                    // 组合RB：pd模式                 (conbine with LB for change to dance)
#define JS_SWITCH_Y         4       // 
                                    // 组合RB：初始位置模式            (conbine with LB for change to zero position)
#define JS_START2_BT        14
#endif

#define AXIS_DEAD_ZONE  1000

#define MIN_SPEED_X -0.5
#define MAX_SPEED_X 1.0
#define MIN_SPEED_Y -0.4
#define MAX_SPEED_Y 0.4
#define MIN_SPEED_R -0.6
#define MAX_SPEED_R 0.6

#define AXIS_VALUE_MAX 32767

#define STAND_HEIGHT 1.0
#define STAND_HEIGHT_MIN    1.0
#define STAND_HEIGHT_MAX    3.0

class COMPublisher : public rclcpp::Node{
public:
    COMPublisher(const char *_js_dev) : Node("COM_publisher"){
        if (strlen(_js_dev) >= 128){
            printf("dev:%s error\n", _js_dev);
            exit(-1);
        }

        strcpy(_js_dev_name, _js_dev);
        
        while (1){
            js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
            if (js_fd < 0){
                printf("open:%s failed\n", _js_dev_name);
                sleep(1);      
            }
            else{
                printf("open js dev: %s\n", _js_dev_name);
                break;
            }
        }
        
        com_pub = this->create_publisher<communication::msg::MotionCommands>("motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));
        js_loop_thread_ = std::thread(&COMPublisher::js_loop, this);
    }

    ~COMPublisher(){
        if (js_fd > 0){
            close(js_fd);
        }
    }

private:
    mutable std::mutex lock_;

    char _js_dev_name[128] = {0};
    int js_fd;
    double js_axis[20] = {0};   // original data of js axis data
    double js_bt[20] = {0};    // original data of ja button data
    std::thread js_loop_thread_;

    double velxy[2] = {0};                      //x y速度       (x,y speed)
    double velxy_filt[2] = {0};                 //x y速度滤波值  (x,y speed filter)
    double stand_height = STAND_HEIGHT;
    double height_filt = STAND_HEIGHT;
    double velr = 0;                            //旋转速度       (rotation speed)
    double velr_filt = 0;

    bool launch_lock = false;           // 防止多次启动程序，True时不允许启动程序。    (launch programe lock)

    bool LB_press = false;              // 长按改变状态，弹起恢复                   (pressed for change state, release for recover)
    bool RB_press = false;              // 长按改变状态，弹起恢复
    // 按下RB的变量
    bool normal_mode = false;           // 按下改变状态，切换为普通模式，站立走路跑步   (change to normal state,for stand run and walk)
    bool zero_torque_mode = false;      // 按下改变状态，切换为零力模式               (change to zero torque mode)
    bool pd_brake_mode = false;         // 按下改变状态，切换为pd抱死模式             (change to zero torque mode)
    bool initial_pos_mode = false;      // 按下改变状态，切换为初始位置模式            (set motors to zero position)
    // 按下LB的变量
    bool host_mode = false;             // 按下改变状态，切换为host起身模式           (change to host mode, for stand up)
    bool dance_mode = false;            // 按下改变状态，切换为跳舞模式               (change to dance mode)

    bool dance_flag = false;            // 按下改变状态，暂停或继续跳舞               (stop or continue dancing)

    double vel_offset = 0.0;

    // timer_callback to publish messages
    void timer_callback(){                 
        auto message = communication::msg::MotionCommands();{    // initialize a ROS2 message
            const std::lock_guard<std::mutex> guard(lock_);

            velxy[0] = (js_axis[JS_VELX_AXIS] * JS_VELX_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velxy[1] = (js_axis[JS_VELY_AXIS] * JS_VELY_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velr = (js_axis[JS_VELR_AXIS] * JS_VELR_AXIS_DIR) / (double)AXIS_VALUE_MAX;

            velxy[0] = fabs(velxy[0]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[0] : 0;
            velxy[1] = fabs(velxy[1]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[1] : 0;
            velr = fabs(velr) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velr : 0;
            
            //按定义最大速度缩放
            if (velxy[0] > 0){
                velxy[0] *= MAX_SPEED_X;
            }
            else if (velxy[0] < 0){
                velxy[0] *= -MIN_SPEED_X;
            }

            if (velxy[1] > 0){
                velxy[1] *= MAX_SPEED_Y;
            }
            else if (velxy[1] < 0){
                velxy[1] *= -MIN_SPEED_Y;
            }

            if (velr > 0){
                velr *= MAX_SPEED_R;
            }
            else if (velr < 0){
                velr *= -MIN_SPEED_R;
            }

            velxy_filt[0] = velxy[0] * 0.03 + velxy_filt[0] * 0.97;
            velxy_filt[1] = velxy[1] * 0.03 + velxy_filt[1] * 0.97;

            velr_filt = velr * 0.05 + velr_filt *  0.95;

            message.vel_des.x = velxy_filt[0] + vel_offset;
            message.vel_des.y = velxy_filt[1];
            message.yawdot_des = velr_filt;
            // message.mode = mode;

            // RB组合键
            message.btn_1 = normal_mode ? 1 : 0;
            message.btn_2 = zero_torque_mode ? 1 : 0;
            message.btn_3 = pd_brake_mode ? 1 : 0;
            message.btn_4 = initial_pos_mode ? 1 : 0;

            // LB组合键
            message.btn_5 = dance_mode ? 1 : 0;
            message.btn_6 = host_mode ? 1 : 0; 
            // message.btn_7 = 
            // message.btn_8 =  

            // 纯按键
            message.btn_9 = dance_flag ? 1 : 0;
            // message.btn_10 =
            // message.btn_11 = 
            // message.btn_12 = 

            height_filt = height_filt * 0.9 + stand_height * 0.1;
            message.height_des = height_filt;
        }

        com_pub->publish(message);
    }

    void reset_value()
    {
        const std::lock_guard<std::mutex> guard(lock_);
        memset(js_axis, 0, sizeof(js_axis));
        memset(velxy, 0, sizeof(velxy));
        memset(velxy_filt, 0, sizeof(velxy_filt));
        velr_filt = 0;
        height_filt = STAND_HEIGHT;
    }

    void js_loop(){
        while (1){
            ssize_t len;
            struct js_event event;
            
            // 读取js端口数据到event (read js date to event)
            len = read(js_fd, &event, sizeof(event));

            if (len == sizeof(event)){
                if (event.type & JS_EVENT_AXIS){  // axis event
                    //printf("Axis: %d -> %d\n", (int)event.number, (int)event.value);
                    js_axis[event.number] = event.value;
                }
                else if (event.type & JS_EVENT_BUTTON){ // button event
                    //printf("Button: %d -> %d\n", (int)event.number, (int)event.value);
                    if (event.value){
                        switch (event.number){
                        case JS_STOP_BT:{
                            system("killall -SIGINT robot_controller");
                            system("killall -SIGINT pt_main_thread");
                            system("killall -SIGINT bxi_example_py");
                            system("killall -SIGINT bxi_example_py_trunk");
                            system("killall -SIGINT bxi_example_py_ankle");
                            system("killall -SIGINT bxi_example_py_foot");
                            system("killall -SIGINT hardware");
                            system("killall -SIGINT hardware_trunk");
                            system("killall -SIGINT hardware_trunk_neck");
                            system("killall -SIGINT hardware_ankle");
                            printf("kill robot_controller\n");//robot_controller

                            launch_lock = false;

                            reset_value();
                        }break;

                        case JS_START_BT:{
                            if(launch_lock == false){
                                system("mkdir -p /var/log/bxi_log");
                                // // sim
                                // system("ros2 launch bxi_example_py_elf3 example_launch_demo.py > /var/log/bxi_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log  2>&1 &");
                                // // real
                                system("ros2 launch bxi_example_py_elf3 example_launch_demo_hw.py > /var/log/bxi_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log  2>&1 &");
                                printf("run robot\n");//robot_controller
                            
                                reset_value();

                                launch_lock = true;
                            }else{
                                printf("\nprograme already exist! stop launch!!\n\n");
                            }
                        }break;

                        case JS_LB_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            LB_press = true;
                        }break;

                        case JS_RB_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            RB_press = true;
                        }break;

                        case JS_SWITCH_X:{
                            if(LB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                dance_mode = !dance_mode;
                                printf("dance_mode: %d\n", dance_mode);
                            }else if(RB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                normal_mode = !normal_mode;
                                printf("normal mode: %d\n", normal_mode);
                            }else{
                                const std::lock_guard<std::mutex> guard(lock_);
                                dance_flag = !dance_flag;
                                printf("dance_flag: %d\n", dance_flag);
                            }
                        }break;

                        case JS_SWITCH_Y:{
                            if(LB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                printf("LB + Y\n");
                            }else if(RB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                initial_pos_mode = !initial_pos_mode;
                                printf("initial_pos_mode: %d\n", initial_pos_mode);
                            }else{
                                const std::lock_guard<std::mutex> guard(lock_);
                                printf("Y\n");
                            }
                        }break;

                        case JS_SWITCH_A:{
                            if(LB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                host_mode = !host_mode;
                                printf("host mode:%d\n", host_mode);
                            }else if(RB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                zero_torque_mode = !zero_torque_mode;
                                printf("zero torque mode:%d\n", zero_torque_mode);
                            }else{
                                const std::lock_guard<std::mutex> guard(lock_);
                                printf("A\n");
                            }
                        }break;

                        case JS_SWITCH_B:{
                            if(LB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                printf("LB + B\n");
                            }else if(RB_press){
                                const std::lock_guard<std::mutex> guard(lock_);
                                pd_brake_mode = !pd_brake_mode;
                                printf("pd brake mode:%d\n", pd_brake_mode);
                            }else{
                                const std::lock_guard<std::mutex> guard(lock_);
                                printf("B\n");
                            }
                        }break;

                        default:
                            break;
                        }
                    }
                    else if(!event.value){
                        switch(event.number){
                        case JS_LB_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            LB_press = false;
                        }break;

                        case JS_RB_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            RB_press = false;
                        }break;
                        
                        default:
                            break;
                        }
                    }
                }
                else{
                    printf("unknown event:%u\n", event.type);
                }
            }
            if (len <= 0){
                printf("js dev lost, retry\n");
                close(js_fd);
                while (1){
                    js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
                    if (js_fd < 0){
                        printf("open:%s failed\n", _js_dev_name);
                        sleep(1);
                    }
                    else{
                        printf("open js dev: %s\n", _js_dev_name);
                        break;
                    }
                }
            }
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub;
};

int main(int argc, const char *argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>("/dev/input/js0"));
    rclcpp::shutdown();

    return 0;
}
