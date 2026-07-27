#include <iostream>
#include <atomic>
#include <communication/msg/motion_commands.hpp>
#include <linux/joystick.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using namespace std;

// XBOX joystick mapping reported by jstest for the deployed controller.
#define JS_VELX_AXIS 3
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1

#define JS_START_BT 11
#define JS_SWITCH_A 0
#define JS_SWITCH_B 1
#define JS_SWITCH_X 3
#define JS_SWITCH_Y 4

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

static void run_shell_command(const char *command)
{
    int ret = system(command);
    (void)ret;
}

static void stop_robot_processes()
{
    // Stopping the hardware driver makes the ROS launch file shut down the
    // controller and this remote cleanly. Signalling all three processes here
    // as well would deliver duplicate SIGINTs during launch teardown.
    run_shell_command(
        "killall -SIGINT hardware_elf3 bxi_example_hw 2>/dev/null");
}

class COMPublisher : public rclcpp::Node{
public:
    COMPublisher(const char *_js_dev) : Node("COM_publisher"){
        if (strlen(_js_dev) >= 128){
            printf("dev:%s error\n", _js_dev);
            exit(-1);
        }

        strcpy(_js_dev_name, _js_dev);
        
        while (rclcpp::ok()){
            js_fd = open(_js_dev_name, O_RDONLY | O_NONBLOCK);
            if (js_fd < 0){
                printf("open:%s failed\n", _js_dev_name);
                sleep(1);      
            }
            else{
                printf("open js dev: %s\n", _js_dev_name);
                break;
            }
        }

        if (!rclcpp::ok()){
            return;
        }
        
        com_pub = this->create_publisher<communication::msg::MotionCommands>("motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));
        js_loop_thread_ = std::thread(&COMPublisher::js_loop, this);
    }

    ~COMPublisher(){
        running_.store(false);
        if (js_loop_thread_.joinable()){
            js_loop_thread_.join();
        }
        if (js_fd >= 0){
            close(js_fd);
            js_fd = -1;
        }
    }

private:
    mutable std::mutex lock_;

    char _js_dev_name[128] = {0};
    int js_fd = -1;
    double js_axis[20] = {0};   // original data of js axis data
    double js_bt[20] = {0};    // original data of ja button data
    std::thread js_loop_thread_;
    std::atomic<bool> running_{true};

    double velxy[2] = {0};                      //x y速度       (x,y speed)
    double velxy_filt[2] = {0};                 //x y速度滤波值  (x,y speed filter)
    double stand_height = STAND_HEIGHT;
    double height_filt = STAND_HEIGHT;
    double velr = 0;                            //旋转速度       (rotation speed)
    double velr_filt = 0;

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
    bool vibration_flag = false;        // Y: 启动/停止吊挂振动测试
    bool joint_test_flag = false;       // A: 启动/停止整机关节测试

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
            message.btn_7 = joint_test_flag ? 1 : 0;
            // message.btn_8 =  

            // 纯按键
            message.btn_9 = dance_flag ? 1 : 0;
            message.btn_10 = vibration_flag ? 1 : 0;
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
        dance_flag = false;
        vibration_flag = false;
        joint_test_flag = false;
    }

    void handle_button_press(int button)
    {
        if (button == JS_START_BT){
            printf("EMERGENCY STOP: terminating robot programs\n");
            fflush(stdout);
            stop_robot_processes();
            reset_value();
        }
        else if (button == JS_SWITCH_X){
            const std::lock_guard<std::mutex> guard(lock_);
            dance_flag = !dance_flag;
            printf("dance_flag: %d\n", dance_flag);
        }
        else if (button == JS_SWITCH_Y){
            const std::lock_guard<std::mutex> guard(lock_);
            vibration_flag = !vibration_flag;
            printf("vibration_flag: %d\n", vibration_flag);
        }
        else if (button == JS_SWITCH_A){
            const std::lock_guard<std::mutex> guard(lock_);
            joint_test_flag = !joint_test_flag;
            printf("joint_test_flag: %d\n", joint_test_flag);
        }
        else if (button == JS_SWITCH_B){
            printf("B\n");
        }
    }

    void js_loop(){
        while (running_.load() && rclcpp::ok()){
            ssize_t len;
            struct js_event event;
            
            // 读取js端口数据到event (read js date to event)
            len = read(js_fd, &event, sizeof(event));

            if (len == sizeof(event)){
                if (event.type & JS_EVENT_AXIS){  // axis event
                    //printf("Axis: %d -> %d\n", (int)event.number, (int)event.value);
                    if (event.number < 20){
                        const std::lock_guard<std::mutex> guard(lock_);
                        js_axis[event.number] = event.value;
                    }
                }
                else if (event.type & JS_EVENT_BUTTON){ // button event
                    if (event.value){
                        handle_button_press(
                            static_cast<int>(event.number));
                    }
                }
                else{
                    printf("unknown event:%u\n", event.type);
                }
            }
            if (
                len < 0
                && (errno == EAGAIN || errno == EWOULDBLOCK)
            ){
                usleep(5000);
                continue;
            }
            if (len <= 0 && running_.load() && rclcpp::ok()){
                printf("js dev lost, retry\n");
                close(js_fd);
                js_fd = -1;
                while (running_.load() && rclcpp::ok()){
                    js_fd = open(_js_dev_name, O_RDONLY | O_NONBLOCK);
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
    auto node = std::make_shared<COMPublisher>("/dev/input/js0");
    if (rclcpp::ok()){
        rclcpp::spin(node);
    }
    node.reset();
    if (rclcpp::ok()){
        rclcpp::shutdown();
    }

    return 0;
}
