// VR-Based Cartesian Teleoperation
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <array>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <iomanip>

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>

#include "examples_common.h"
#include "weighted_ik.h"
#include <ruckig/ruckig.hpp>

struct LoggerCommand
{
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    int record = 0;
    bool has_valid_data = false;
};

struct RobotStep
{
    double timestamp;  // Relative to recording start (starts at 0.0)
    
    // Observations (robot state)
    std::array<double, 7> joint_positions;    // q[7] 
    std::array<double, 7> joint_velocities;   // dq[7]
    std::array<double, 16> ee_pose;           // O_T_EE[16] - 4x4 transformation matrix
    
    // Actions (velocity commands sent to robot)
    std::array<double, 7> velocity_commands;  // target joint velocities
    
    // Future: gripper state (not implemented yet)
    // std::array<double, 2> gripper_state;   // gripper width, force
};

class FrankaLogger
{
private:
    std::atomic<bool> running_{true};
    LoggerCommand current_command_;
    std::mutex command_mutex_;
    
    // Joint logging state
    std::atomic<bool> is_recording_{false};
    int last_record_state_ = 0;
    std::vector<RobotStep> robot_log_buffer_;
    std::mutex log_mutex_;
    std::chrono::steady_clock::time_point recording_start_time_;
    std::chrono::steady_clock::time_point last_log_time_;
    static constexpr double LOG_INTERVAL_SEC = 0.04; // 25Hz logging

    int server_socket_;
    const int PORT = 8888;

    // VR mapping parameters
    struct VRParams
    {
        double vr_smoothing = 0.05;       // Less for more responsive control

        // Deadzones to prevent drift from small sensor noise
        double position_deadzone = 0.001;   // 1mm
        double orientation_deadzone = 0.03; // ~1.7 degrees

        // Workspace limits to keep the robot in a safe area
        double max_position_offset = 0.75;   // 75cm from initial position
    } params_;

    // VR Target Pose
    Eigen::Vector3d vr_target_position_;
    Eigen::Quaterniond vr_target_orientation_;

    // VR filtering state
    Eigen::Vector3d filtered_vr_position_{0, 0, 0};
    Eigen::Quaterniond filtered_vr_orientation_{1, 0, 0, 0};

    // Initial poses used as a reference frame
    Eigen::Affine3d initial_robot_pose_;
    Eigen::Vector3d initial_vr_position_{0, 0, 0};
    Eigen::Quaterniond initial_vr_orientation_{1, 0, 0, 0};
    bool vr_initialized_ = false;

    // Joint space tracking
    std::array<double, 7> current_joint_angles_;
    std::array<double, 7> neutral_joint_pose_;
    std::unique_ptr<WeightedIKSolver> ik_solver_;
    
    // Q7 limits
    double Q7_MIN;
    double Q7_MAX;
    static constexpr double Q7_SEARCH_RANGE = 0.75; // look for q7 angle candidates in +/- this value in the current joint range 
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6; // Tolerance for optimization
    static constexpr int Q7_MAX_ITERATIONS = 100; // Max iterations for optimization

    // Ruckig trajectory generator for smooth joint space motion
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7> ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;
    
    // Gradual activation to prevent sudden movements
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5; // Faster activation
    
    // Franka joint limits for responsive teleoperation 
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY = {1.5, 1.5, 1.8, 1.8, 2.0, 2.0, 2.0};     // Increase for responsiveness
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0}; // Increase for snappier response
    static constexpr std::array<double, 7> MAX_JOINT_JERK = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};  // Higher jerk for snappier response
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

public:
    FrankaLogger(bool bidexhand = false)
        : Q7_MIN(bidexhand ? -0.2 : -2.89), Q7_MAX(bidexhand ? 1.9 : 2.89)
    {
        setupNetworking();
        robot_log_buffer_.reserve(10000); // Pre-allocate for ~6.7 minutes at 25Hz
        
        // Create state_log directory if it doesn't exist
        std::filesystem::create_directories("state_log");
    }

    ~FrankaLogger()
    {
        running_ = false;
        close(server_socket_);
    }

    void setupNetworking()
    {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0)
        {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(PORT);

        if (bind(server_socket_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            throw std::runtime_error("Failed to bind socket");
        }

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT << " for robot pose data" << std::endl;
    }

    void networkThread()
    {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_)
        {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr *)&client_addr, &client_len);

            if (bytes_received > 0)
            {
                buffer[bytes_received] = '\0';

                LoggerCommand cmd;
                int parsed_count = sscanf(buffer, "%lf %lf %lf %lf %lf %lf %lf %d",
                                          &cmd.pos_x, &cmd.pos_y, &cmd.pos_z,
                                          &cmd.quat_x, &cmd.quat_y, &cmd.quat_z, &cmd.quat_w, &cmd.record);

                if (parsed_count == 8)
                {
                    cmd.has_valid_data = true;

                    std::lock_guard<std::mutex> lock(command_mutex_);
                    current_command_ = cmd;
                    
                    // Handle recording state transitions
                    handleRecordingStateChange(cmd.record);

                    if (!vr_initialized_)
                    {
                        initial_vr_position_ = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                        initial_vr_orientation_ = Eigen::Quaterniond(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                        filtered_vr_position_ = initial_vr_position_;
                        filtered_vr_orientation_ = initial_vr_orientation_;

                        vr_initialized_ = true;
                        std::cout << "VR reference pose initialized!" << std::endl;
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // Handle recording state transitions (0->1 start, 1->0 stop and save)
    void handleRecordingStateChange(int new_record_state)
    {
        if (last_record_state_ == 0 && new_record_state == 1)
        {
            // Start recording
            std::lock_guard<std::mutex> lock(log_mutex_);
            robot_log_buffer_.clear();
            is_recording_ = true;
            recording_start_time_ = std::chrono::steady_clock::now();
            last_log_time_ = recording_start_time_;
            std::cout << "Started robot data logging at 25Hz" << std::endl;
        }
        else if (last_record_state_ == 1 && new_record_state == 0)
        {
            // Stop recording and save
            is_recording_ = false;
            saveJointLog();
            std::cout << "Stopped robot data logging and saved file" << std::endl;
        }
        last_record_state_ = new_record_state;
    }
    
    // Log robot state and commands at 25Hz during recording
    void logRobotStep(const franka::RobotState& robot_state, const std::array<double, 7>& velocity_commands)
    {
        if (!is_recording_) return;
        
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_since_last_log = std::chrono::duration<double>(current_time - last_log_time_).count();
        
        // Log immediately when recording starts or at regular intervals
        bool is_first_sample = robot_log_buffer_.empty();
        
        if (is_first_sample || elapsed_since_last_log >= LOG_INTERVAL_SEC)
        {
            std::lock_guard<std::mutex> lock(log_mutex_);
            
            RobotStep step;
            
            // Relative timestamp 
            step.timestamp = std::chrono::duration<double>(current_time - recording_start_time_).count();
            
            // Observations: robot state
            for (int i = 0; i < 7; i++) {
                step.joint_positions[i] = robot_state.q[i];
                step.joint_velocities[i] = robot_state.dq[i];
            }
            
            // End-effector pose (4x4 transformation matrix)
            for (int i = 0; i < 16; i++) {
                step.ee_pose[i] = robot_state.O_T_EE[i];
            }
            
            // Actions: velocity commands
            for (int i = 0; i < 7; i++) {
                step.velocity_commands[i] = velocity_commands[i];
            }
            
            robot_log_buffer_.push_back(step);
            last_log_time_ = current_time;
        }
    }
    
    // Save robot log to CSV file
    void saveJointLog()
    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        if (robot_log_buffer_.empty()) {
            std::cout << "No robot data to save" << std::endl;
            return;
        }
        
        // Generate filename with timestamp and length
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        std::ostringstream filename_stream;
        filename_stream << "state_log/" 
                       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
                       << "_len" << robot_log_buffer_.size() << ".csv";
        std::string filename = filename_stream.str();
        
        // Write CSV file
        std::ofstream csv_file(filename);
        if (!csv_file.is_open()) {
            std::cerr << "Failed to create log file: " << filename << std::endl;
            return;
        }
        
        // Write header
        csv_file << "timestamp,"
                 << "joint0_pos,joint1_pos,joint2_pos,joint3_pos,joint4_pos,joint5_pos,joint6_pos,"
                 << "joint0_vel,joint1_vel,joint2_vel,joint3_vel,joint4_vel,joint5_vel,joint6_vel,"
                 << "joint0_cmd,joint1_cmd,joint2_cmd,joint3_cmd,joint4_cmd,joint5_cmd,joint6_cmd,"
                 << "ee_00,ee_01,ee_02,ee_03,ee_10,ee_11,ee_12,ee_13,ee_20,ee_21,ee_22,ee_23,ee_30,ee_31,ee_32,ee_33\n";
        
        // Write data
        for (const auto& step : robot_log_buffer_) {
            csv_file << std::fixed << std::setprecision(6) << step.timestamp;
            
            // Joint positions
            for (int i = 0; i < 7; i++) {
                csv_file << "," << std::setprecision(6) << step.joint_positions[i];
            }
            
            // Joint velocities
            for (int i = 0; i < 7; i++) {
                csv_file << "," << std::setprecision(6) << step.joint_velocities[i];
            }
            
            // Velocity commands (actions)
            for (int i = 0; i < 7; i++) {
                csv_file << "," << std::setprecision(6) << step.velocity_commands[i];
            }
            
            // End-effector pose (4x4 matrix)
            for (int i = 0; i < 16; i++) {
                csv_file << "," << std::setprecision(6) << step.ee_pose[i];
            }
            
            csv_file << "\n";
        }
        
        csv_file.close();
        std::cout << "Saved " << robot_log_buffer_.size() << " robot steps to: " << filename << std::endl;
    }
    
    // This function's only job is to calculate the desired target pose from VR data.
    void updateVRTargets(const LoggerCommand &cmd)
    {
        if (!cmd.has_valid_data || !vr_initialized_)
        {
            return;
        }

        // Current VR pose
        Eigen::Vector3d vr_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond vr_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        vr_quat.normalize();

        // Smooth incoming VR data to reduce jitter
        double alpha = 1.0 - params_.vr_smoothing;
        filtered_vr_position_ = params_.vr_smoothing * filtered_vr_position_ + alpha * vr_pos;
        filtered_vr_orientation_ = filtered_vr_orientation_.slerp(alpha, vr_quat);

        // Calculate deltas from the initial VR pose
        Eigen::Vector3d vr_pos_delta = filtered_vr_position_ - initial_vr_position_;
        Eigen::Quaterniond vr_quat_delta = filtered_vr_orientation_ * initial_vr_orientation_.inverse();

        // Apply deadzones to prevent drift
        if (vr_pos_delta.norm() < params_.position_deadzone)
        {
            vr_pos_delta.setZero();
        }
        double rotation_angle = 2.0 * acos(std::abs(vr_quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone)
        {
            vr_quat_delta.setIdentity();
        }

        // Apply workspace limits
        if (vr_pos_delta.norm() > params_.max_position_offset)
        {
            vr_pos_delta = vr_pos_delta.normalized() * params_.max_position_offset;
        }

        // The final calculation just updates the vr_target_
        vr_target_position_ = initial_robot_pose_.translation() + vr_pos_delta;
        vr_target_orientation_ = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        vr_target_orientation_.normalize();
    }

    // Helper function to clamp q7 within limits
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }
    
    // Convert Eigen types to arrays for geofik interface
    std::array<double, 3> eigenToArray3(const Eigen::Vector3d& vec) const {
        return {vec.x(), vec.y(), vec.z()};
    }
    
    std::array<double, 9> quaternionToRotationArray(const Eigen::Quaterniond& quat) const {
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        return {rot(0,0), rot(0,1), rot(0,2),
                rot(1,0), rot(1,1), rot(1,2), 
                rot(2,0), rot(2,1), rot(2,2)};
    }

public:
    void run(const std::string &robot_ip)
    {
        try
        {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);

            // Move to a suitable starting joint configuration
            std::array<double, 7> q_goal = {{0.0,
                                             -0.48,
                                             0.0,
                                             -2.0,
                                             0.0,
                                             1.57,
                                             -0.85}};
            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This command will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // Collision behavior
            robot.setCollisionBehavior(
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}}, {{100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 60.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
                {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});

            // Joint impedance for smooth motion (instead of Cartesian)
            robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

            // Initialize poses from the robot's current state
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));
            
            // Initialize joint angles
            for (int i = 0; i < 7; i++) {
                current_joint_angles_[i] = state.q[i];
                neutral_joint_pose_[i] = state.q[i];  // Use the initial joint configuration as neutral
            }
            
            // Create IK solver with neutral pose and weights
            // Joint weights for base stabilization: higher weights for base joints (0,1)
            std::array<double, 7> base_joint_weights = {{
                2.0,  // Joint 0 (base rotation) - high penalty for stability
                2.0,  // Joint 1 (base shoulder) - high penalty for stability  
                1.0,  // Joint 2 (elbow) - normal penalty
                1.0,  // Joint 3 (forearm) - normal penalty
                1.0,  // Joint 4 (wrist) - normal penalty
                1.0,  // Joint 5 (wrist) - normal penalty
                1.0   // Joint 6 (hand) - normal penalty
            }};
            
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_,
                1.0,  // manipulability weight
                2.0,  // neutral distance weight  
                2.0,  // current distance weight
                base_joint_weights,  // per-joint weights for base stabilization
                false // verbose = false for production use
            );
            
            // Initialize Ruckig trajectory generator (but don't set initial state yet)
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            
            // Set up joint limits for safe teleoperation (but don't set positions yet)
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i] = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i] = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i] = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            
            std::cout << "Ruckig trajectory generator configured with 7 DOFs" << std::endl;

            // Initialize VR targets to the robot's starting pose
            vr_target_position_ = initial_robot_pose_.translation();
            vr_target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            std::thread network_thread(&FrankaLogger::networkThread, this);

            std::cout << "Waiting for VR data..." << std::endl;
            while (!vr_initialized_ && running_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (vr_initialized_)
            {
                std::cout << "VR initialized! Starting real-time control." << std::endl;
                this->runVRControl(robot);
            }

            running_ = false;
            if (network_thread.joinable())
                network_thread.join();
        }
        catch (const franka::Exception &e)
        {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    void runVRControl(franka::Robot &robot)
    {
        auto vr_control_callback = [this](
                                       const franka::RobotState &robot_state,
                                       franka::Duration period) -> franka::JointVelocities
        {
            // Update VR targets from latest command (~50Hz)
            LoggerCommand cmd;
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                cmd = current_command_;
            }
            
            // Log will be done after we have the velocity commands
            updateVRTargets(cmd);

            // Initialize Ruckig with actual robot state on first call
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = 0.0; // Start with zero velocity command
                    ruckig_input_.current_acceleration[i] = 0.0; // Start with zero acceleration
                    ruckig_input_.target_position[i] = robot_state.q[i]; // Start with current position as target
                    ruckig_input_.target_velocity[i] = 0.0; // Start with zero target velocity
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
                std::cout << "Ruckig initialized for velocity control!" << std::endl;
                std::cout << "Starting with zero velocity commands to smoothly take over control" << std::endl;
            } else {
                // Update current joint state for Ruckig using previous Ruckig output for continuity
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i] = robot_state.q[i];
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = ruckig_output_.new_velocity[i]; // Use our own velocity command for continuity
                    ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i]; // Use Ruckig's acceleration
                }
            }
            
            // Calculate activation factor for gradual activation
            auto current_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(current_time - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);
            
            // Debug output for velocity control
            static int debug_counter = 0;
            debug_counter++;
            
            // Solve IK for VR target pose to get target joint angles
            std::array<double, 3> target_pos = eigenToArray3(vr_target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(vr_target_orientation_);
            
            // Calculate q7 search range around current value
            double current_q7 = current_joint_angles_[6];
            // Use full Franka Q7 range for IK solving, not bidexhand limits
            double q7_start = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end = std::min(2.89, current_q7 + Q7_SEARCH_RANGE);
            
            
            // Solve IK with weighted optimization
            WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
                target_pos, target_rot, current_joint_angles_,
                q7_start, q7_end, Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS
            );
            
            if (debug_counter % 100 == 0) {
                std::cout << "IK: " << (ik_result.success ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m") << " | Joints: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(2) << current_joint_angles_[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << std::endl;
            }
            
            // Set Ruckig targets based on IK solution and gradual activation
            if (ruckig_initialized_) {
                if (ik_result.success) {
                    // Gradually blend from current position to IK solution for target position
                    for (int i = 0; i < 7; i++) {
                        double current_pos = current_joint_angles_[i];
                        double ik_target_pos = ik_result.joint_angles[i];
                        ruckig_input_.target_position[i] = current_pos + activation_factor * (ik_target_pos - current_pos);
                        // Always target zero velocity for smooth stops
                        ruckig_input_.target_velocity[i] = 0.0;
                    }
                    // Enforce q7 limits
                    ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
                }
                // If IK fails, keep previous targets (don't change target_position/velocity)
            }
            
            // Always run Ruckig to generate smooth velocity commands
            ruckig::Result ruckig_result = trajectory_generator_->update(ruckig_input_, ruckig_output_);
            
            std::array<double, 7> target_joint_velocities;
            
            if (ruckig_result == ruckig::Result::Working || ruckig_result == ruckig::Result::Finished) {
                // Use Ruckig's smooth velocity output
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency fallback: zero velocity to stop smoothly
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = 0.0;
                }
                if (debug_counter % 100 == 0) {
                    std::cout << "Ruckig error, using zero velocity for safety" << std::endl;
                }
            }
            
            // Log robot state and velocity commands
            logRobotStep(robot_state, target_joint_velocities);
            
            // Debug output for the first few commands
            // if (debug_counter <= 10 || debug_counter % 100 == 0) {
            //     std::cout << "Target vel: ";
            //     for (int i = 0; i < 7; i++) std::cout << std::fixed << std::setprecision(4) << target_joint_velocities[i] << " ";
            //     std::cout << " [activation: " << std::setprecision(3) << activation_factor << "]" << std::endl;
            // }

            if (!running_)
            {
                return franka::MotionFinished(franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
            }
            return franka::JointVelocities(target_joint_velocities);
        };

        try
        {
            robot.control(vr_control_callback);
        }
        catch (const franka::ControlException &e)
        {
            std::cerr << "VR control exception: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname> [bidexhand]" << std::endl;
        std::cerr << "  bidexhand: true (default) for BiDexHand limits, false for full range" << std::endl;
        return -1;
    }

    bool bidexhand = false;
    if (argc == 3)
    {
        std::string bidexhand_arg = argv[2];
        bidexhand = (bidexhand_arg == "true" || bidexhand_arg == "1");
    }

    try
    {
        FrankaLogger logger(bidexhand);
        // Add a signal handler to gracefully shut down on Ctrl+C
        logger.run(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}