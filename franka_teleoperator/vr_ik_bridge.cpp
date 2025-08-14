// VR IK Bridge - Python bindings for Franka VR teleoperation
// Integrates weighted IK solver with TCP VR receiver for LeRobot

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "include/weighted_ik.h"
#include "include/geofik.h"
#include <Eigen/Dense>

namespace py = pybind11;

struct VRPose {
    std::array<double, 3> position;
    std::array<double, 4> quaternion;  // x, y, z, w
    std::string fist_state;
    bool valid = false;
    std::chrono::steady_clock::time_point timestamp;
};

struct VRTeleopConfig {
    int tcp_port = 8000;
    double smoothing_factor = 0.7;
    double position_deadzone = 0.001;   // 1mm
    double orientation_deadzone = 0.03; // ~1.7 degrees
    double max_position_offset = 0.75;  // 75cm from initial position
    bool verbose = false;
};

class VRIKBridge {
private:
    // VR TCP receiver
    int tcp_socket_ = -1;
    int client_socket_ = -1;
    std::atomic<bool> running_{false};
    std::thread tcp_thread_;
    
    // VR state
    VRPose current_vr_pose_;
    VRPose initial_vr_pose_;
    VRPose filtered_vr_pose_;
    std::mutex vr_mutex_;
    std::atomic<bool> vr_initialized_{false};
    
    // Robot pose tracking
    Eigen::Affine3d initial_robot_pose_;
    std::atomic<bool> robot_initialized_{false};
    
    // IK solver
    std::unique_ptr<WeightedIKSolver> ik_solver_;
    
    // Configuration
    VRTeleopConfig config_;
    
    // Q7 limits (can be configured for different end effectors)
    double q7_min_ = -2.89;  // Full Franka range
    double q7_max_ = 2.89;
    double q7_search_range_ = 0.5;  // Balanced range to allow movement while preventing large jumps
    double q7_optimization_tolerance_ = 1e-6;
    int q7_max_iterations_ = 100;

public:
    VRIKBridge(const VRTeleopConfig& config = VRTeleopConfig()) 
        : config_(config) {
        
        // Initialize filtered pose
        filtered_vr_pose_.position = {0.0, 0.0, 0.0};
        filtered_vr_pose_.quaternion = {0.0, 0.0, 0.0, 1.0};
        
        if (config_.verbose) {
            std::cout << "VRIKBridge initialized with TCP port " << config_.tcp_port << std::endl;
        }
    }
    
    ~VRIKBridge() {
        stop();
    }
    
    bool start_tcp_server() {
        if (running_) {
            std::cerr << "TCP server already running" << std::endl;
            return false;
        }
        
        // Setup TCP socket
        tcp_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (tcp_socket_ < 0) {
            std::cerr << "Failed to create TCP socket" << std::endl;
            return false;
        }
        
        // Allow socket reuse
        int opt = 1;
        setsockopt(tcp_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(config_.tcp_port);
        
        if (bind(tcp_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Failed to bind TCP socket to port " << config_.tcp_port << std::endl;
            close(tcp_socket_);
            return false;
        }
        
        if (listen(tcp_socket_, 1) < 0) {
            std::cerr << "Failed to listen on TCP socket" << std::endl;
            close(tcp_socket_);
            return false;
        }
        
        running_ = true;
        tcp_thread_ = std::thread(&VRIKBridge::tcp_receiver_thread, this);
        
        if (config_.verbose) {
            std::cout << "VR TCP server started on port " << config_.tcp_port << std::endl;
        }
        
        return true;
    }
    
    void stop() {
        if (running_) {
            running_ = false;
            
            if (client_socket_ >= 0) {
                close(client_socket_);
                client_socket_ = -1;
            }
            
            if (tcp_socket_ >= 0) {
                close(tcp_socket_);
                tcp_socket_ = -1;
            }
            
            if (tcp_thread_.joinable()) {
                tcp_thread_.join();
            }
            
            if (config_.verbose) {
                std::cout << "VR TCP server stopped" << std::endl;
            }
        }
    }
    
    bool setup_ik_solver(const std::array<double, 7>& neutral_pose,
                        double manipulability_weight = 1.0,
                        double neutral_distance_weight = 2.0, 
                        double current_distance_weight = 2.0,
                        const std::array<double, 7>& joint_weights = {{2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0}}) {
        
        try {
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_pose,
                manipulability_weight,
                neutral_distance_weight,
                current_distance_weight,
                joint_weights,
                config_.verbose
            );
            
            if (config_.verbose) {
                std::cout << "IK solver initialized with neutral pose" << std::endl;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to setup IK solver: " << e.what() << std::endl;
            return false;
        }
    }
    
    void set_q7_limits(double q7_min, double q7_max) {
        q7_min_ = q7_min;
        q7_max_ = q7_max;
        
        if (config_.verbose) {
            std::cout << "Q7 limits set to [" << q7_min_ << ", " << q7_max_ << "]" << std::endl;
        }
    }
    
    void set_initial_robot_pose(const std::array<double, 16>& transformation_matrix) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Map(transformation_matrix.data());
        initial_robot_pose_ = Eigen::Affine3d(T);
        robot_initialized_ = true;
        
        if (config_.verbose) {
            std::cout << "Initial robot pose set" << std::endl;
        }
    }
    
    bool is_vr_connected() const {
        return vr_initialized_ && current_vr_pose_.valid;
    }
    
    bool is_ready() const {
        return vr_initialized_ && robot_initialized_ && ik_solver_ != nullptr;
    }
    
    // Main function: get joint targets given current joint state
    std::array<double, 7> get_joint_targets(const std::array<double, 7>& current_joints) {
        if (!is_ready()) {
            if (config_.verbose && !is_ready()) {
                std::cout << "VRIKBridge not ready: vr=" << vr_initialized_ 
                         << " robot=" << robot_initialized_ 
                         << " ik=" << (ik_solver_ != nullptr) << std::endl;
            }
            return current_joints;  // Return current joints if not ready
        }
        
        // Get current VR pose with thread safety
        VRPose vr_pose;
        {
            std::lock_guard<std::mutex> lock(vr_mutex_);
            vr_pose = filtered_vr_pose_;
        }
        
        if (!vr_pose.valid) {
            return current_joints;  // No valid VR data
        }
        
        // Calculate VR pose delta from initial
        Eigen::Vector3d vr_pos_delta(
            vr_pose.position[0] - initial_vr_pose_.position[0],
            vr_pose.position[1] - initial_vr_pose_.position[1], 
            vr_pose.position[2] - initial_vr_pose_.position[2]
        );
        
        // Apply workspace limits
        if (vr_pos_delta.norm() > config_.max_position_offset) {
            vr_pos_delta = vr_pos_delta.normalized() * config_.max_position_offset;
        }
        
        // Calculate target position: initial robot + VR delta
        Eigen::Vector3d target_position = initial_robot_pose_.translation() + vr_pos_delta;
        
        // Calculate orientation delta (matching franka_logger_client.cpp approach)
        Eigen::Quaterniond initial_vr_quat(
            initial_vr_pose_.quaternion[3], initial_vr_pose_.quaternion[0],
            initial_vr_pose_.quaternion[1], initial_vr_pose_.quaternion[2]
        );
        Eigen::Quaterniond current_vr_quat(
            vr_pose.quaternion[3], vr_pose.quaternion[0],
            vr_pose.quaternion[1], vr_pose.quaternion[2]
        );
        
        // Calculate VR orientation delta: current * initial^-1
        Eigen::Quaterniond vr_quat_delta = current_vr_quat * initial_vr_quat.inverse();
        
        // Apply delta to initial robot orientation: delta * initial_robot_orientation
        Eigen::Quaterniond target_orientation = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        target_orientation.normalize();
        
        // Convert to arrays for IK solver
        std::array<double, 3> target_pos = {target_position.x(), target_position.y(), target_position.z()};
        
        Eigen::Matrix3d rot_matrix = target_orientation.toRotationMatrix();
        std::array<double, 9> target_rot = {
            rot_matrix(0,0), rot_matrix(0,1), rot_matrix(0,2),
            rot_matrix(1,0), rot_matrix(1,1), rot_matrix(1,2),
            rot_matrix(2,0), rot_matrix(2,1), rot_matrix(2,2)
        };
        
        // Calculate Q7 search range around current value
        double current_q7 = current_joints[6];
        double q7_start = std::max(q7_min_, current_q7 - q7_search_range_);
        double q7_end = std::min(q7_max_, current_q7 + q7_search_range_);
        
        // Debug logging for IK inputs (when verbose)
        if (config_.verbose) {
            static int debug_counter = 0;
            debug_counter++;
            if (debug_counter % 50 == 0) {  // Log every 50 calls (~2Hz at 100Hz)
                std::cout << "IK Input - Target pos: [" 
                         << std::fixed << std::setprecision(3)
                         << target_pos[0] << ", " << target_pos[1] << ", " << target_pos[2] << "]" << std::endl;
                std::cout << "VR delta: [" 
                         << vr_pos_delta.x() << ", " << vr_pos_delta.y() << ", " << vr_pos_delta.z() << "]" << std::endl;
                std::cout << "Q7 range: [" << q7_start << ", " << q7_end << "] around " << current_q7 << std::endl;
            }
        }
        
        // Solve IK
        WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
            target_pos, target_rot, current_joints,
            q7_start, q7_end, q7_optimization_tolerance_, q7_max_iterations_
        );
        
        if (ik_result.success) {
            if (config_.verbose) {
                static int success_counter = 0;
                success_counter++;
                if (success_counter % 50 == 0) {
                    std::cout << "IK Success - Joint targets: [";
                    for (int i = 0; i < 7; i++) {
                        std::cout << std::fixed << std::setprecision(3) << ik_result.joint_angles[i];
                        if (i < 6) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
            return ik_result.joint_angles;
        } else {
            if (config_.verbose) {
                std::cout << "IK failed, returning current joints" << std::endl;
            }
            return current_joints;  // IK failed, stay at current position
        }
    }
    
    // Get current VR pose info (for debugging)
    py::dict get_vr_status() {
        py::dict status;
        status["connected"] = is_vr_connected();
        status["initialized"] = vr_initialized_.load();
        
        if (vr_initialized_) {
            std::lock_guard<std::mutex> lock(vr_mutex_);
            status["position"] = py::cast(current_vr_pose_.position);
            status["quaternion"] = py::cast(current_vr_pose_.quaternion);
            status["fist_state"] = current_vr_pose_.fist_state;
            status["valid"] = current_vr_pose_.valid;
        }
        
        return status;
    }

private:
    void tcp_receiver_thread() {
        if (config_.verbose) {
            std::cout << "VR TCP receiver thread started" << std::endl;
        }
        
        while (running_) {
            try {
                // Wait for client connection
                if (client_socket_ < 0) {
                    if (config_.verbose) {
                        std::cout << "Waiting for VR TCP connection..." << std::endl;
                    }
                    
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    
                    client_socket_ = accept(tcp_socket_, (struct sockaddr*)&client_addr, &client_len);
                    if (client_socket_ < 0) {
                        if (running_) {
                            std::cerr << "Failed to accept VR client connection" << std::endl;
                        }
                        continue;
                    }
                    
                    if (config_.verbose) {
                        std::cout << "VR client connected from " << inet_ntoa(client_addr.sin_addr) << std::endl;
                    }
                }
                
                // Receive VR data
                char buffer[1024];
                ssize_t bytes_received = recv(client_socket_, buffer, sizeof(buffer) - 1, 0);
                
                if (bytes_received <= 0) {
                    // Client disconnected
                    if (config_.verbose) {
                        std::cout << "VR client disconnected" << std::endl;
                    }
                    close(client_socket_);
                    client_socket_ = -1;
                    continue;
                }
                
                buffer[bytes_received] = '\0';
                parse_vr_message(std::string(buffer));
                
            } catch (const std::exception& e) {
                std::cerr << "Error in VR TCP receiver: " << e.what() << std::endl;
                if (client_socket_ >= 0) {
                    close(client_socket_);
                    client_socket_ = -1;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        if (config_.verbose) {
            std::cout << "VR TCP receiver thread stopped" << std::endl;
        }
    }
    
    void parse_vr_message(const std::string& message) {
        // Parse VR message format: "Right wrist:, x, y, z, qx, qy, qz, qw, leftFist: state"
        // Real format: "Right wrist:, 0.123, 0.750, 0.263, -0.234, -0.435, -0.295, 0.818, leftFist: open"
        
        size_t wrist_pos = message.find("Right wrist:");
        if (wrist_pos == std::string::npos) {
            return; // Not a wrist message
        }
        
        try {
            // Use comma-separated parsing for the actual format
            std::string data_part = message.substr(wrist_pos + 13); // Skip "Right wrist:,"
            
            // Remove commas and parse as space-separated
            std::string clean_data = data_part;
            std::replace(clean_data.begin(), clean_data.end(), ',', ' ');
            
            std::istringstream iss(clean_data);
            
            VRPose new_pose;
            
            // Parse position and quaternion (7 numbers)
            iss >> new_pose.position[0] >> new_pose.position[1] >> new_pose.position[2];
            iss >> new_pose.quaternion[0] >> new_pose.quaternion[1] 
                >> new_pose.quaternion[2] >> new_pose.quaternion[3];
            
            // Parse fist state - find "leftFist:" and get the next token
            std::string token;
            while (iss >> token) {
                if (token == "leftFist:") {
                    iss >> new_pose.fist_state;
                    break;
                }
            }
            
            new_pose.valid = true;
            new_pose.timestamp = std::chrono::steady_clock::now();
            
            // Apply VR coordinate transformation (from logger_node.py)
            // VR: +x=right, +y=up, +z=forward → Robot: +x=forward, +y=left, +z=up
            std::array<double, 3> robot_position = {
                new_pose.position[2],   // Robot X = VR Z (forward)
                -new_pose.position[0],  // Robot Y = -VR X (left = -right)
                new_pose.position[1]    // Robot Z = VR Y (up)
            };
            new_pose.position = robot_position;
            
            // Apply quaternion transformation to match logger_node.py exactly
            // Convert VR quaternion to rotation matrix
            Eigen::Quaterniond vr_quat(new_pose.quaternion[3], new_pose.quaternion[0], 
                                      new_pose.quaternion[1], new_pose.quaternion[2]);
            Eigen::Matrix3d vr_matrix = vr_quat.toRotationMatrix();
            
            // Define coordinate transformation matrix from VR to robot
            // VR: [right, up, forward] → Robot: [forward, left, up]  
            // This maps: VR_x→Robot_y, VR_y→Robot_z, VR_z→Robot_x
            // Include handedness flip by negating one axis
            Eigen::Matrix3d transform_matrix;
            transform_matrix << 0,  0,  1,   // Robot X = VR Z (forward)
                               -1,  0,  0,   // Robot Y = -VR X (left = -right)  
                                0,  1,  0;   // Robot Z = VR Y (up)
            
            // Apply transformation: R_robot = T * R_vr * T^-1
            Eigen::Matrix3d robot_matrix = transform_matrix * vr_matrix * transform_matrix.transpose();
            
            // Convert back to quaternion and store in new_pose
            Eigen::Quaterniond robot_quat(robot_matrix);
            robot_quat.normalize();
            new_pose.quaternion = {robot_quat.x(), robot_quat.y(), robot_quat.z(), robot_quat.w()};
            
            // Apply smoothing
            if (vr_initialized_) {
                double alpha = 1.0 - config_.smoothing_factor;
                
                // Smooth position
                for (int i = 0; i < 3; i++) {
                    filtered_vr_pose_.position[i] = config_.smoothing_factor * filtered_vr_pose_.position[i] + 
                                                   alpha * new_pose.position[i];
                }
                
                // Smooth quaternion (simplified - could use slerp)
                for (int i = 0; i < 4; i++) {
                    filtered_vr_pose_.quaternion[i] = config_.smoothing_factor * filtered_vr_pose_.quaternion[i] + 
                                                     alpha * new_pose.quaternion[i];
                }
                
                // Normalize quaternion
                double norm = 0.0;
                for (int i = 0; i < 4; i++) {
                    norm += filtered_vr_pose_.quaternion[i] * filtered_vr_pose_.quaternion[i];
                }
                norm = std::sqrt(norm);
                for (int i = 0; i < 4; i++) {
                    filtered_vr_pose_.quaternion[i] /= norm;
                }
                
            } else {
                // First VR data - set as initial pose
                filtered_vr_pose_ = new_pose;
                initial_vr_pose_ = new_pose;
                vr_initialized_ = true;
                
                if (config_.verbose) {
                    std::cout << "Initial VR pose captured!" << std::endl;
                }
            }
            
            filtered_vr_pose_.fist_state = new_pose.fist_state;
            filtered_vr_pose_.valid = true;
            filtered_vr_pose_.timestamp = new_pose.timestamp;
            
            {
                std::lock_guard<std::mutex> lock(vr_mutex_);
                current_vr_pose_ = filtered_vr_pose_;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing VR message: " << e.what() << std::endl;
        }
    }
};

// Python bindings
PYBIND11_MODULE(vr_ik_bridge, m) {
    m.doc() = "VR IK Bridge for Franka teleoperation with LeRobot";
    
    // VRTeleopConfig struct
    py::class_<VRTeleopConfig>(m, "VRTeleopConfig")
        .def(py::init<>())
        .def_readwrite("tcp_port", &VRTeleopConfig::tcp_port)
        .def_readwrite("smoothing_factor", &VRTeleopConfig::smoothing_factor)
        .def_readwrite("position_deadzone", &VRTeleopConfig::position_deadzone)
        .def_readwrite("orientation_deadzone", &VRTeleopConfig::orientation_deadzone)
        .def_readwrite("max_position_offset", &VRTeleopConfig::max_position_offset)
        .def_readwrite("verbose", &VRTeleopConfig::verbose);
    
    // WeightedIKResult struct
    py::class_<WeightedIKResult>(m, "WeightedIKResult")
        .def_readonly("success", &WeightedIKResult::success)
        .def_readonly("joint_angles", &WeightedIKResult::joint_angles);
    
    // VRIKBridge class
    py::class_<VRIKBridge>(m, "VRIKBridge")
        .def(py::init<const VRTeleopConfig&>(), py::arg("config") = VRTeleopConfig())
        .def("start_tcp_server", &VRIKBridge::start_tcp_server)
        .def("stop", &VRIKBridge::stop)
        .def("setup_ik_solver", &VRIKBridge::setup_ik_solver,
             py::arg("neutral_pose"),
             py::arg("manipulability_weight") = 1.0,
             py::arg("neutral_distance_weight") = 2.0,
             py::arg("current_distance_weight") = 2.0,
             py::arg("joint_weights") = std::array<double, 7>{{2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0}})
        .def("set_q7_limits", &VRIKBridge::set_q7_limits)
        .def("set_initial_robot_pose", &VRIKBridge::set_initial_robot_pose)
        .def("is_vr_connected", &VRIKBridge::is_vr_connected)
        .def("is_ready", &VRIKBridge::is_ready)
        .def("get_joint_targets", &VRIKBridge::get_joint_targets)
        .def("get_vr_status", &VRIKBridge::get_vr_status);
}