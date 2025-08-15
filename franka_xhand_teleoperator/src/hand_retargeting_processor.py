#!/usr/bin/env python3
"""
Hand Retargeting Processor - Wrapper for dex-retargeting pipeline
Processes VR landmarks data into XHand joint targets using dex-retargeting
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add dex-retargeting paths
dex_retargeting_path = Path(__file__).parent.parent / "dex_retargeting"
sys.path.insert(0, str(dex_retargeting_path))
sys.path.insert(0, str(dex_retargeting_path / "src"))
sys.path.insert(0, str(dex_retargeting_path / "example" / "vector_retargeting"))

logger = logging.getLogger(__name__)

class HandRetargetingProcessor:
    """
    Wrapper for dex-retargeting to process VR landmarks data.
    Converts VR hand landmarks to XHand joint targets using retargeting.
    """
    
    def __init__(self, config=None):
        """Initialize the hand retargeting processor"""
        self.config = config or {}
        self.verbose = self.config.get('verbose', False)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.7)
        
        # State tracking
        self.is_initialized = False
        self.retargeting = None
        self.retargeting_to_sapien = None
        self.last_target_joints = None
        self.hand_type = None
        self.robot_name = None
        
        # XHand joint mapping (will be populated during setup)
        self.xhand_joint_order = None
        
        logger.info("Hand retargeting processor initialized")
    
    def setup(self, robot_dir: str, config_path: str, hand_type: str = "Right"):
        """
        Setup the retargeting pipeline
        
        Args:
            robot_dir: Path to robot URDF directory
            config_path: Path to retargeting config file (.yml)
            hand_type: "Right" or "Left" hand
        """
        try:
            # Import retargeting components
            from dex_retargeting.retargeting_config import RetargetingConfig
            from vr_hand_detector import VRHandDetector
            
            # Setup retargeting config
            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            config = RetargetingConfig.load_from_file(config_path)
            self.retargeting = config.build()
            
            # Extract robot name from config path for robot-specific adaptations
            config_file = Path(config_path)
            robot_name = config_file.stem.split('_')[0]  # e.g., "xhand" from "xhand_right_dexpilot.yml"
            
            # Initialize VR hand detector for landmark preprocessing (disable TCP)
            self.hand_detector = VRHandDetector(
                hand_type=hand_type, 
                robot_name=robot_name, 
                use_tcp=False  # We'll feed landmarks manually
            )
            
            # Store hand type for processing
            self.hand_type = hand_type
            self.robot_name = robot_name
            
            # Get joint mapping for XHand (from retargeting to XHand physical order)
            self.retargeting_to_sapien = self._create_joint_mapping(config)
            
            # Define XHand joint order (physical robot order)
            self.xhand_joint_order = [
                "joint_0.pos", "joint_1.pos", "joint_2.pos", "joint_3.pos",
                "joint_4.pos", "joint_5.pos", "joint_6.pos", "joint_7.pos", 
                "joint_8.pos", "joint_9.pos", "joint_10.pos", "joint_11.pos"
            ]
            
            self.is_initialized = True
            
            if self.verbose:
                logger.info(f"Hand retargeting processor setup completed")
                logger.info(f"Robot: {robot_name}, Hand: {hand_type}")
                logger.info(f"Joint mapping shape: {self.retargeting_to_sapien.shape if self.retargeting_to_sapien is not None else None}")
                logger.info(f"Retargeting type: {self.retargeting.optimizer.retargeting_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup hand retargeting processor: {e}")
            return False
    
    def process_landmarks_data(self, landmarks_data, current_joints: List[float]) -> Dict[str, float]:
        """
        Process VR landmarks data into XHand joint targets
        
        Args:
            landmarks_data: VRLandmarks object with landmarks array
            current_joints: Current 12-DOF XHand joint positions
            
        Returns:
            Dict[str, float]: Hand joint actions with 'hand_' prefix
        """
        if not self.is_initialized:
            logger.warning("Hand retargeting processor not initialized")
            return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
        
        if not landmarks_data.valid or len(landmarks_data.landmarks) == 0:
            if self.verbose:
                logger.debug("Invalid landmarks data, returning current joints")
            return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
        
        try:
            # Debug: verify verbose is working
            if self.verbose:
                print(f"📊 Processing landmarks data (verbose={self.verbose})")
            
            # Convert landmarks to numpy array (21, 3)
            landmarks_array = np.array(landmarks_data.landmarks)
            
            if landmarks_array.shape != (21, 3):
                logger.warning(f"Invalid landmarks shape: {landmarks_array.shape}, expected (21, 3)")
                return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
            
            if self.verbose:
                # Log key landmark positions for debugging
                wrist = landmarks_array[0]  # Wrist
                thumb_tip = landmarks_array[4]  # Thumb tip
                index_tip = landmarks_array[8]  # Index tip
                logger.info(f"Key landmarks: wrist={wrist}, thumb_tip={thumb_tip}, index_tip={index_tip}")
            
            # Feed landmarks to hand detector for proper preprocessing
            self.hand_detector.latest_landmarks = landmarks_array
            
            # Use hand detector's processing pipeline to get proper joint positions
            _, joint_pos, _, _ = self.hand_detector.detect()
            
            if joint_pos is None:
                if self.verbose:
                    print("❌ Hand detection failed, returning current joints")
                return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
            
            # Perform retargeting using the landmarks
            retargeting_type = self.retargeting.optimizer.retargeting_type
            indices = self.retargeting.optimizer.target_link_human_indices
            
            if self.verbose:
                logger.info(f"Retargeting type: {retargeting_type}")
                logger.info(f"Target indices shape: {indices.shape if hasattr(indices, 'shape') else len(indices)}")
            
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
                if self.verbose:
                    logger.info(f"Position retargeting ref_value shape: {ref_value.shape}")
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                if self.verbose:
                    logger.info(f"Vector retargeting ref_value shape: {ref_value.shape}")
            
            # Get retargeted joint positions
            qpos = self.retargeting.retarget(ref_value)
            
            if self.verbose:
                print(f"🔧 RAW retargeting output (URDF order): {[f'{q:.3f}' for q in qpos]}")
            
            # Apply smoothing BEFORE mapping (like XHandVRTeleoperator)
            if self.last_target_joints is not None:
                qpos_before_smooth = qpos.copy()
                qpos = self._apply_smoothing_np(qpos, self.last_target_joints)
                if self.verbose:
                    logger.info(f"After smoothing: {[f'{q:.3f}' for q in qpos]}")
            
            # Map to XHand joint order if mapping is available
            if self.retargeting_to_sapien is not None:
                if self.verbose:
                    logger.info(f"Joint mapping indices: {self.retargeting_to_sapien}")
                target_joints = qpos[self.retargeting_to_sapien]
                if self.verbose:
                    logger.info(f"After joint mapping (XHand order): {[f'{q:.3f}' for q in target_joints]}")
                
                # Apply specific joint inversions as in XHandVRTeleoperator
                if len(target_joints) > 3:
                    original_joint_3 = target_joints[3]
                    target_joints[3] = -target_joints[3]  # Invert index bend for XHand
                    if self.verbose:
                        logger.info(f"Index bend inversion: {original_joint_3:.3f} -> {target_joints[3]:.3f}")
            else:
                target_joints = qpos
            
            if self.verbose:
                print(f"🎯 FINAL XHand joints: thumb=[{target_joints[0]:.3f}, {target_joints[1]:.3f}, {target_joints[2]:.3f}], index=[{target_joints[3]:.3f}, {target_joints[4]:.3f}, {target_joints[5]:.3f}]")
                print(f"🎯 ALL joints: {[f'{q:.3f}' for q in target_joints]}")
            
            # Ensure we have exactly 12 joints for XHand
            if len(target_joints) != 12:
                logger.warning(f"Expected 12 joints, got {len(target_joints)}")
                return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
            
            # Store the qpos (before mapping) for next smoothing
            self.last_target_joints = qpos.copy()
            
            # Convert to action format with hand prefix
            hand_action = {}
            for i in range(12):
                hand_action[f"hand_joint_{i}.pos"] = float(target_joints[i])
            
            if self.verbose:
                logger.debug(f"Hand retargeting targets: {[f'{j:.3f}' for j in target_joints]}")
            
            return hand_action
            
        except Exception as e:
            logger.error(f"Error in hand retargeting processing: {e}")
            # Return current joints as safe fallback
            return {f"hand_joint_{i}.pos": float(current_joints[i]) for i in range(12)}
    
    def _create_joint_mapping(self, config):
        """
        Create joint mapping from retargeting output (URDF order) to XHand physical order.
        This matches the mapping used in the original XHand teleoperator.
        """
        try:
            # Get retargeting joint names (URDF order) - these come from the built retargeting system
            retargeting_joint_names = self.retargeting.joint_names
            
            # Desired XHand order (matches XHandVRTeleoperator)
            desired_xhand_joint_names = [
                'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2',
                'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
                'right_hand_mid_joint1', 'right_hand_mid_joint2',
                'right_hand_ring_joint1', 'right_hand_ring_joint2',
                'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
            ]
            
            if self.verbose:
                logger.info(f"Retargeting URDF joints: {retargeting_joint_names}")
                logger.info(f"Desired XHand order: {desired_xhand_joint_names}")
            
            # Create mapping from retargeting output indices to desired XHand indices
            retargeting_to_xhand = []
            for desired_joint in desired_xhand_joint_names:
                if desired_joint in retargeting_joint_names:
                    retargeting_to_xhand.append(retargeting_joint_names.index(desired_joint))
                else:
                    logger.error(f"Desired joint {desired_joint} not found in retargeting joints!")
                    retargeting_to_xhand.append(0)  # fallback
            
            mapping = np.array(retargeting_to_xhand)
            
            if self.verbose:
                logger.info(f"Joint mapping created: {len(mapping)} joints mapped")
                logger.info(f"Mapping indices: {mapping}")
            
            return mapping
                
        except Exception as e:
            logger.error(f"Failed to create joint mapping: {e}")
            return np.arange(12)  # Default identity mapping
    
    def _apply_smoothing(self, current: List[float], previous: List[float]) -> List[float]:
        """Apply exponential smoothing to joint targets"""
        alpha = 1.0 - self.smoothing_factor  # Convert to smoothing weight
        return [alpha * c + (1 - alpha) * p for c, p in zip(current, previous)]
    
    def _apply_smoothing_np(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to numpy arrays (like XHandVRTeleoperator)"""
        alpha = 1.0 - self.smoothing_factor  # Convert to smoothing weight
        return alpha * current + (1 - alpha) * previous
    
    def is_ready(self) -> bool:
        """Check if processor is ready"""
        return self.is_initialized and self.retargeting is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            "initialized": self.is_initialized,
            "retargeting_ready": self.retargeting is not None,
            "hand_type": self.hand_type,
            "robot_name": self.robot_name,
            "joint_mapping_size": len(self.retargeting_to_sapien) if self.retargeting_to_sapien is not None else 0
        }