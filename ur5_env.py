import gym
from gym import error, spaces, utils
from gym.utils import seeding
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import String, Bool, Float64MultiArray
from sensor_msgs.msg import JointState, Image, CameraInfo
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, ListControllers
from ur_control_wrapper.srv import GetPose
from ur_control_wrapper.srv import SetPose
from ur_control_wrapper.srv import SetJoints
from ur_control_wrapper.srv import GetJoints
from ur_control_wrapper.srv import InverseKinematics, InverseKinematicsRequest
import rospy
import numpy as np
import time

from ur_control_wrapper.srv import GetWrench
from ur_control_wrapper.srv import GetCamImage
from ur_control_wrapper.srv import GetAlignedDepthImg

class Ur5Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rospy.init_node('ur_admittance_control_node', anonymous=True)

        self.action_space = None
        self.observation_space = None
        self.reward_range = (-2, 2)

        self.arm_pose = None
        self.arm_orientation = None
        self.default_joints = [0.0, -2.0 * np.pi / 3, -np.pi / 3.0, np.pi / 2.0, -np.pi / 2.0, 0.0]

        self.interval = 1 / 30
        self.force_vel_scale = 1./10
        self.pose_vel_scale = 0.08

        rospy.Service('ur_control_wrapper/switch_admittance_control', SetBool, self.switch_admittance_control)
        rospy.Service('ur_control_wrapper/check_admittance_control', Trigger, self.check_admittance_control)

        self.joint_velocity_pub = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10)


    def step(self, action):
        '''
        A step() function that takes linear end-effector velocity as the agent's current action
        '''

        # switch to joint velocity control
        controller_state = self.check_controller_state(silent=True)
        if controller_state["joint_group_vel_controller"] != "running":
            self.switch_controller("joint_group")

        # convert linear joint velocity to joint angular velocity
        vel_magnitude = np.linalg.norm(action)
        pose_offset_vel = self.pose_vel_scale * action / vel_magnitude
        desired_next_joint_state = self.compute_inverse_kinematics(pose_offset_vel)
        if len(desired_next_joint_state.position) == 0:
            self.set_joint_velocity(np.zeros(6))
            print("The desired pose cannot be reached.")
            self.switch_controller("scaled_pos")
            return
        current_joint_state = self.get_angle()
        jv_input = np.array(desired_next_joint_state.position) - np.array(current_joint_state.position)
        jv_input = jv_input * vel_magnitude*self.force_vel_scale
        self.set_joint_velocity(jv_input)

        # get current state information
        curr_pose = self.get_pose()
        curr_wrench = self.get_wrench().WrenchStamped.wrench
        curr_image_raw = self.get_image_raw().Image
        curr_depth_image = self.get_depth_image_raw().Image
        ob = [curr_pose, curr_wrench, curr_image_raw, curr_depth_image]
        
        # currently only returns observation for testing purposes
        return ob, 1, 1, 1


    def reset(self):
        
        # switch to scaled position control
        controller_state = self.check_controller_state(silent=True)
        if controller_state["scaled_pos_joint_traj_controller"] != "running":
            self.switch_controller("scaled_pos")

        # reset robot to home position
        self.set_default_angles()

        # reset attributes
        self.arm_pose = None
        self.arm_orientation = None


    def render(self, mode='human', close=False):
        ...
    

    def get_wrench(self):
        # get force and torch information on each joint
        rospy.wait_for_service("ur_control_wrapper/get_wrench")
        get_current_wrench = rospy.ServiceProxy("ur_control_wrapper/get_wrench", GetWrench)
        current_wrench = None
        try:
            current_wrench = get_current_wrench()
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        
        return current_wrench


    def get_image_raw(self):
        '''
        This function gets current colored image from the wrist camera
        '''
        rospy.wait_for_service("ur_control_wrapper/get_cam_image")
        get_current_img_raw = rospy.ServiceProxy("ur_control_wrapper/get_cam_image", GetCamImage)
        current_img_raw = None
        try:
            current_img_raw = get_current_img_raw()
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        
        return current_img_raw


    def get_depth_image_raw(self):
        '''
        This function gets current depth image from the wrist camera
        '''
        rospy.wait_for_service("ur_control_wrapper/get_aligned_depth_img")
        get_current_depth_img = rospy.ServiceProxy("ur_control_wrapper/get_aligned_depth_img", GetAlignedDepthImg)
        current_depth_img = None
        try:
            current_depth_img = get_current_depth_img()
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        
        return current_depth_img


    def get_pose(self):
        rospy.wait_for_service("ur_control_wrapper/get_pose")
        get_current_pose = rospy.ServiceProxy("ur_control_wrapper/get_pose", GetPose)
        current_pose = None
        try:
            current_pose = get_current_pose().pose
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        
        return current_pose


    def move_arm(self, action):
        rospy.wait_for_service("ur_control_wrapper/set_pose")
        set_current_pose = rospy.ServiceProxy("ur_control_wrapper/set_pose", SetPose)

        try:
          current_pose = self.get_pose()
          current_pose.position.x += (action[0] / 100)
          current_pose.position.y += (action[1] / 100)
          current_pose.position.z += (action[2] / 100)

          response = set_current_pose(current_pose)
          pose = response.response_pose
          is_reached = response.is_reached

        except rospy.ServiceException as exc:
          print("Service did not process request: " + str(exc))


    def set_default_angles(self):
          rospy.wait_for_service("ur_control_wrapper/set_joints")
          set_joints = rospy.ServiceProxy("ur_control_wrapper/set_joints", SetJoints)
          joints = JointState()
          joints.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
          joints.position = self.default_joints
          try:
              response = set_joints(joints)
          except rospy.ServiceException as exc:
              print("Service did not process request: " + str(exc))


    def get_angle(self):
        rospy.wait_for_service("ur_control_wrapper/get_joints")
        get_current_joints = rospy.ServiceProxy("ur_control_wrapper/get_joints", GetJoints)
        current_joints = None
        try:
            current_joints = get_current_joints().joints
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return current_joints

    
    def compute_inverse_kinematics(self, desired_pose_offset, silent=True):
        # desired_pose_offset format --> [offset_x, offset_y, offset_z]
        joint_state = None
        current_pose = self.get_pose()
        desired_pose = current_pose
        desired_pose.position.x += desired_pose_offset[0]
        desired_pose.position.y += desired_pose_offset[1]
        desired_pose.position.z += desired_pose_offset[2]
        rospy.wait_for_service("ur_control_wrapper/inverse_kinematics")
        compute_ik = rospy.ServiceProxy("ur_control_wrapper/inverse_kinematics", InverseKinematics)
        try:
            req = InverseKinematicsRequest()
            req.pose = desired_pose
            response = compute_ik(req)
            # response = compute_ik(desired_pose)
            solution_found, joint_state = response.solution_found, response.joint_state
            if solution_found:
                if not silent:
                    print("joint state: ")
                    print(joint_state)
            else:
                print("Solution is not found.")
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return joint_state


    def set_joint_velocity(self, jv_input):
        # can only be called under joint velocity control
        controller_state = self.check_controller_state(silent=True)
        if controller_state["joint_group_vel_controller"] != "running":
            print("joint_group_vel_controller is not started. Failed to set joint velocity.")
            return

        if isinstance(jv_input, str):
            jv_float = [float(entry) for entry in jv_input.split(" ")] # [0., 0., 0., 0., 0., 0.] # 6 entries
        else:
            jv_float = jv_input
        data = Float64MultiArray()
        data.data = jv_float
        self.joint_velocity_pub.publish(data)
        return


    def check_controller_state(self, silent=False):
        rospy.wait_for_service("/controller_manager/list_controllers")
        list_controllers = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        response = list_controllers()
        controller_state = {}
        for single_controller in response.controller:
            if single_controller.name == "joint_group_vel_controller" \
                or single_controller.name == "scaled_pos_joint_traj_controller":
                controller_state[single_controller.name] = single_controller.state
        if not silent:
            if controller_state["scaled_pos_joint_traj_controller"] == "running":
                print("Current controller is scaled_pos_joint_traj_controller.")
            elif controller_state["joint_group_vel_controller"] == "running":
                print("Current controller is joint_group_vel_controller.")
            else:
                print("Neither scaled_pos_joint_traj_controller nor joint_group_vel_controller is running.")
        return controller_state

    
    def switch_controller(self, desired_controller):
        # desired_controller is either 'joint_group' or 'scaled_pos'
        controller_switched = False
        if desired_controller != 'joint_group' and desired_controller != 'scaled_pos':
            print("Wrong controller input.")
            return controller_switched

        rospy.wait_for_service("controller_manager/switch_controller")
        switch_controller = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)

        try:
            req = SwitchControllerRequest()
            if desired_controller == 'joint_group':
                req.start_controllers = ['joint_group_vel_controller']
                req.stop_controllers = ['scaled_pos_joint_traj_controller']
            else:
                req.start_controllers = ['scaled_pos_joint_traj_controller']
                req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = 2
            req.start_asap = False
            req.timeout = 0.0
            response = switch_controller(req)
            if response.ok:
                controller_switched = True
                if desired_controller == 'joint_group':
                    print("Controller successfully switched to joint_group_vel_controller.")
                else:
                    print("Controller successfully switched to scaled_pos_joint_traj_controller.")
            else:
                print("Controller switch failed.")
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return controller_switched


    def switch_admittance_control(self, data):
        if data.data:
            message = "Admittance control mode is turned on."
            controller_state = self.check_controller_state()
            if controller_state["joint_group_vel_controller"] != "running":
                self.switch_controller("joint_group")
        else:
            message = "Admittance control mode is turned off."
            controller_state = self.check_controller_state()
            if controller_state["scaled_pos_joint_traj_controller"] != "running":
                self.switch_controller("scaled_pos")
        self.admittance_control_mode = data.data
        success = True
        return success, message


    def check_admittance_control(self, data):
        if self.admittance_control_mode:
            message = "Admittance control mode is on."
        else:
            message = "Admittance control mode is off."
        success = True
        return success, message
