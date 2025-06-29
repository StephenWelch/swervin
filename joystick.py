import time
import mujoco
import mujoco.viewer
import pygame
import numpy as np
import math
import os

# --- Constants and Configuration ---
MODEL_PATH = "assets/models/reduced/mjcf/scene.xml"
# Wheel locations (x, y) relative to the chassis center
# FR, FL, BR, BL
WHEEL_POSITIONS = np.array([
    [0.2, 0.15],   # FR
    [0.2, -0.15],  # FL
    [-0.2, 0.15],  # BR
    [-0.2, -0.15]   # BL
])

MAX_LINEAR_VELOCITY = 30.0  # m/s
MAX_ANGULAR_VELOCITY = 300.0  # rad/s
MAX_ROT_CENTER_OFFSET = 0.5 # meters

# Actuator names from the XML file
DRIVE_MOTORS = ["FR_drive_motor", "FL_drive_motor", "BR_drive_motor", "BL_drive_motor"]
STEER_MOTORS = ["FR_steer_motor", "FL_steer_motor", "BR_steer_motor", "BL_steer_motor"]

def swerve_inverse_kinematics(linear_vel, angular_vel, rot_center):
    """
    Calculates the required wheel speeds and steering angles.
    Args:
        linear_vel (np.array): Desired linear velocity [vx, vy] in m/s.
        angular_vel (float): Desired angular velocity in rad/s.
        rot_center (np.array): Point to rotate around [px, py] in meters.
    Returns:
        tuple: (drive_speeds, steer_angles)
    """
    num_wheels = WHEEL_POSITIONS.shape[0]
    drive_speeds = np.zeros(num_wheels)
    steer_angles = np.zeros(num_wheels)

    for i in range(num_wheels):
        r_i = WHEEL_POSITIONS[i]
        rot_vec = r_i - rot_center
        
        # Velocity of the wheel due to rotation
        # v_rot = omega (k-hat) x rot_vec
        v_rot = np.array([-angular_vel * rot_vec[1], angular_vel * rot_vec[0]])
        
        # Total wheel velocity
        v_wheel = linear_vel + v_rot

        drive_speeds[i] = np.linalg.norm(v_wheel)
        steer_angles[i] = math.atan2(v_wheel[1], v_wheel[0])

    return drive_speeds, steer_angles


def main():
    """Main function to run the joystick controller."""
    # --- Pygame Initialization ---
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick found. Please connect one.")
        return

    joystick = pygame.joystick.Joystick(1)
    joystick.init()

    print(f"Initialized joystick: {joystick.get_name()}")

    # --- MuJoCo Initialization ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    data = mujoco.MjData(model)

    # --- Main Control Loop ---
    running = True
    
    # Store previous steer angles for optimization
    last_steer_angles = np.zeros(len(STEER_MOTORS))
    def key_cb(keycode: int):
        if chr(keycode) == 'R':
            gantry_eq_id = model.equality("gantry").id
            if data.eq_active[gantry_eq_id] == 1:
                data.eq_active[gantry_eq_id] = 0
            else:
                data.eq_active[gantry_eq_id] = 1
                # mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
                print("Releasing gantry")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        # Get the ID of the mocap body for the rotation center visualization.
        try:
            mocap_id = model.body("rotation_center_vis").mocapid[0]
        except (KeyError, IndexError):
            print("Warning: Mocap body 'rotation_center_vis' not found in the model. Visualization will be disabled.")
            mocap_id = -1

        while running and viewer.is_running():
            start_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.JOYDEVICEREMOVED:
                    print("Joystick disconnected.")
                    running = False

            # --- Get Joystick Input ---
            # Xbox controller mapping can vary. This is a common setup.
            # Left stick: linear velocity
            vx = -joystick.get_axis(1)  # Inverted Y axis
            vy = joystick.get_axis(0)

            # Right stick: center of rotation offset
            px = joystick.get_axis(2) 
            py = -joystick.get_axis(3) # Inverted Y axis

            # Triggers: angular velocity
            # LT (axis 4) and RT (axis 5) are often 0 to 1
            # Some drivers map them to axis 2 and 5 from -1 to 1 when idle
            # Let's assume they are mapped to axis 4 and 5 from -1 to 1.
            # On Windows they are axis 4 and 5, but might be different on other OS.
            try:
                rt_trigger = (joystick.get_axis(5) + 1) / 2 # Right trigger (clockwise)
                lt_trigger = (joystick.get_axis(4) + 1) / 2 # Left trigger (counter-clockwise)
                omega = (rt_trigger - lt_trigger)
            except pygame.error:
                # Fallback for controllers with fewer axes (e.g. axis 2 for triggers)
                trigger_axis = joystick.get_axis(2)
                omega = -trigger_axis


            # Apply deadzones and scaling
            deadzone = 0.015
            vx = 0 if abs(vx) < deadzone else vx
            vy = 0 if abs(vy) < deadzone else vy
            px = 0 if abs(px) < deadzone else px
            py = 0 if abs(py) < deadzone else py
            omega = 0 if abs(omega) < deadzone else omega

            linear_velocity_world_2d = np.array([vx, vy]) * MAX_LINEAR_VELOCITY
            angular_velocity = omega * MAX_ANGULAR_VELOCITY
            rotation_center = np.array([px, py]) * MAX_ROT_CENTER_OFFSET

            # --- Frame Transformation for World-Relative Control ---
            # Get chassis orientation to transform world commands to the body frame.
            chassis_quat = data.body("chassis").xquat

            # Joystick linear velocity is in the world frame. It needs to be
            # rotated into the robot's body frame for the inverse kinematics.
            linear_velocity_world_3d = np.array([linear_velocity_world_2d[0], linear_velocity_world_2d[1], 0])
            inv_chassis_quat = np.empty(4)
            mujoco.mju_negQuat(inv_chassis_quat, chassis_quat)
            linear_velocity_body_3d = np.empty(3)
            mujoco.mju_rotVecQuat(linear_velocity_body_3d, linear_velocity_world_3d, inv_chassis_quat)
            linear_velocity_body = linear_velocity_body_3d[:2]

            # --- Update visualization for rotation center ---
            if mocap_id != -1:
                # The rotation_center is in the chassis frame.
                # We need to transform it to the world frame for the mocap body.
                chassis_pos = data.body("chassis").xpos

                # Local position of rotation center
                rot_center_local = np.array([rotation_center[0], rotation_center[1], 0])
                
                # Rotated offset in world frame
                offset_world = np.empty(3)
                mujoco.mju_rotVecQuat(offset_world, rot_center_local, chassis_quat)

                # World position of rotation center
                rot_center_world = chassis_pos + offset_world
                
                # Set the position of the mocap body
                data.mocap_pos[mocap_id] = rot_center_world

            # --- Inverse Kinematics ---
            drive_speeds, steer_angles = swerve_inverse_kinematics(
                linear_velocity_body, angular_velocity, rotation_center
            )

            # --- Control Optimization ---
            # If the wheel has to turn more than 90 degrees, it's faster to
            # reverse the drive motor and turn to the supplementary angle.
            for i in range(len(STEER_MOTORS)):
                angle_diff = steer_angles[i] - last_steer_angles[i]
                
                # Wrap to [-pi, pi]
                while angle_diff > math.pi: angle_diff -= 2 * math.pi
                while angle_diff < -math.pi: angle_diff += 2 * math.pi

                if abs(angle_diff) > math.pi / 2:
                    drive_speeds[i] *= -1
                    steer_angles[i] += math.pi
            
            last_steer_angles = steer_angles.copy()
            # Wrap final angles to [-pi, pi] for the controller
            steer_angles = np.arctan2(np.sin(steer_angles), np.cos(steer_angles))


            # --- Set Actuator Controls ---
            for i, name in enumerate(DRIVE_MOTORS):
                data.ctrl[model.actuator(name).id] = drive_speeds[i]

            for i, name in enumerate(STEER_MOTORS):
                data.ctrl[model.actuator(name).id] = steer_angles[i]
            
            # --- Simulation Step ---
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_sleep = model.opt.timestep - (time.time() - start_time)
            if time_until_next_sleep > 0:
                time.sleep(time_until_next_sleep)

    pygame.quit()

if __name__ == "__main__":
    main() 