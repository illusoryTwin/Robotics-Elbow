import mujoco 
import mujoco.viewer
import time 
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('robotic_elbow.xml')
data = mujoco.MjData(model)

vels, accs = [], []
qpos = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 0.01:
        step_start = time.time()
        data.qvel = 0
        for i in range(-120, 0, 2):
            data.qpos = np.deg2rad(i)
            mujoco.mj_step(model, data)
            qpos.append(data.qpos.copy())
            vels.append(data.qvel.copy())
            accs.append(data.qacc.copy())

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

end_time = 0.01
t = np.linspace(0, end_time, len(vels))


# Plotting position, velocity, and acceleration over time using subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot qpos (position)
axs[0].plot(t, qpos, label='Position', color='blue')
axs[0].set_ylabel('Position (rad)')
axs[0].set_title('Robot Elbow Joint Position vs Time')
axs[0].legend()
axs[0].grid(True)

# Plot velocity
axs[1].plot(t, vels, label='Velocity', color='green')
axs[1].set_ylabel('Velocity (rad/s)')
axs[1].set_title('Robot Elbow Joint Velocity vs Time')
axs[1].legend()
axs[1].grid(True)

# Plot acceleration
axs[2].plot(t, accs, label='Acceleration', color='red')
axs[2].set_ylabel('Acceleration (rad/s²)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title('Robot Elbow Joint Acceleration vs Time')
axs[2].legend()
axs[2].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()


# plt.sublot(3, 1, 1)
# plt.plot(t, qpos, label='Velocity')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (rad/s)')
# plt.title('Robot Elbow Joint Velocity vs Time')
# plt.legend()
# plt.grid(True)

# plt.sublot(3, 1, 2)
# plt.plot(t, vels, label='Velocity')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (rad/s)')
# plt.title('Robot Elbow Joint Velocity vs Time')
# plt.legend()
# plt.grid(True)

# plt.sublot(3, 1, 3)
# plt.plot(t, accs, label='Acceleration')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (rad/s)')
# plt.title('Robot Elbow Joint Acceleration vs Time')
# plt.legend()
# plt.grid(True)

# plt.show()
