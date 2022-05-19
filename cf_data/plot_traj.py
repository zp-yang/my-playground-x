
# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import trajectory as traj
from numpy.polynomial import Polynomial

# %%
old_data = {}
new_data = {}
fdata_old = "old_data.json"
fdata_new = "new_data.json"
with open(fdata_old) as fs:
    old_data = json.load(fs)

with open(fdata_new) as fs:
    new_data = json.load(fs)
# %%
old_pos = np.array(old_data["Positions"], dtype="float")/1000
new_pos = np.array(new_data["Positions"], dtype="float")/1000

# %%

drone_num = 1

traj_data = {}
ftraj = "geometry_9drone.json"
with open(ftraj) as fs:
    traj_data = json.load(fs)

for drone_num in range(6):
    T_list = traj_data[str(drone_num)]['T']
    traj_coeff = np.array(traj_data[str(drone_num)]['trajectory'])
    traj_coeff = traj_coeff[:, 1:]
    traj_coeff.shape
    n = 8
    x_coeff = traj_coeff[:, 0:n]
    y_coeff = traj_coeff[:, n:2*n]
    z_coeff = traj_coeff[:, 2*n:3*n]

    P_list_x = [Polynomial(pi) for pi in x_coeff]
    traj_x = traj.Trajectory1D(T=T_list, P=P_list_x)
    x_data = traj_x.eval()

    P_list_y = [Polynomial(pi) for pi in y_coeff]
    traj_y = traj.Trajectory1D(T=T_list, P=P_list_y)
    y_data = traj_y.eval()

    P_list_z = [Polynomial(pi) for pi in z_coeff]
    traj_z = traj.Trajectory1D(T=T_list, P=P_list_z)
    z_data = traj_z.eval()

    fig1 = plt.figure(figsize=(20,10))
    ax1 = fig1.add_subplot(1,2,1,projection="3d")
    ax1.plot(old_pos[drone_num,0,:], old_pos[drone_num,1,:], old_pos[drone_num,2,:], label="old")
    ax1.plot(x_data[1], y_data[1], z_data[1], label="planned")
    ax1.set_xlabel("x[m]")
    ax1.set_ylabel("y[m]")
    ax1.set_zlabel("z[m]")
    ax1.legend()

    ax2 = fig1.add_subplot(1,2,2,projection="3d")
    ax2.plot(new_pos[drone_num,0,:], new_pos[drone_num,1,:], new_pos[drone_num,2,:], label="new")
    ax2.plot(x_data[1], y_data[1], z_data[1], label="planned")
    ax1.set_xlabel("x[m]")
    ax1.set_ylabel("y[m]")
    ax1.set_zlabel("z[m]")
    ax2.legend()
    plt.suptitle("Drone {} trajectory".format(drone_num))
    plt.savefig("drone_{}_traj_comparision.png".format(drone_num))
# %%
