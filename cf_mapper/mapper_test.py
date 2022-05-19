import time
from typing_extensions import Self
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from concurrent.futures import ThreadPoolExecutor

URI = 'radio://0/80/250K/E7E7E7E7E7'

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
# logging.basicConfig(level=logging.ERROR)

class Plotter():
    def __init__(self, data_src) -> None:
        self.data_src = data_src
        self.xlim = data_src.xlim
        self.ylim = data_src.ylim
        self.res = data_src.res

        fig, ax = plt.subplots(figsize=(10,10))
        self.fig = fig
        self.ax = ax
        self.w = data_src.w
        self.h = data_src.h
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        # self.title = ax.text(0.5,0.85, "--", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
        #         transform=ax.transAxes, ha="center")

        self.ax_img = self.ax.imshow(data_src.grid.T, cmap=plt.get_cmap("binary"), extent=[
            self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]], animated=True, vmin=0, vmax=1)

        self.animator = None
    
    def update(self, i):
        grid = self.data_src.get_grid()
        if grid is not None:
            self.ax_img.set_data(grid.T)
            # self.title.set_text("frame " + str(frame))
            #     
    def run(self):
        if self.animator is None:
            self.animator = FuncAnimation(
                self.fig,
                self.update,
                frames=20,
                interval=50,
            )
            plt.show()

# Use callback fucntions to get pose data from drone, asynchronous, not as limited as SyncLogger
class Mapper():
    def __init__(self, URI=URI, xlim=(-2.5,2.5), ylim=(-2.5,2.5), res=0.05) -> None:
        self.URI = URI
        self.xlim = xlim
        self.ylim = ylim
        self.res = res
        self.w = int(self.xlim[1] - self.xlim[0])
        self.h = int(self.ylim[1] - self.ylim[0])
        self.grid = np.ones((int((self.xlim[-1]-self.xlim[0])/self.res), int((self.ylim[-1]-self.ylim[0])/self.res)))

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        # Connect callbacks from the Crazyflie API
        self.cf.connected.add_callback(self.connected_cb)
        self.cf.disconnected.add_callback(self.disconnect_cb)
        self.cf.open_link(URI)

        self.sensor_ang = [0, 180, 90, -90] # f b l r (degs)
        self.dist_thresh = 2.5
        self.pos = None
        self.att = None
        self.range_meas = None
        self.data = {
            "pos": None, # x y z
            "att": None, # r p y
            "range": None, # f b l r u d
        }
        self.drone_ready = False

    def connected_cb(self, URI):
        print("Connected to ", URI)

        pose_log = LogConfig("Postion Log", period_in_ms=100)
        pose_log.add_variable("stateEstimate.x")
        pose_log.add_variable("stateEstimate.y")
        pose_log.add_variable("stateEstimate.z")
        pose_log.add_variable("stateEstimate.roll")
        pose_log.add_variable("stateEstimate.pitch")
        pose_log.add_variable("stateEstimate.yaw")

        try:
            self.cf.log.add_config(pose_log)
            pose_log.data_received_cb.add_callback(self.pose_cb)
            pose_log.start()
        except KeyError as e:
            print('Could not start log configuration,'
                    '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print("Could not add postion log config")

        range_log = LogConfig("Measurement Log", period_in_ms=100)
        range_log.add_variable('range.front')
        range_log.add_variable('range.back')
        range_log.add_variable('range.up')
        range_log.add_variable('range.left')
        range_log.add_variable('range.right')
        range_log.add_variable('range.zrange')

        try:
            self.cf.log.add_config(range_log)
            range_log.data_received_cb.add_callback(self.range_cb)
            range_log.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Measurement log config, bad configuration.')

    def disconnect_cb(self, URI):
        print("Disconnected")

    def pose_cb(self, timestamp, data, logconf):
        self.pos = np.array([
            data["stateEstimate.x"],
            data["stateEstimate.y"],
            data["stateEstimate.z"],
        ])
        self.att = np.array([
            data["stateEstimate.roll"],
            data["stateEstimate.pitch"],
            data["stateEstimate.yaw"],
        ])
        self.data["pos"] = self.pos
        self.data["att"] = self.att
    
    def range_cb(self, timestamp, data, logconf):
        self.range_meas = np.array([
            data["range.front"],
            data["range.back"],
            data["range.left"],
            data["range.right"],
            data["range.up"],
            data["range.zrange"],
        ]) / 1000.
        self.data["range"] = self.range_meas

        if self.drone_ready:
            distance = np.array(self.range_meas[0:4])
            # print("f: {:.4f} b: {:.4f} l: {:.4f} r: {:.4f} yaw: {:.4f}".format(distance[0],distance[1],distance[2],distance[3], self.att[2]))
            # print("x: {:.4f} y: {:.4f} z: {:.4f}".format(self.pos[0], self.pos[1], self.pos[2]))

            for angle, dist in zip(self.sensor_ang, distance):
                if dist > self.dist_thresh:
                    # continue
                    pass
                if self.pos is not None and self.att is not None:
                    self.measure(
                        self.pos[0], 
                        self.pos[1], 
                        np.deg2rad(self.att[2]+angle), 
                        dist
                    )

    def get_ij(self, x, y):
        i = int(np.floor((x + self.w)/(2*self.res)))
        j = int(np.floor((-y + self.h)/(2*self.res)))
        if i < 0 or i >= self.grid.shape[0]:
            raise ValueError('x out of bounds')
        if j < 0 or j >= self.grid.shape[1]:
            raise ValueError('y out of bounds')
        return i, j
    
    def get_probability(self, x, y):
        i, j = self.get_ij(x, y)
        return self.grid[i, j]

    def prob_update(self, i, j, prob):
        p0 = self.grid[i,j]
        alpha = 0.5
        p = alpha*p0 + (1 - alpha)*prob
        if p > 1:
            p = 1
        elif p < 0:
            p = 0
        self.grid[i, j] = p
    
    def measure(self, x, y, theta, dist):
        p = np.array([x, y])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        intersections = []
        t_list = np.arange(0, dist, self.res/2)
        for i_t, t in enumerate(t_list):
            m = p + R@np.array([t, 0])
            if i_t == len(t_list) - 1:
                # positive information where we see things
                prob = 1
            else:
                # negative information, where we don't see anything
                prob = 0
            try:
                i, j = self.get_ij(m[0], m[1])
                if len(intersections) == 0 or intersections[-1] != (i, j):
                    intersections += [(i, j, prob)]
            except ValueError as e:
                continue
        for i, m in enumerate(intersections):
            self.prob_update(i=m[0], j=m[1], prob=m[2])
    
    def get_data(self):
        return self.data

    def get_cf(self):
        return self.cf

    def get_grid(self):
        return self.grid

    def set_drone_ready(self, ready=False):
        self.drone_ready = ready

class Mover():
    def __init__(self, mapper=None) -> None:
        self.mapper = mapper
        self.cf : Crazyflie = None
        if self.mapper is not None:
            self.cf = mapper.get_cf()

        self.var_y_history = [1000] * 10
        self.var_x_history = [1000] * 10
        self.var_z_history = [1000] * 10
        self.is_converged = False
        self.converge_thresh = 0.001

        self.drone_ready = False
        self.keep_fly = False

    def set_initial_position(self, x, y, z, yaw_deg):
        self.cf.param.set_value('kalman.initialX', x)
        self.cf.param.set_value('kalman.initialY', y)
        self.cf.param.set_value('kalman.initialZ', z)
        self.cf.param.set_value('kalman.initialYaw', np.deg2rad(yaw_deg))
    
    def reset_estimator(self):
        self.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self.cf.param.set_value('kalman.resetEstimation', '0')

        print('Waiting for estimator to find position...')

        pos_var_log = LogConfig(name='Kalman Variance', period_in_ms=500)
        pos_var_log.add_variable('kalman.varPX', 'float')
        pos_var_log.add_variable('kalman.varPY', 'float')
        pos_var_log.add_variable('kalman.varPZ', 'float')

        try:
            self.cf.log.add_config(pos_var_log)
            self.start_t = time.time()

            pos_var_log.data_received_cb.add_callback(self.pos_var_cb)
            pos_var_log.start()
        except KeyError as e:
            print('Could not start log configuration,'
                    '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print("Could not add postion log config")
    
    def pos_var_cb(self, timestamp, data, logconf):
        now = time.time()
        elapsed = now - self.start_t

        self.var_x_history.append(data['kalman.varPX'])
        self.var_x_history.pop(0)
        self.var_y_history.append(data['kalman.varPY'])
        self.var_y_history.pop(0)
        self.var_z_history.append(data['kalman.varPZ'])
        self.var_z_history.pop(0)

        min_x = min(self.var_x_history)
        max_x = max(self.var_x_history)
        min_y = min(self.var_y_history)
        max_y = max(self.var_y_history)
        min_z = min(self.var_z_history)
        max_z = max(self.var_z_history)

        print("elapsed: {:.2f} -- {:.4f} {:.4f} {:.4f}".
              format(elapsed, max_x - min_x, max_y - min_y, max_z - min_z))

        if (max_x - min_x) < self.converge_thresh and (
                max_y - min_y) < self.converge_thresh and (
                max_z - min_z) < self.converge_thresh:
            self.is_converged = True
            logconf.stop()
            logconf.delete()
            print("Estimator has converged...")
        
        if elapsed > 10:
            print("Elapsed 10 seconds...")
            logconf.stop()
            logconf.delete()
            self.cf.close_link()
            print("Estimator cannot converge...")
            raise Exception("Estimator cannot converge in determined time!")
        
    def takeoff(self, height=0.5, duration=5):
        if self.is_converged:
            print("estimator converged, preparing to take off...")
            # self.cf.high_level_commander.takeoff(height, duration)
            self.cf.commander.send_hover_setpoint(vx=0, vy=0, yawrate=0, zdistance=0.5)
            self.keep_fly = True
            # time.sleep(duration)
            self.mapper.set_drone_ready(ready=True)
            

    def sequence(self, sequence):
        pass

    def hover(self):
        vx = 0
        vy = 0
        yawrate = 0
        zdist = 0.3
        while self.keep_fly:
            self.cf.commander.send_hover_setpoint(vx=vx, vy=vy, yawrate=yawrate, zdistance=zdist)
            time.sleep(0.1)
        print("stop flying...")

    def set_keep_fly(self, condition=False):
        self.keep_fly = condition
    


if __name__ == '__main__':
    # Use callback function to get pose data
    try:
        mapper = Mapper(URI, res=0.05)
        plotter = Plotter(mapper)

        mover = Mover(mapper=mapper)
        mover.set_initial_position(0, 0, 0, 0)
        mover.reset_estimator()

        time.sleep(8)
        # mover.takeoff()
        mapper.set_drone_ready(True)

        with ThreadPoolExecutor(max_workers=5) as tpe:
            # tpe.submit(mover.hover)
            plotter.run()

    except Exception as e:
        mover.set_keep_fly(False)
        print(e)
