import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class boundary:

    def __init__(self, D, n1, n2, S1, S2):
        self.D = D
        self.n1 = n1
        self.n2 = n2
        self.centerPoint = np.array([S1,S2])


    #Function to return start and end points for boundary line of specified length
    def getLine(self, length):
        # Get vector b pependicular to line defined by n1,n2
        # Set b1 = 1 and then solve n.b = 0
        x_low = self.centerPoint[0] - 0.1 * self.centerPoint[0]
        x_high = self.centerPoint[0] + 0.1 * self.centerPoint[0]
        y_low = (-self.D - self.n1 * x_low)/self.n2
        y_high = (-self.D - self.n1 * x_high)/self.n2
        vector1 = np.array([x_low,y_low])
        vector2 = np.array([x_high,y_high])
        vector = vector2 - vector1
        # Make b a unit vector
        unit = vector/np.linalg.norm(vector)
        # Now add and subtract unit vector multiplied by half the length to the boundary point S1 and S2
        # This gives the start and end points of a line length L centered upon point S1 and S2
        start = vector1 + (length/2) * unit
        end = vector1 - (length/2) * unit
        x = np.array([start[0],end[0]])
        y = np.array([start[1],end[1]])
        return(x,y)

class boundary3D:

    def __init__(self, D, n1, n2, n3, S1, S2, S3):
        self.D = D
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.centerPoint = np.array([S1, S2, S3])


    def getPlane(self, size):
        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set
        d = -self.centerPoint.dot([self.n1,self.n2, self.n3])
        # create x,y
        x, y = np.meshgrid(np.arange((self.centerPoint[0])-size,(self.centerPoint[0])+size, 2), np.arange((self.centerPoint[1])-size,(self.centerPoint[1]+size), 2))
        # calculate corresponding z
        z = (-self.n1 * x - self.n2 * y - d) * 1. / self.n3
        return x, y, z


class bxd_plotter_2d:

    def __init__(self, path_data, path_colour="tomato", point_colour="teal", bound_colour="orange",
                 bound_size = 5, double_bounds = False, zoom=False, all_bounds=True, ):
        plt.ion()
        self.path_colour = path_colour
        self.follow_current_box = zoom
        self.all_bounds = all_bounds
        self.point_colour = point_colour
        self.bound_colour = bound_colour
        self.bound_size = bound_size
        self.double_bounds = double_bounds
        self.path_data = path_data
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scatter = self.ax.scatter([], [], s=3, color=self.point_colour, alpha=0.1)
        self.bound_lines = []
        b1 = self.ax.plot([], [], color=self.bound_colour)
        b2 = self.ax.plot([], [], color=self.bound_colour)
        self.bound_lines.append(b1)
        self.bound_lines.append(b2)
        self.path_lines = []
        try:
            colour = plt.cm.copper(np.linspace(0, 1, len(self.path_data)))
            for i in range(0, len(self.path_data)-1):
                path = self.ax.plot(np.array([self.path_data[i][0], self.path_data[i + 1][0]]), np.array([self.path_data[i][1], self.path_data[i + 1][1]]),
                        color=colour[i], alpha=0.5)
                self.path_lines.append(path)
        except:
            print("couldnt print path ")
        self.fig.show()

    def plot_bxd_from_file(self, point_file, bound_file):
        points, bounds = self.read_file(point_file, bound_file)
        self.plot_update(points, bounds)

    def read_file(self, point, bound):
        point_file = open(point,'r')
        bound_file = open(bound, 'r')
        b = bound_file.readlines()
        p = point_file.readlines()
        points = []
        for p_line in p:
            p_line = p_line.replace('[', '')
            p_line = p_line.replace(']', '')
            words = p_line.split()
            points.append([float(words[0]),float(words[1])])
        bounds = []
        for b_line in b:
            b_line = b_line.replace('[', '')
            b_line = b_line.replace(']', '')
            b_line = b_line.replace('(', '')
            b_line = b_line.replace(')', '')
            b_line = b_line.replace(',', '')
            words = b_line.split('array')
            d = float(words[0])
            n = words[1].replace('array','')
            n = n.split()
            s = words[2].replace('array', '')
            s = s.replace('\n', '')
            s = s.split()
            bo = boundary(d, float(n[0]), float(n[1]), float(s[0]), float(s[1]))
            bounds.append(bo)
        return np.array(points), bounds

    def plot_bxd_from_array(self, points, bounds):
        boundList = []
        for b in bounds:
            bo = boundary(b[0],b[1][0],b[1][1],b[2][0],b[2][1])
            boundList.append(bo)
        self.plot_update(np.array(points), boundList)

    def plot_update(self, points, bounds):
        self.scatter.set_offsets(points[:,:2])
        if self.follow_current_box:
            self.ax.set_xlim([(points[-1,0])-5, (points[-1,0])+5])
            self.ax.set_ylim([(points[-1,1])-5, (points[-1,1])+5])
        else:
            self.ax.set_xlim([min(points[:,0])-2, max(points[:,0])+1])
            self.ax.set_ylim([min(points[:,1])-2, max(points[:,1])+1])
        self.bound_lines = []
        for b in bounds:
            line_start, line_end = b.getLine(self.bound_size)
            bl = self.ax.plot(line_start, line_end, color= self.bound_colour)
            self.bound_lines.append(bl)
            if self.double_bounds:
                line_start2, line_end2 = b.getLine(self.bound_size * 2)
                bl2 = self.ax.plot(line_start2, line_end2, color='grey')
                self.bound_lines.append(bl2)
        self.fig.canvas.draw()
        plt.pause(3)


class bxd_plotter_3d:

    def __init__(self, path_data, path_colour="tomato", point_colour="teal", bound_colour="orange",
                 bound_size = 5, double_bounds = False):
        plt.ion()
        self.path_colour = path_colour
        self.point_colour = point_colour
        self.bound_colour = bound_colour
        self.bound_size = bound_size
        self.double_bounds = double_bounds
        self.path_data = path_data
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = self.ax.scatter([], [], [], s=3, color=self.point_colour, alpha=0.25)
        self.bound_lines = []
        for i in range(0, 2):
            path = self.ax.plot([], [], zs=[], color=self.bound_colour, alpha=0.5)
            self.bound_lines.append(path)
        self.path_lines = []
        colour = plt.cm.bone(np.linspace(0, 1, len(self.path_data)))
        for i in range(0, len(self.path_data)-1):
            path = self.ax.plot(np.array([self.path_data[i][0], self.path_data[i + 1][0]]), np.array([self.path_data[i][1],self.path_data[i + 1][1]]),zs=np.array([self.path_data[i][2], self.path_data[i + 1][2]]), color=self.path_colour, alpha=0.5)
            self.path_lines.append(path)
        self.fig.show()
        self.fig.canvas.draw()

    def plot_bxd_from_array(self, points, bounds):
        boundList = []
        for b in bounds:
            bo = boundary3D(b[0],b[1][0],b[1][1],b[1][2],b[2][0],b[2][1],b[2][2])
            boundList.append(bo)
        self.plot_update(np.array(points), boundList)

    def plot_update(self, points, bounds):
        self.scatter._offsets3d = (points[:,0], points[:,1], points[:,2])
        self.ax.set_xlim([min(points[:,0])-1, max(points[:,0])+1])
        self.ax.set_ylim([min(points[:,1])-1, max(points[:,1])+1])
        self.ax.set_zlim([min(points[:,2])-1, max(points[:,2])+1])
        self.bound_lines = []
        for i in range(0,len(bounds)):
            xx,yy,zz = bounds[i].getPlane(self.bound_size)
            self.ax.plot_surface(xx, yy, zz, color=self.bound_colour, alpha=0.5)
        self.fig.canvas.draw()
        plt.pause(7)

    def plot_bxd_from_file(self, point_file, bound_file):
        points, bounds = self.read_file(point_file, bound_file)
        self.plot_update(points, bounds)

    def read_file(self, point, bound):
        point_file = open(point,'r')
        bound_file = open(bound, 'r')
        b = bound_file.readlines()
        p = point_file.readlines()
        points = []
        for p_line in p:
            p_line = p_line.replace('[', '')
            p_line = p_line.replace(']', '')
            words = p_line.split()
            points.append([float(words[0]),float(words[1]),float(words[2])])
        bounds = []
        for b_line in b:
            b_line = b_line.replace('[', '')
            b_line = b_line.replace(']', '')
            b_line = b_line.replace('(', '')
            b_line = b_line.replace(')', '')
            b_line = b_line.replace(',', '')
            words = b_line.split('array')
            d = float(words[0])
            n = words[1].replace('array','')
            n = n.split()
            s = words[2].replace('array', '')
            s = s.replace('\n', '')
            s = s.split()
            bo = boundary3D(d, float(n[0]), float(n[1]), float(n[2]), float(s[0]), float(s[1]), float(s[2]))
            bounds.append(bo)
        return np.array(points), bounds











