import matplotlib.pyplot as plt
import numpy as np

# Function to create points for an arc
def create_arc(center, radius, start_angle, end_angle):
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num=100)
    points = np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])
    return points

def plot_arc_sec(start_angle_left, end_angle_left, radius_left):
    # Arc parameters for the left arc
    center_left = (0.5, 0.5)
    radius_left = radius_left
    start_angle_left = start_angle_left
    end_angle_left = end_angle_left

    # Arc parameters for the right arc
    center_right = (0.5, 0.5)
    radius_right = 0.2
    start_angle_right = -1*end_angle_left + 270 -90
    end_angle_right = 90 + (90-start_angle_left)

    # Create the left and right arcs
    arc_left_points = create_arc(center_left, radius_left, start_angle_left, end_angle_left)
    arc_right_points = create_arc(center_right, radius_right, start_angle_right, end_angle_right)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Fill the areas enclosed by the arcs and their connecting lines with transparency
    polygon_left = np.vstack((center_left, arc_left_points, center_left))
    polygon_right = np.vstack((center_right, arc_right_points, center_right))

    ax.fill(polygon_left[:, 0], polygon_left[:, 1], 'blue', alpha=0.5)
    ax.fill(polygon_right[:, 0], polygon_right[:, 1], 'blue', alpha=0.5)

    # Draw the arcs
    ax.plot(arc_left_points[:, 0], arc_left_points[:, 1], 'blue')
    ax.plot(arc_right_points[:, 0], arc_right_points[:, 1], 'blue')

    # Draw lines connecting the arcs to their centers
    ax.plot([center_left[0], arc_left_points[0, 0]], [center_left[1], arc_left_points[0, 1]], 'blue')
    ax.plot([center_left[0], arc_left_points[-1, 0]], [center_left[1], arc_left_points[-1, 1]], 'blue')
    ax.plot([center_right[0], arc_right_points[0, 0]], [center_right[1], arc_right_points[0, 1]], 'blue')
    ax.plot([center_right[0], arc_right_points[-1, 0]], [center_right[1], arc_right_points[-1, 1]], 'blue')

    # Set equal scaling
    ax.set_aspect('equal')
    #ax.set_xlim(0.23, 0.77)
    #ax.set_ylim(0.23, 0.77)

    # Show the plot
    plt.show()

'''
#red tailed hawks
a = (180-34)/2
b = a + 34 + 116.5
#plot_arc_sec(73.5, 229)
plot_arc_sec(a, b)


#human
plot_arc_sec(30, 190)

#cat
plot_arc_sec(20, 190)

# (180-{bino})/2 --> 20
# ({20} + {bino} + {mono})

#horse
a = (180-65)/2
b = a + 65 + 163
plot_arc_sec(57.5, 57.5+65+146)'''

for i in range(0,7,1):
    plot_arc_sec(0, 180, i)