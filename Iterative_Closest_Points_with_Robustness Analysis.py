# Import section
import json
import math
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
from pyvista import set_plot_theme
import PyQt5
from scipy.spatial import KDTree as kdt
from scipy.spatial.transform import Rotation as Rot
import sys
import time
import warnings

# End import section

# Config section

# General
warnings.filterwarnings("ignore")
set_plot_theme('document')

# Options
option_include_every_nth_point_source = 1 # If set to 1, it will take all the points available in the polygon file. Else it takes only the nth points. So you can reduce the density and therefor the calculation time in tradeoff for accuracy
option_include_every_nth_point_destination = 1  #
option_opacity_sphere = 0.45  # The opacity of the sphere
option_size_point_on_sphere = 16 # The size of the matched and unmatched points on the sphere.
option_color_matched_points = "green"  # Color of the matched points on the sphere
option_color_unmatched_points = "red"  # Color of the unmatched points on the sphere
option_color_histogram = "maroon"  # histogram color
option_width_histogram = 0.4  # histogram width
option_default_file_path_source = "C:/Users/Essi/Desktop/Ehsan Pazooki_ICP/Data/"  # Where the file is located
option_default_file_name_source = "Human_noise.ply"  # first poly model file
option_default_file_path_destination = "C:/Users/Essi/Desktop/Ehsan Pazooki_ICP/Data/"   # Where the file is located
option_default_file_name_destination = "Human_noise.ply"  # second poly model file
option_shift_parameter_for_shifting_destination_points = np.array([[0], [0], [
    0]])  # The shifting parameter, that will be applied to the second polygon-file, so you can make some more tests on it.
option_switch_covariance_or_rotation_translation_comparison = True  # True=>Covariance ; False=> rotation translation ; Sets the algorithm to use as calculating the ICP.
option_ICP_iterations = 500  # Maximum allowed quantity of iterations
option_show_estimate_every = 50  # After how many calculations do you want do get a new estimate on how long it will take to calculate. Lower numbers could result in spam. Higher numbers may be unsatisfying
option_radius_closest_points_correspondences = 3000  # In which radius in which the points shall be considered for the correspondence Matrix. Lower number means less matches. Maybe none. bigger Number means more matched, maybe also higher processing time. Depends on the scale and representation of the model
option_distance_threshold_for_matching_points = 50  # This is the threshold for matching points. If the rotated points will get in the threshold of the source point, than it'll be a match.
# Erklärung: diese Option gibt den maximalen Abstand an sodass ein Punkt als gemachted gilt --> wählt man die Option größer wird der matching percentage größer
option_example_point_sphere_rotation = [1, 0, 0]  # You may want to change this to the exact point you want to look at on the sphere. That will than be a accurate representation of that specific points movement. The initial point [1,0,0] is just an example point, where you can intuitivley feel/see the rotation of the model.
option_choose_show_labels_for_matches = 0  # You can choose between 0,1,2 or 3. 0 means no Labels will be displayed. 1 Means all Labels. 2 Means just the matches will be displayed and 3 means just the unmatched will be displayed.
option_label_sphere_point_size = 2  # The size of the point of the labels (recommend to be less than option_size_point_on_sphere).
option_label_sphere_font_size = 14  # The size of the label's text.
step_width_Opt3 = 10 # Step width for Option 3, in degree, for =1 approximately 1700 runs


# End config section

# Function definition section

# Data Plotting Function using pyvista
def pvPlot(P, Q, image, each_angle, pSize=5, qSize=5):
    srcPoints = pv.PolyData(Q.T)  # source datapoints
    dstPoints = pv.PolyData(P.T)  # destination datapoints
    plotter = pv.Plotter()  # pv plotter class from pyvista
    plotter.add_text("Point Plot", font_size=10)
    plotter.add_mesh(srcPoints,
                     color='lightblue',
                     point_size=pSize,
                     render_points_as_spheres=True,
                     show_scalar_bar=False,
                     label='Source Data')  # source points plotting

    plotter.add_mesh(dstPoints,
                     color='orange',
                     point_size=qSize,
                     render_points_as_spheres=True,
                     show_scalar_bar=False,
                     label='Destination Data')  # destination points plotting
    plotter.show_grid()
    plotter.add_legend(size=(0.1, 0.1))
    if image == "first":
        plotter.show(auto_close=False,
                     screenshot=f"images/Before_icp_{each_angle}_.png")  # {} must be used because the angle values are variable (not constant), ploting before ICP computation
    else:
        plotter.show(auto_close=False,
                     screenshot=f"images/after_icp_{each_angle}_.png")  # ploting after ICP computaion
    plotter.clear()
    plotter = "Helo"


# Generate a another one from one points array for Test
def generateDataFromSrc(srcPoints, angle, t):
    """
    This function, duplicates variable t,
    Generates rotational vector, and then
    the generated source vector is applied
    to source data points
    """
    shift = t
    R = Rot.from_rotvec(angle).as_matrix()
    return R.dot(srcPoints) + shift


# Data Centering
def centerData(data, excludeIndices=[]):  # Have verified everything. Should work as intendet.
    """
    This function is used to calculate the
    center data using mean of all datapoints.
    """
    reducedData = np.delete(data, excludeIndices,
                            axis=1)  # Return a new array with sub-arrays along an axis deleted, 1, columns
    center = np.array([reducedData.mean(axis=1)]).T
    return center, data - center


# Finding Corresponding Points(get closest points) --> (defining the corresponding matrix)
def getvalueCorrespondences(P, Q):  # Have verified everything. Should work as intendet.
    """
    This function is used to calculate the
    correspondence points, which are basically the
    closest points between source and destination matrix.
    """
    tree = kdt(Q.T) # kdt performs a nearest-neighbour algorithm for choosing the closest points of Q.
                    # Q.T = Matrix Q transposed
    ret = []
    pSize = P.shape[1]  # p matrix size
    for i in range(pSize):
        index = tree.query_ball_point(P[:, i], r=option_radius_closest_points_correspondences) # get all corresponding points in the tree with the given radius r
        if (index):  # in boolean terms a empty list is False and a list with items is true. So we check if there even are points to calculate with..
            _Q = Q[:, index] # takes one column at position index = Spaltenvektor
            # get same number of points as iterations from Q
            _t = np.abs(_Q.T - P[:, i]) # computes the distance between points of Q and P with the correct index
            # get the transpose of Q and subtract with P points iteration
            j = index[(np.sum(_t * _t, axis=-1) ** (1. / 2)).argmin()] # computes the index with the smallest deviation between Q and P points
            # append to ret with correspondence formula
            ret.append((i, j))
    return ret


# Computing Cross Covariance(to compute the covarianve between the data sources)
def calcCrosscovariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    """
    This function is used to calculate the crosscovariance
    or deterministic of a matrix.
    """
    cov = np.zeros((3, 3))  # generation of 3*3 zero matrix
    excludeIndices = []
    for i, j in correspondences:  # looping through correspondences list generated in getvalueCorrespondences() before calling this function
        pPoint = P[:, [i]]
        qPoint = Q[:, [j]]
        weight = kernel(pPoint - qPoint)
        if weight < 0.01: excludeIndices.append(i)
        cov += weight * qPoint.dot(pPoint.T)
    return cov, excludeIndices


# ICP algorithm
def computeICP(P, Q, iterations, switch_covariance_or_rotation_translation_comparison=True, kernel=lambda diff: 1.0):
    """
    This function uses the above functions to calculate ICP
    """
    centerpositionQ, Qcentered = centerData(Q)
    newP = P.copy()
    excludeIndices = []
    Rfound = Rot.from_rotvec([0, 0, 0]).as_matrix()
    tfound = [[0], [0], [0]]
    previous_cov = ""
    previous_R = ""
    previous_T = ""

    for i in range(iterations):
        cov_list = []
        R_list = []
        T_list = []
        previous_cov_list = []
        previous_R_list = []
        previous_T_list = []
        centerP, Pcentered = centerData(newP, excludeIndices)

        # Corresponding matrix
        correspondences = getvalueCorrespondences(Pcentered, Qcentered)
        # Computation of cross covariance matrix
        cov, excludeIndices = calcCrosscovariance(Pcentered, Qcentered, correspondences, kernel)

        # SVD computation
        U, S, VT = np.linalg.svd(cov)  # Singular Value Decomposition calculation
        R = U.dot(VT)
        t = centerpositionQ - R.dot(centerP)

        Rfound = Rfound.dot(R)
        tfound += t

        newP = R.dot(newP) + t
        normValue = np.linalg.norm(newP - Q)  # The linalg norm() function returns the norm of the given matrix, which is distance between q and p points
        isSuccess = True

        """
        Covariance comparision (fuer Abbruchkriterium)
        ----------------------
        Cross Covariance Matrix is one of the parameters, that 
        describe the output of the ICP. We can compare the norm 
        value of this matrix by each iteration with the norm value 
        of the same matrix by previous iteration. So we can recognize,
        if the Cross Covariance Matrix doesn't change any more, and 
        thus ICP Computaion could be stopped automathically, without 
        running useless and unnecessary iterations.

        Rotational_Translation_Comparision (fuer Abbruchkriterium)
        ----------------------------------
        This method of comparison is analog to the previous method. 
        But it uses the norm value of the Rotation Matrix.
        """
        if (option_switch_covariance_or_rotation_translation_comparison):  # covariance comparison
            if previous_cov != "":
                if np.linalg.norm(previous_cov-cov) < 1e-9:
                    return newP, normValue, isSuccess, Rfound, tfound, i, cov
        else:  # rotational_translation_comparison
            if previous_R != "" or previous_T != "":
                if np.linalg.norm(previous_T-t) < 1e-9 and np.linalg.norm(previous_R-r) < 1e-9:
                    return newP, normValue, isSuccess, Rfound, tfound, i, cov

        previous_cov = cov
        previous_R = R
        previous_T = t

    isSuccess = False
    if normValue < 10e-6:
        isSuccess = True

    # return values from main function
    return newP, normValue, isSuccess, Rfound, tfound, iterations, cov


# Calculate the right rotation matrix
def rotation_matrix(angle, order='xyz'):  # Have verified everything. Should work as intendet.
    """Converts a given angle of rotation to the euler matrix. You may than rotate your model with this matrix as you like.

    Parameters
    ----------
    angle : array[float,float,float]
        The rotation to generate the euler matrix from, represented as three angles in radians.
    order : string, optional
        The order in which the euler matrix shall be calculated. It has to be in the form of each axis separated by a comma.
        By default it is 'x,y,z'.

    Returns
    -------
    matrix(3x3)
        The euler matrix of the given rotation.
    """
    c1 = np.cos(angle[0])  # x
    s1 = np.sin(angle[0])  # x
    c2 = np.cos(angle[1])  # y
    s2 = np.sin(angle[1])  # y
    c3 = np.cos(angle[2])  # z
    s3 = np.sin(angle[2])  # z

    if order == 'xzx':
        matrix = np.array([[c2, -c3 * s2, s2 * s3],
                           [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                           [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]])
    elif order == 'xyx':
        matrix = np.array([[c2, s2 * s3, c3 * s2],
                           [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                           [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3]])
    elif order == 'yxy':
        matrix = np.array([[c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                           [s2 * s3, c2, -c3 * s2],
                           [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3]])
    elif order == 'yzy':
        matrix = np.array([[c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                           [c3 * s2, c2, s2 * s3],
                           [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3]])
    elif order == 'zyz':
        matrix = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                           [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                           [-c3 * s2, s2 * s3, c2]])
    elif order == 'zxz':
        matrix = np.array([[c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                           [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                           [s2 * s3, c3 * s2, c2]])
    elif order == 'xyz':
        matrix = np.array([[c2 * c3, -c2 * s3, s2],
                           [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                           [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])
    elif order == 'xzy':
        matrix = np.array([[c2 * c3, -s2, c2 * s3],
                           [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                           [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3]])
    elif order == 'yxz':
        matrix = np.array([[c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                           [c2 * s3, c2 * c3, -s2],
                           [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2]])
    elif order == 'yzx':
        matrix = np.array([[c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                           [s2, c2 * c3, -c2 * s3],
                           [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3]])
    elif order == 'zyx':
        matrix = np.array([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                           [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                           [-s2, c2 * s3, c2 * c3]])
    elif order == 'zxy':
        matrix = np.array([[c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                           [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                           [-c2 * s3, s2, c2 * c3]])
    return matrix


# Convert Angle to XYZ, so it can be projected onto the sphere
def angle_to_xyz(angle):  # Have verified everything. Should work as intendet.
    """Converts a given angle to xyz-coordinates.
    It'll calculate with the euler representation.
    There are also other representations, like sphere coordinates.
    Sphere coordinates is a coordinate system, that defines its points onto a sphere via two angles and a radius.
    Whereas the Euler representation the rotation of the model represents. It is defined through three angles.

    We are working with a model, that gets rotated and we want to be able to closely represent the rotation of the model onto a sphere,
    where we than can visually indicate if this specific rotation has matched with the ICP or not.
    Therefor we need to use the exact same rotation of the model and that's achived through the Euler matrix.
    We first calculate the Euler matrix of the corresponding angles and than create a normalized point that represents an example point in the model and is mapped onto the sphere.
    You may want to change this point in the options option_example_point_sphere_rotation to whatever you like.
    The point [1,0,0] is an intuitive example point that represents the movement of the whole model.
    This point will often be used as to explain the principle of the Euler angles, therefor it's so intuitive and the best fit for the whole model.
    However you may choose to change this point as it's just an example point.
    If you choose to only evaluate one point, than you definitely want to change it to a specific point of the model, so it represents exactly this points movement in the model on the sphere.

    Parameters
    ----------
    angle : array[float,float,float]
        The angles to generate the coordinates from in degree.

    Returns
    -------
    List(float,float,float)
        The coordinates on the sphere according to the chosen representation.
    """
    radian_angle = [np.radians(i) for i in angle]  # Convert to radians from degree
    eulerMatrix = rotation_matrix(radian_angle)
    point = option_example_point_sphere_rotation
    point = point / np.linalg.norm(point)
    return np.dot(eulerMatrix, point)


# Just convert every angle of a list via angle_to_xyz() to XYZ for projecting them onto a sphere
def anglelist_to_coordlist(anglelist):  # Have verified everything. Should work as intendet.
    """Converts a given list of angles to a list of xyz-coordinates.
    For more information see angle_to_xyz()'s documentation.

    Parameters
    ----------
    anglelist : array[array[float,float,float]]
        The angles to generate the coordinates from in degree.

    Returns
    -------
    Array[List(float,float,float)]
        The coordinates on the sphere according to the chosen representation.
    """
    return [angle_to_xyz(a) for a in anglelist]


# Sphere drawing with all the points
def draw_sphere(matched_list, unmatched_list):  # Have verified everything. Should work as intendet.
    """Displays the angles on a sphere.
    You are able to change the color of the matched and unmatched points as well as to change the size that the points will have on the sphere and the sphere's opacity.
    To do so you have to change option_color_matched_points, option_color_unmatched_points, option_size_point_on_sphere or option_opacity_sphere accordingly.

    Parameters
    ----------
    matched_list : array[array[float,float,float]]
        The matched angles.

    unmatched_list : array[array[float,float,float]]
        The not matched angles.

    Returns
    -------
    none
    """
    # Initialize the plot with sphere
    sphere = pv.Sphere(radius=1)
    plot = pv.Plotter()
    plot.add_mesh(sphere, opacity=option_opacity_sphere)
    #

    # Convert the angles to coords
    matched_list_coords = anglelist_to_coordlist(matched_list)
    unmatched_list_coords = anglelist_to_coordlist(unmatched_list)
    #

    # Convert coords to points and set onto the mesh of the sphere if there are any.
    if matched_list_coords:  # If there is any item in the list
        matched_points = pv.PointSet(points=matched_list_coords)
        plot.add_mesh(matched_points, color=option_color_matched_points,
                      point_size=option_size_point_on_sphere)  # chnage points size
    if unmatched_list_coords:  # if there is any item in that list
        unmatched_points = pv.PointSet(points=unmatched_list_coords)
        plot.add_mesh(unmatched_points, color=option_color_unmatched_points,
                      point_size=option_size_point_on_sphere)  # change points size

    # Labels
    shall_show_labels = True
    labels_list = []
    coords_of_labels_list = []
    if (option_choose_show_labels_for_matches == 1):  # show all matches and unmatches
        labels_list = matched_list + unmatched_list
        coords_of_labels_list = matched_list_coords + unmatched_list_coords
    elif (option_choose_show_labels_for_matches == 2):  # just show matches
        labels_list = matched_list
        coords_of_labels_list = matched_list_coords
    elif (option_choose_show_labels_for_matches == 3):  # just show unmatches
        labels_list = unmatched_list
        coords_of_labels_list = unmatched_list_coords
    else:  # Show no labels
        shall_show_labels = False

    if (shall_show_labels and coords_of_labels_list and labels_list):
        labels_sphere(plot, labels_list, coords_of_labels_list)

    plot.show_grid()
    plot.show(screenshot=option_default_file_path_destination + str('Sphere.png'))


def labels_sphere(plot, labels_list, coords_of_labels_list):
    """This function creates the labels for the sphere out of the coordinates and the angles.

    Parameters
    ----------
    plot : pv.Plotter()
        Where to plot the labels onto.

    labels_list : array[array[float,float,float]]
        The angles for the labels

    coords_of_labels_list : array[array[float,float,float]]
        The coords where the labels shall be located on the sphere

    Returns
    -------
    None
    """
    poly = pv.PolyData(coords_of_labels_list)
    poly["My Labels"] = [f"Angles {i}" for i in (labels_list)]
    plot.add_point_labels(poly, "My Labels", point_size=option_label_sphere_point_size,
                          font_size=option_label_sphere_font_size)


# Add the labels to histogram
def addlabels(x, y):
    """Adds labels at the x and y-axis for labeling the histogram's axes.

    Parameters
    ----------
    x : array[]
        The x-Labels

    y : array[]
        The y-Labels

    Returns
    -------
    none
    """
    for i in range(len(x)):
        plt.text(i, y[i], y[i])


# Draw/Plot the Histogram
def histogram_plotting(hist_list):  # Have verified everything. Should work as intendent.
    """Draws the Histogram
    Therefor it calculates the labels for the x and y-axis and shows them.

    Parameters
    ----------
    hist_list : array[float]
        The accuracy of the points that has been calculated

    Returns
    -------
    none
    """
    x_labels = ["0-20", "21-40", "41-60", "61-80", "81-99", "99-100"]
    a = len([i for i in hist_list if i < 20])
    b = len([i for i in hist_list if i > 20 and i < 40])
    c = len([i for i in hist_list if i > 40 and i < 60])
    d = len([i for i in hist_list if i > 60 and i < 80])
    e = len([i for i in hist_list if i > 80 and i < 99])
    f = len([i for i in hist_list if i > 99])
    y_values = [a, b, c, d, e, f]
    plt.figure()
    plt.bar(x_labels, y_values, color=option_color_histogram,
            width=option_width_histogram)
    addlabels(x_labels, y_values)
    plt.xlabel("Accuracy")
    plt.ylabel("Count of Accuracies")
    plt.title("Histogram of accuracy ranges")
    plt.savefig(option_default_file_path_destination + str('_histogram.png'))
    plt.show()



# Separates a nested list into a list. e.g. [[x1,y1,z1],[x2,y2,z2]]=>[x1,y1,z1,x2,y2,z2]
def separate_nested_list(nested_list):  # Have verified everything. Should work as intendet.
    """seperates a nested list.
    its like this conversion: [[x1,y1,z1],[x2,y2,z2], ...]=>[x1,y1,z1,x2,y2,z2, ...]

    Parameters
    ----------
    nested_list : array[]
        The list to be unnested

    Returns
    -------
    List()
        The list that has been unnested.
    """
    separated_list = []
    for inner_list in nested_list:
        for item in inner_list:
            separated_list.append(item)
    return separated_list


# Comparing the ICP iteration
def compare_data_of_icp_algorithm_of_nth_iteration(P, Q, angle, iteration, matched_list, unmatched_list,
                                                   hist_list):  # Have verified everything. Should work as intendet.
    """Compares the Icp algorithm's data with the n-th iteration.
    It'll calculate the percentage of matching point at the given percentage.

    Parameters
    ----------
    P : list(list())
        The destination datapoints.

    Q : list(list())
        The source datapoints.

    angle : array[float, float, float]
        The angle that was used in the icp for this calculation

    iteration : integer
        The iteration at which the icp was.

    matched_list : list()
        The list, where the points will be appended to, which did match

    unmatched_list : list()
        The list, where the points will be appended to, which did not match

    hist_list : list()
        The list, where the matching percentages will be appended to.

    Returns
    -------
    float
        The matching percentage of this calculation, which was also included in hist_list.
    """
    # get all individual points out of the nested lists P and Q
    p_list = separate_nested_list(P)
    q_list = separate_nested_list(Q)
    #

    # calculate the matching percent of the points
    matched_percent = 0
    if iteration < 5:  # If the iteration is less than 5, it'll always be 100%
        matched_percent = 100
    else:
        final_percent = []
        #for each in range(0, len(q_list)):  # We'll iterate through all points of the source
           # if (abs(p_list[each] - q_list[each]) < option_distance_threshold_for_matching_points) or (
            #        q_list[each] == p_list[
              #  each]):  # if the destination point is in range or exactly on point of the source point, than it will be true.
              #  final_percent.append(True)  # One Match
            #else:
              #  final_percent.append(False)  # Not a match
        matched_percent = ((np.sum(np.linalg.norm(P - Q, axis=0, ord=2) < option_distance_threshold_for_matching_points)) / np.shape(Q)[1]) * 100
        #matched_percent = final_percent.count(True) * 100 / len(final_percent)  # Matching points in percent.
    #

    hist_list.append(matched_percent)

    # plotting function calling
    if matched_percent > 98:  # if you scored at least 99%, than you match!
        matched_list.append(np.degrees(angle))
    else:
        unmatched_list.append(np.degrees(angle))

    return matched_percent


# Calculate ICP for one angle.
def calculate_ICP_for_given_angle(angles, P, Q, matched_list, unmatched_list, hist_list,
                                  toggle_stats=True):  # Have verified everything. Should work as intendet.
    """Compares the Icp algorithm's data with the n-th iteration.
    It'll calculate the percentage of matching point at the given percentage.

    Parameters
    ----------
    angles : array[array[float, float, float]]
        The angles, where the icp shall be calculated from.

    P : list(list())
        The destination datapoints.

    Q : list(list())
        The source datapoints.

    matched_list : list()
        The list, where the points will be appended to, which did match

    unmatched_list : list()
        The list, where the points will be appended to, which did not match

    hist_list : list()
        The list, where the matching percentages will be appended to.

    toggle_stats : boolean, optional
        A variable to toggle the displaying of the stats for the angle used. You may want to turn this off, if you compare a lot of angles, because it could spam.

    Returns
    -------
    None
    """
    iteration_counter = 0
    start = time.time()
    time_took = 0
    for angle in angles:
        # ICP Step 2: Shifting the second model
        P = generateDataFromSrc(P, angle, option_shift_parameter_for_shifting_destination_points)
        minPoints = min(P.shape[1], Q.shape[1])
        P, Q = P[:, :minPoints], Q[:, :minPoints]
        # Display polygons before ICP
        if (toggle_stats):
            pvPlot(P, Q, "first", str(np.degrees(angle)))
            # Calculate the ICP
        movedP, normValue, isSuccess, R, t, iteration, cov = computeICP(P, Q, option_ICP_iterations,
                                                                   option_switch_covariance_or_rotation_translation_comparison)

        # Display polygons after ICP
        if (toggle_stats):
            pvPlot(movedP, Q, "second", str(np.degrees(angle)))
        # compare the calculations
        matched_percent = compare_data_of_icp_algorithm_of_nth_iteration(movedP, Q, angle, iteration, matched_list,
                                                                         unmatched_list, hist_list)
        # Show some stats
        if (toggle_stats):
            print("\nRotation & Transform Matrix for, ", str(np.degrees(angle)), "degree\n")
            print("Rotation : \n", R)
            print("transform : \n", t)
            print("ICP stoped after ", iteration, " iterations")
            print("ICP Matched ", math.ceil(matched_percent), "%")
            print("Cross Covariance Matrix: \n", cov)
        iteration_counter += 1
        if (
                iteration_counter % option_show_estimate_every == 0):  # Every 'show_estimate_every'th-step you will be shown an estimate
            time_took = time.time() - start
            print("Time took so far in seconds: %f:" % time_took)
            print("The program will probably need %f total seconds to finish." % (
                        (time_took / iteration_counter) * len(angles)))


# End function definition section

# ICP step 1: ply file reading
srcData = o3d.io.read_point_cloud(option_default_file_path_source + option_default_file_name_source)
destData = o3d.io.read_point_cloud(option_default_file_path_destination + option_default_file_name_destination)
Q = np.asarray(srcData.points[::option_include_every_nth_point_source]).T
P = np.asarray(destData.points[::option_include_every_nth_point_destination]).T

matched_list = []
unmatched_list = []
hist_list = []

chosen_setup_finish = False
chosen_setup = 0
angles = []
while (not chosen_setup_finish):
    try:
        chosen_setup = int(input("You can choose between the following 3 setup's:\n"
                                 "1. Custom-One-Point-Setup(Type in 1 for this setup)\n"
                                 "You will be able to select one specific angle for each axis in this setup.\n"
                                 "2. Custom-View-All-Points-In-Range-Setup(Type in 2 for this setup)\n"
                                 "You will be able to pick the starting and ending angle for all axes together, as well as the steps for each individual axis.\n"
                                 "Then every possible combination will be calculated and shown in the endresults.\n"
                                 "3. 360°-Setup(Type in 3 for this setup)\n"
                                 "You won't be able to choose anything. It will go through every axis seperately in steps of 1° once around and then show the results.\n"
                                 "Which setup do you choose?:"))
    except ValueError:
        print("\nYou made a non valid input. Please only type in Numbers, nothing else!\n")
        chosen_setup = 0

    if (chosen_setup == 1):  # Custom one point
        chosen_setup_finish = True
        angles.append(np.radians(json.loads(input(
            "You picked the Custom-One-Point-Setup!\nPlease enter the X Y and Z angles in the form [x,y,z] in degree:"))))
        calculate_ICP_for_given_angle(angles, P, Q, matched_list, unmatched_list, hist_list,
                                      True)  # don't show stats, because it would be to much spam.
    elif (chosen_setup == 2):  # Custom view all
        chosen_setup_finish = True
        angle_start = json.loads(
            input("You picked the Custom-View-All-Points-In-Range-Setup!\nPlease enter the start angle in degree in [x,y,z] :"))
        angle_end = json.loads(input("Please enter the end angle in degree in [x,y,z] :"))
        step_width = json.loads(
            input("Please also enter the step width for each axis in the form of [x,y,z] in degree:"))
        print("Now every possible combination will be calculated")
        for x_angle in range(angle_start[0], angle_end[0] + 1, step_width[0]):  # x
            for y_angle in range(angle_start[1], angle_end[1] + 1, step_width[1]):  # y
                for z_angle in range(angle_start[2], angle_end[2] + 1, step_width[2]):  # z
                    angles.append(np.radians([x_angle, y_angle, z_angle]))
        calculate_ICP_for_given_angle(angles, P, Q, matched_list, unmatched_list, hist_list,
                                      False)  # don't show stats, because it would be to much spam.

        # Man definiert sein Gebiet mit den Start- und Endwinkel in x,y,z-Richtung, indem man die Start und Endwinkel für jede Richtung,...
        # ... sowie die Verfeinerung mittels der step_width Variable angibt.

    elif (chosen_setup == 3):  # 360°
        chosen_setup_finish = True
        print("You picked the 360°-Setup!\nNow every axis will be iterated through!")
        for angle in range(0, 360, step_width_Opt3):  # from 0 to 360° in steps of 1
            angles.append(np.radians([angle, 90, 0]))  # x-Axis
            # angles.append(np.radians([0, angle, 0]))  # y-Axis
            # angles.append(np.radians([0, 0, angle]))  # z-Axis
            # angles.append(np.radians([45, 0, angle]))  # Inbetween axis
            # #angles.append(np.radians([0, 45, angle]))  # Inbetween axis
            # #angles.append(np.radians([0, 135, angle]))  # Inbetween axis
            # angles.append(np.radians([135, 0, angle]))  # Inbetween axis


            # The options above (under the choosen_setup == 3) can be uncommented and changed

        calculate_ICP_for_given_angle(angles, P, Q, matched_list, unmatched_list, hist_list,
                                      False)  # don't show stats, because it would be to much spam.
        # Erklärung: Das Gebiet, auf der Kugel, wird in mehreren definierten Linien abgesucht
    else:
        print("\nYour input has not contained the right numbers. Please choose either 1, 2 or 3!\nRetry!\n")

histogram_plotting(hist_list)  # Display the Histogram
draw_sphere(matched_list, unmatched_list)  # Display the Sphere

'''
used Links:
https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb

Die Quelle zum Pyvista und Visualisierungen damit:
https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.html

Die Quelle zum Abbruchkriterium:
https://www.tutorialspoint.com/python-program-to-check-if-two-given-matrices-are-identical
https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

Die Quelle zum kdtree und Integration davon:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html

http://open3d.org/html/tutorial/Basic/kdtree.html#Using-search_radius_vector_3d

Die Quelle zur Bewertung von dem Abbruchkriterium:
https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/amp/

Die Quelle zum Histogramm:
Quelle: Hermann Schichl, Roland Steinbauer: Einführung in das mathematische Arbeiten. 2. überarbeitete Auflage. Springer, 2012, ISBN 978-3-642-28646-9, S. 382 ff.)
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

Die Quellen zur Rotations- und Transformationsmatrix :
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
https://www.petercollingridge.co.uk/tutorials/3d/pygame/matrix-transformations/
Die Quellen zur Euler Matrix:
https://stackoverflow.com/questions/1568568/how-to-convert-euler-angles-to-directional-vector
https://programming-surgeon.com/en/euler-angle-python-en/
https://en.wikipedia.org/wiki/Euler_angles




Die Quelle zum Numpy Array:
https://numpy.org/doc/stable/reference/generated/numpy.array.html


Die Quelle zur Methode, um die Punkte der Modelle beim Bedarf teilweise zu löschen:
https://note.nkmk.me/en/python-numpy-delete/


'''