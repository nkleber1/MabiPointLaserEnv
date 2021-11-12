import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
num_images = 25
mean = 5
std = 3

def generate():
    for i in range(num_images):
        e = round(np.random.normal(mean, std))
        while e < 3:
            e = round(np.random.normal(mean, std))

        [x, y] = simple_polygon(e)

        # pgon = polyshape(2 * x, 2 * y);
        # figure1 = figure('visible', 'off');
        # plot(pgon);
        # set(gca, 'Visible', 'off')
        # saveas(figure1, strcat('plots_2/r_map', int2str(i), '.png'));
        # fprintf('Generating polygon: %d  \n', i);
#
        # fid = strcat('poly_points_2/r_map', int2str(i), '.csv');
        # csvwrite(fid, [x, y]);


def free_boundary(tri):
    boundary = set()
    for i in range(len(tri.neighbors)):
        for k in range(3):
            if tri.neighbors[i][k] == -1:
                nk1, nk2 = (k+1)%3, (k+2)%3
                boundary.add(tri.simplices[i][nk1])
                boundary.add(tri.simplices[i][nk2])
    return np.array(list(boundary))


def simple_polygon(numSides):  # return function [x, y, dt]
    # oldState = warning('off', 'MATLAB:TriRep:PtsNotInTriWarnId');

    fudge = np.ceil(numSides/10)
    points = np.random.uniform(size=(int(numSides+fudge), 2))
    tri = Delaunay(points)
    boundary_edges = free_boundary(tri)

    # points = points[boundary_edges]
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    numEdges = len(boundary_edges)

    while numEdges is not numSides:
        if numEdges > numSides:
            triIndex = vertexAttachments(dt, boundaryEdges(:,1))
            triIndex = triIndex(randperm(numel(triIndex)))
            keep = (cellfun('size', triIndex, 2) ~= 1)

        if (numEdges < numSides) or all(keep):
            triIndex = edgeAttachments(dt, boundaryEdges);
            triIndex = triIndex(randperm(numel(triIndex)));
            triPoints = dt([triIndex{:}], :);
            keep = all(ismember(triPoints, boundaryEdges(:,1)), 2);
#
    #    if all(keep)
    #        warning('Couldn''t achieve desired number of sides!');
    #        break
#
    #    triPoints = dt.Triangulation
    #    triPoints(triIndex{find(~keep, 1)}, :) = []
    #    dt = TriRep(triPoints, x, y)
    #    boundaryEdges = freeBoundary(dt)
    #    numEdges = size(boundaryEdges, 1)

    print(boundary_edges)
    print('1')
    x = dt.X(boundary_edges, 1)
    y = dt.X(boundary_edges, 2)
    return x, y

generate()