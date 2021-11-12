'''
Wrapper for VTK mesh.
'''
# Imports
import os
import numpy as np
import cv2 as cv
import vtk
from stl import mesh
from vtk_pointlaser.local_info import get_local_info, correct_rotations_m1
#from vtk.util.numpy_support import vtk_to_numpy

class Mesh:
    def __init__(self, mesh_nr, args, raycast_tol=1, renderer=None):
        self.args = args

        # Load mesh
        mesh_file = os.path.join(args.dataset_dir, args.mesh_dir, args.mesh_file + str(mesh_nr) + '.stl')
        # mesh, mesh_file, scale = self.transform(mesh_file=mesh_file)
        self._mesh = self.load_stl(mesh_file)
        # Load map
        map_file = os.path.join(args.dataset_dir, args.encodings_dir, args.encodings_dir + "_" + str(args.vae_latent_dims), args.mesh_file + str(mesh_nr) + '.npy')
        self._map_encoding = np.load(map_file)

        # Build BSP tree for ray-casting
        self._bsp_tree = vtk.vtkModifiedBSPTree()
        self._bsp_tree.SetDataSet(self._mesh)
        self._bsp_tree.BuildLocator()
        # Implicit function to find if point is inside/outside surface and at what distance
        # https://vtk.org/doc/nightly/html/classvtkImplicitFunction.html#details
        self._implicit_function = vtk.vtkImplicitPolyDataDistance()
        self._implicit_function.SetInput(self._mesh)
        # Ray casting tolerance
        self._tol = raycast_tol
        # Mesh boundaries
        bounds = self._mesh.GetBounds()
        self.min_bounds = np.array(bounds[::2])
        self.max_bounds = np.array(bounds[1::2])
        # Set-up rendering
        self.renderer = renderer

    @property
    def renderer(self):
        return self._renderer

    @renderer.setter
    def renderer(self, ren, color='LightGrey'):
        self._renderer = ren
        if ren is not None:
            # Add room mesh to scene
            room_actor = ren.add_object(self._mesh, color=color)
            # Modify visual properties
            room_actor.GetProperty().SetOpacity(0.6)
            ren.reset_camera()
            # Show mesh edges
            self._display_edges()

    def _display_edges(self):
        '''
        Highlight prominent mesh edges.
        '''
        feature_edges = vtk.vtkFeatureEdges()
        feature_edges.SetInputData(self._mesh)
        feature_edges.Update()
        # Visualize
        self._renderer.add_object(feature_edges, pipeline=True)

    @classmethod
    def load_stl(self, file):
        '''
        Load STL file into VTK mesh.
        '''
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file)
        # Read the STL file
        reader.Update()
        polydata = reader.GetOutput()
        # If there are no points in 'vtkPolyData' something went wrong
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError('No point data could be loaded from ' + file)
        return polydata

    def intersect(self, p1, p2):
        '''
        Intersect line segment defined by end points with the mesh.
        '''
        points = vtk.vtkPoints()
        # Perform intersection
        code = self._bsp_tree.IntersectWithLine(p1, p2, self._tol, points,
                                                None)
        if code == 0:
            # Return p2 if no intersection is found
            return p2.copy()
        return np.array(points.GetData().GetTuple3(0))

    def is_inside(self, position):
        '''
        Check if point is inside mesh.
        '''
        return self._implicit_function.FunctionValue(position) <= 0

    def sample_position(self, dmin=0, dmax=np.inf):
        '''
        Sample a random position inside mesh which is between dmin and dmax
        distance from mesh boundary. dmin is offset from mesh, dmax is laser range.
        '''
        while True:
            position = np.random.uniform(self.min_bounds + dmin,
                                         self.max_bounds - dmin)
            signed_dist = self._implicit_function.FunctionValue(position)
            if (dmin < -signed_dist < dmax):
                return position

    def sample_position_verification(self, dmin=0, dmax=np.inf):
        while True:
            position = np.random.uniform(self.min_bounds + dmin,
                                         self.max_bounds - dmin)
            signed_dist = self._implicit_function.FunctionValue(position)
            region, _ = correct_rotations_m1(position)
            if (dmin < -signed_dist < dmax) and region > 0:
                return position

    def transform(self, mesh_file):
        mesh_env = mesh.Mesh.from_file(mesh_file)
        data = mesh_env.data

        data['normals'] *= self.args.rescale
        data['vectors'] *= self.args.rescale
        data['attr'] *= self.args.rescale
        transformed_mesh = mesh.Mesh(data)
        mesh_path = os.path.join(self.args.dataset_dir, self.args.mesh_file + '_rescaled' + str(self.args.mesh_nr) + '.stl')
        transformed_mesh.save(mesh_path)

        return transformed_mesh, mesh_path, self.args.rescale
