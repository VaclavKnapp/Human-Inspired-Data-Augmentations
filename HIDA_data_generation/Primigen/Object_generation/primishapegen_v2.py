import os
import sys
import argparse
import pickle
import bpy
import numpy as np
from mathutils import Vector
from pathlib import Path
import cv2
import random
import shutil
import time
import uuid
import json
from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial.transform import Rotation
import tempfile

def check_placement_diversity_simple(primitive_tracking, warp_cache_file, min_distance=0.3):
    """
    Simple placement diversity check using temporary file for current warp.
    """
    # Extract centroids from current configuration
    current_centroids = []
    for primitive in primitive_tracking:
        centroid = primitive.get('final_centroid', [0, 0, 0])
        current_centroids.append(centroid)
    
    # Load existing centroids for this warp (if any)
    existing_centroids_list = []
    if os.path.exists(warp_cache_file):
        try:
            with open(warp_cache_file, 'rb') as f:
                existing_centroids_list = pickle.load(f)
        except:
            existing_centroids_list = []
    
    # Check against existing placements in this warp
    for i, existing_centroids in enumerate(existing_centroids_list):
        if len(current_centroids) != len(existing_centroids):
            continue
            
        # Calculate max distance between corresponding centroids
        max_distance = 0
        for curr_c, exist_c in zip(current_centroids, existing_centroids):
            dist = np.linalg.norm(np.array(curr_c) - np.array(exist_c))
            max_distance = max(max_distance, dist)
        
        # If all centroids are too close, reject
        if max_distance < min_distance:
            return False, f"Too close to placement {i+1} (max distance: {max_distance:.3f})"
    
    # Accept this placement and save centroids
    existing_centroids_list.append(current_centroids)
    try:
        with open(warp_cache_file, 'wb') as f:
            pickle.dump(existing_centroids_list, f)
    except Exception as e:
        print(f"Warning: Could not save centroids: {e}")
    
    return True, f"Diverse placement (total placements in warp: {len(existing_centroids_list)})"

# Common functions (from common.py - assumed to be available)
def normalize(v):
    return v / np.linalg.norm(v)

def rotateVector(points, axis, angle):
    """Rotate points around axis by angle (in radians)"""
    axis = normalize(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    rotated = points * cos_angle + np.cross(axis, points) * sin_angle + \
              axis * np.dot(points, axis).reshape(-1, 1) * (1 - cos_angle)
    return rotated

def subPix(image, x, y):
    """Bilinear interpolation for sub-pixel access"""
    # Clamp coordinates to valid range
    x = np.clip(x, 0, image.shape[1] - 1.0001)
    y = np.clip(y, 0, image.shape[0] - 1.0001)
    
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
    
    dx, dy = x - x0, y - y0
    
    return (1 - dx) * (1 - dy) * image[y0, x0] + \
           dx * (1 - dy) * image[y0, x1] + \
           (1 - dx) * dy * image[y1, x0] + \
           dx * dy * image[y1, x1]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')


def find_texture_images(texture_folder):
    """Find all texture images in the specified folder"""
    if not texture_folder:
        return None
    
    import os
    import glob
    
    # Valid texture file extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    
    # Find all files with valid extensions
    texture_images = []
    for ext in extensions:
        texture_images.extend(glob.glob(os.path.join(texture_folder, f'*{ext}')))
        texture_images.extend(glob.glob(os.path.join(texture_folder, f'*{ext.upper()}')))
    
    # Sort for deterministic results (with seed)
    texture_images.sort()
    
    if not texture_images:
        print(f"No texture images found in {texture_folder}")
        return None
    
    print(f"Found {len(texture_images)} texture images in {texture_folder}")
    return texture_images


class Shape(object):
    def __init__(self):
        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

    def genShape(self):
        self.points = np.reshape([], (-1,3)).astype(float)
        self.uvs = np.reshape([], (-1,2)).astype(float)
        self.faces = np.reshape([], (-1,3)).astype(int)
        self.facesUV = np.reshape([], (-1,3)).astype(int)
        self.matNames = []
        self.matStartId = []

    def permuteMatIds(self, ratio = 0.25):
        if len(self.matStartId) == 0:
            print("no mats")
            return
        newIds = [self.matStartId[0]]

        for i in range(1, len(self.matStartId)):
            neg = self.matStartId[i] - self.matStartId[i-1]
            negCount = -int(ratio * neg)

            if i != len(self.matStartId) - 1:
                pos = self.matStartId[i+1] - self.matStartId[i]
            else:
                pos = len(self.faces) - self.matStartId[i]

            posCount = int(ratio * pos)

            offset = np.random.permutation(posCount - negCount)[0] + negCount

            newIds.append(self.matStartId[i] + offset)
        self.matStartId = newIds

    def computeNormals(self):
        vec0 = self.points[self.faces[:,1]-1] - self.points[self.faces[:,0]-1]
        vec1 = self.points[self.faces[:,2]-1] - self.points[self.faces[:,1]-1]
        areaNormals = np.cross(vec0, vec1)
        self.normals = self.points.copy()
        vertFNs = np.zeros(len(self.points), int)
        vertFMaps = np.zeros((len(self.points), 200), int)
        for iF, face in enumerate(self.faces):
            for id in face:
                vertFMaps[id-1, vertFNs[id-1]] = iF
                vertFNs[id-1] += 1

        for i in range(len(self.points)):
            faceNormals = areaNormals[vertFMaps[i,:vertFNs[i]]]
            normal = np.average(faceNormals, axis=0)
            self.normals[i] = normalize(normal).reshape(-1)
        return self.normals

    def translate(self, translation):
        self.points += translation

    def rotate(self, axis, degAngle):
        self.points = rotateVector(self.points, axis, np.deg2rad(degAngle))

    def reCenter(self):
        minP = np.min(self.points, 0)
        maxP = np.max(self.points, 0)
        center = 0.5*minP + 0.5*maxP
        self.translate(-center)

    def addShape(self, otherShape, shape_id):
        curPN = len(self.points)
        curUN = len(self.uvs)
        curFN = len(self.faces)
        if curPN == 0:
            self.points = np.copy(otherShape.points)
            self.uvs = np.copy(otherShape.uvs)
            self.faces = np.copy(otherShape.faces)
            self.facesUV = np.copy(otherShape.facesUV)
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
            num_new_mats = len(otherShape.matNames)
            self.materialSubShapeIdx = [shape_id] * num_new_mats
        else:
            self.points = np.vstack([self.points, otherShape.points])
            
            # Transform UVs to avoid overlaps between sub-shapes
            uv_offset_x = (shape_id % 5) * 0.2
            uv_offset_y = (shape_id // 5) * 0.2
            uv_offset = np.array([uv_offset_x, uv_offset_y])
            
            new_uvs = otherShape.uvs.copy()
            new_uvs = new_uvs * 0.18
            new_uvs = new_uvs + uv_offset
            
            self.uvs = np.vstack([self.uvs, new_uvs])
            self.faces = np.vstack([self.faces, otherShape.faces + curPN])
            self.facesUV = np.vstack([self.facesUV, otherShape.facesUV + curUN])
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
            num_new_mats = len(otherShape.matNames)
            
            if not hasattr(self, 'materialSubShapeIdx'):
                self.materialSubShapeIdx = []
            self.materialSubShapeIdx += [shape_id] * num_new_mats

    def genObj(self, filePath, bMat=False, bComputeNormal=False, bScaleMesh=False,
            bMaxDimRange=[0.3, 0.5], texture_images=None, texture_rng=None):
        """Write a .blend file containing the generated object using Blender's API"""

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Convert numpy arrays to lists for Blender
        verts = [tuple(point) for point in self.points]
        faces = [tuple([idx - 1 for idx in face]) for face in self.faces]
        uvs = [tuple(uv) for uv in self.uvs]
        faces_uv = [tuple(idx - 1 for idx in face_uv) for face_uv in self.facesUV]

        # Create a new mesh and object in Blender
        mesh = bpy.data.meshes.new('mesh')
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new('Object', mesh)
        bpy.context.collection.objects.link(obj)

        # Create UV map
        uv_layer = mesh.uv_layers.new(name='UVMap')

        # Assign UV coordinates to each loop
        for face_idx, face in enumerate(mesh.polygons):
            face_uv_indices = faces_uv[face_idx]
            for loop_idx, uv_idx in zip(face.loop_indices, face_uv_indices):
                uv_layer.data[loop_idx].uv = uvs[uv_idx]

        # Optionally compute normals
        if bComputeNormal:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

        # Create materials and assign to faces
        if texture_images and texture_rng:
            # Create a mapping from face to subshape index
            face_to_subshape = np.zeros(len(self.faces), dtype=int)
            
            # Fill the face_to_subshape mapping
            for mat_idx in range(len(self.matNames)):
                if hasattr(self, 'materialSubShapeIdx') and mat_idx < len(self.materialSubShapeIdx):
                    subshape_idx = self.materialSubShapeIdx[mat_idx]
                else:
                    subshape_idx = 0
                    
                start_idx = self.matStartId[mat_idx]
                if mat_idx < len(self.matStartId) - 1:
                    end_idx = self.matStartId[mat_idx + 1]
                else:
                    end_idx = len(self.faces)
                    
                for face_idx in range(start_idx, end_idx):
                    if face_idx < len(face_to_subshape):
                        face_to_subshape[face_idx] = subshape_idx
            
            # Get unique subshape indices
            unique_subshapes = sorted(set(face_to_subshape))
            num_subshapes = len(unique_subshapes)
            
            # Create texture mapping for each subshape
            subshape_texture_paths = {}
            
            if len(texture_images) < num_subshapes:
                available_textures = list(texture_images)
                
                for subshape_idx in unique_subshapes:
                    if available_textures:
                        texture_image_path = texture_rng.choice(available_textures)
                        available_textures.remove(texture_image_path)
                    else:
                        texture_image_path = texture_rng.choice(texture_images)
                    subshape_texture_paths[subshape_idx] = texture_image_path
            else:
                selected_indices = texture_rng.choice(len(texture_images), num_subshapes, replace=False)
                for i, subshape_idx in enumerate(unique_subshapes):
                    subshape_texture_paths[subshape_idx] = texture_images[selected_indices[i]]
            
            # Create materials for each subshape
            materials = []
            subshape_to_material = {}
            
            for subshape_idx in unique_subshapes:
                texture_image_path = subshape_texture_paths[subshape_idx]
                
                material_index = len(materials)
                mat = bpy.data.materials.new(name=f"Material_Shape_{subshape_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is None:
                    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

                tex_image_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
                try:
                    tex_image_node.image = bpy.data.images.load(texture_image_path)
                except:
                    print(f"Failed to load image {texture_image_path}")
                    continue

                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])

                materials.append(mat)
                subshape_to_material[subshape_idx] = material_index

                obj.data.materials.append(mat)
            
            material_indices = [0] * len(mesh.polygons)
            for face_idx, subshape_idx in enumerate(face_to_subshape):
                if face_idx < len(material_indices):
                    material_indices[face_idx] = subshape_to_material[subshape_idx]
        else:
            mat = bpy.data.materials.new(name="Default_Material")
            obj.data.materials.append(mat)
            material_indices = [0] * len(mesh.polygons)

        for poly, mat_idx in zip(mesh.polygons, material_indices):
            poly.material_index = mat_idx

        # Save the Blender file
        bpy.ops.wm.save_as_mainfile(filepath=filePath)

        # Clean up
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

        max_dim = max(self.points.max(axis=0) - self.points.min(axis=0))
        
        # Adjust material_ids to match the number of materials
        material_ids = [0] * len(self.matNames)
        return max_dim, material_ids


class HeightFieldCreator:
    def __init__(self, initSize=(5, 5), maxHeight=(-0.2, 0.2), bFixCorner=True, rng=None):
        self.initSize = initSize
        self.bFixCorner = bFixCorner
        self.initNum = self.initSize[0] * self.initSize[1]
        self.maxHeight = maxHeight
        self.heightField = None
        self.rng = rng if rng is not None else np.random

    def __initializeHeigthField(self):
        heights = self.rng.uniform(self.maxHeight[0], self.maxHeight[1], self.initNum)
        initHeightField = heights.reshape(self.initSize)
        self.initHeightField = initHeightField
        return initHeightField

    def genHeightField(self, targetSize = (36, 36)):
        halfSize = (int(targetSize[0]/6*5), int(targetSize[1]/6*5))
        if halfSize[0] < self.initSize[0] or halfSize[1] < self.initSize[1]:
            print("target size should be double as init size")
            return None
        initHeight = self.__initializeHeigthField()
        if self.bFixCorner:
            bounder = np.zeros((self.initSize[0]+2, self.initSize[1]+2))
            bounder[1:-1, 1:-1] = initHeight
            initHeight = bounder
        heightField_half = cv2.resize(initHeight, halfSize, interpolation=cv2.INTER_CUBIC)
        if self.bFixCorner:
            bounder = np.zeros(halfSize)
            bounder[1:-1, 1:-1] = heightField_half[1:-1, 1:-1]
            initHeight = bounder
        heightField = cv2.resize(initHeight, targetSize)
        self.heightField = heightField
        self.targetSize = targetSize
        return heightField


class Ellipsoid(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, meshRes=(50, 100)):
        super(Ellipsoid, self).__init__()
        if meshRes[1] % 2 != 0:
            print("WARN: longitude res is supposed to be even")
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes
        self.numPoints = (self.meshRes[0] - 2) * self.meshRes[1] + 2

    def genShape(self, matName="mat"):
        super(Ellipsoid, self).__init__()

        # Add north pole
        self.points.append((0, 0, self.axisC))
        self.uvs.append((0.5, 0))

        # Create points for the middle bands
        for iy in range(1, self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)

                theta = np.pi/2.0 - v * np.pi
                phi = u * np.pi

                x = self.axisA * np.cos(theta) * np.cos(phi)
                y = self.axisB * np.cos(theta) * np.sin(phi)
                z = self.axisC * np.sin(theta)

                self.points.append((x, y, z))

        # Add south pole
        self.points.append((0, 0, -self.axisC))

        # Create UVs for middle bands
        for iy in range(1, self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u, v))

        # Add UV for south pole
        self.uvs.append((0.5, 1.0))

        # Create faces
        # Top cap (connecting to north pole)
        north_pole_idx = 1  # 1-based indexing
        for ix in range(self.meshRes[1]):
            p1 = 1 + ix + 1  # Current point on first ring
            p2 = 1 + ((ix + 1) % self.meshRes[1]) + 1  # Next point on first ring (with wrap)
            
            self.faces.append((north_pole_idx, p2, p1))

        # Middle bands
        for iy in range(self.meshRes[0] - 3):  # Between middle bands
            for ix in range(self.meshRes[1]):
                # Current ring
                p1 = 1 + 1 + iy * self.meshRes[1] + ix
                p2 = 1 + 1 + iy * self.meshRes[1] + ((ix + 1) % self.meshRes[1])
                
                # Next ring
                p3 = 1 + 1 + (iy + 1) * self.meshRes[1] + ix
                p4 = 1 + 1 + (iy + 1) * self.meshRes[1] + ((ix + 1) % self.meshRes[1])
                
                # Create two triangles for this quad
                self.faces.append((p1, p4, p2))
                self.faces.append((p1, p3, p4))

        # Bottom cap (connecting to south pole)
        south_pole_idx = 1 + (self.meshRes[0] - 2) * self.meshRes[1] + 1
        last_ring_start = 1 + 1 + (self.meshRes[0] - 3) * self.meshRes[1]
        
        for ix in range(self.meshRes[1]):
            p1 = last_ring_start + ix
            p2 = last_ring_start + ((ix + 1) % self.meshRes[1])
            
            self.faces.append((south_pole_idx, p1, p2))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = [matName]
        self.matStartId = np.asarray([0], int)

    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print("wrong shape of heightfiels")
            return False
        if len(heightFields.shape) == 3:
            heightField = heightFields[0]

        for i,point in enumerate(self.points):
            uv = self.uvs[i]
            normal = np.reshape(point,-1) / (self.axisA, self.axisB, self.axisC)
            normal = normal / np.linalg.norm(normal)
            
            # Ensure UV coordinates are in valid range
            u = np.clip(uv[0], 0, 1)
            v = np.clip(uv[1], 0, 1)
            
            xy = np.array([u, v]) * (heightField.shape[1] - 1, heightField.shape[0] - 1)
            h = subPix(heightField, xy[0], xy[1])
            self.points[i] = point + normal * h

class Cylinder(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, meshRes=(50, 150), radiusRes=20):
        super(Cylinder, self).__init__()
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes

    def genShape(self, matName="mat"):
        super(Cylinder, self).__init__()

        # Create center points for top and bottom caps
        center_top = (0, 0, self.axisC)
        center_bottom = (0, 0, -self.axisC)
        self.points.append(center_top)    # Point 1
        self.points.append(center_bottom) # Point 2
        # Create points for the cylindrical surface
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)
                z = self.axisC - self.axisC * v * 2.0

                self.points.append((x, y, z))

        # Create UVs
        self.uvs.append((0.5, 0.5))  # Center of top cap
        self.uvs.append((0.5, 0.5))  # Center of bottom cap
        
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u, v))

        # Create faces for the cylindrical surface
        point_offset = 2  # Account for the two center points
        for iy in range(self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                curId = point_offset + iy * self.meshRes[1] + ix + 1
                bottomId = point_offset + (iy + 1) * self.meshRes[1] + ix + 1
                if ix == self.meshRes[1] - 1:
                    rightId = point_offset + iy * self.meshRes[1] + 1
                    rightBottomId = point_offset + (iy + 1) * self.meshRes[1] + 1
                else:
                    rightId = point_offset + (iy) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = point_offset + (iy + 1) * self.meshRes[1] + ix + 1 + 1

                self.faces.append((curId, rightBottomId, rightId))
                self.faces.append((curId, bottomId, rightBottomId))

        # Create top cap faces (connecting to center_top at index 1)
        for ix in range(self.meshRes[1]):
            curId = point_offset + ix + 1  # First ring
            if ix == self.meshRes[1] - 1:
                nextId = point_offset + 1  # Wrap around
            else:
                nextId = point_offset + ix + 1 + 1
            
            self.faces.append((1, nextId, curId))  # Note the winding order

        # Create bottom cap faces (connecting to center_bottom at index 2)
        bottom_ring_offset = point_offset + (self.meshRes[0] - 1) * self.meshRes[1]
        for ix in range(self.meshRes[1]):
            curId = bottom_ring_offset + ix + 1  # Last ring
            if ix == self.meshRes[1] - 1:
                nextId = bottom_ring_offset + 1  # Wrap around
            else:
                nextId = bottom_ring_offset + ix + 1 + 1
            
            self.faces.append((2, curId, nextId))  # Note the winding order

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = ["%s_0" % matName, "%s_top" % matName, "%s_bottom" % matName]
        
        # Calculate material start indices
        cylinder_faces = (self.meshRes[0] - 1) * self.meshRes[1] * 2
        top_cap_faces = self.meshRes[1]
        self.matStartId = np.asarray([0, cylinder_faces, cylinder_faces + top_cap_faces], int)

    def applyHeightField(self, heightFields, smoothCircleBoundRate = 0.25):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            heightField = heightFields
        else:
            heightField = heightFields[0]

        i = 0
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                u = float(ix) / (self.meshRes[1] / 2)
                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)

                normal = np.reshape((x,y,0),-1)/ (self.axisA, self.axisB, self.axisC)
                normal = normal / np.linalg.norm(normal)
                
                # Ensure UV coordinates are in valid range
                uv_clipped = np.clip(self.uvs[i], 0, 1)
                xy = uv_clipped * (heightField.shape[1] - 1, heightField.shape[0] - 1)
                h = subPix(heightField, xy[0], xy[1])

                self.points[i] += normal * h
                i+=1

class Cube(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, faceRes=(50, 50)):
        super(Cube,self).__init__()
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.faceRes = faceRes
        self.pointNumPerFace = faceRes[0] * faceRes[1]
        self.numPoints = (self.faceRes[0]) * self.faceRes[1] * 6

    def genShape(self, matName = "mat"):
        super(Cube, self).__init__()

        #uvs
        for iy in range(self.faceRes[0]):
            for ix in range(self.faceRes[1]):
                u = float(ix) / (self.faceRes[1] - 1)
                v = float(iy) / (self.faceRes[0] - 1)
                self.uvs.append((u,v))
        self.uvs = np.reshape(self.uvs, (-1,2))

        #face:
        oneFaces = []
        for iy in range(self.faceRes[0] - 1):
            for ix in range(self.faceRes[1] - 1):
                curId = iy * self.faceRes[1] + ix + 1
                rightId = iy * self.faceRes[1] + ix + 1 + 1
                bottomId = (iy + 1) * self.faceRes[1] + ix + 1
                rightBottomId = (iy + 1) * self.faceRes[1] + ix + 1 + 1

                oneFaces.append((curId, rightBottomId, rightId))
                oneFaces.append((curId, bottomId, rightBottomId))
        oneFaces = np.reshape(oneFaces, (-1,3)).astype(int)
        self.faces = np.vstack([oneFaces,
                                oneFaces + self.pointNumPerFace,
                                oneFaces + self.pointNumPerFace*2,
                                oneFaces + self.pointNumPerFace*3,
                                oneFaces + self.pointNumPerFace*4,
                                oneFaces + self.pointNumPerFace*5])
        self.facesUV = self.faces.copy()

        #points
        #front
        for uv in self.uvs:
            xy = uv * (self.axisA, -self.axisB) * 2.0 + (-self.axisA, self.axisB)
            point = (xy[0], xy[1], self.axisC)
            self.points.append(point)

        # back
        for uv in self.uvs:
            xy = uv * (self.axisA, self.axisB) * 2.0 + (-self.axisA, -self.axisB)
            point = (xy[0], xy[1], -self.axisC)
            self.points.append(point)

        #left
        for uv in self.uvs:
            zy = uv * (self.axisC, -self.axisB) * 2.0 + (-self.axisC, self.axisB)
            point = (-self.axisA, zy[1], zy[0])
            self.points.append(point)

        # right
        for uv in self.uvs:
            zy = uv * (-self.axisC, -self.axisB) * 2.0 + (self.axisC, self.axisB)
            point = (self.axisA, zy[1], zy[0])
            self.points.append(point)

        # up
        for uv in self.uvs:
            xz = uv * (self.axisA, self.axisC) * 2.0 + (-self.axisA, -self.axisC)
            point = (xz[0], self.axisB, xz[1])
            self.points.append(point)

        # down
        for uv in self.uvs:
            xz = uv * (self.axisA, -self.axisC) * 2.0 + (-self.axisA, self.axisC)
            point = (xz[0], -self.axisB, xz[1])
            self.points.append(point)

        self.uvs = np.reshape(np.vstack([self.uvs, self.uvs, self.uvs, self.uvs, self.uvs, self.uvs]), (-1, 2))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matNames = ["%s_%d"%(matName, 0),
                         "%s_%d"%(matName, 1),
                         "%s_%d"%(matName, 2),
                         "%s_%d"%(matName, 3),
                         "%s_%d"%(matName, 4),
                         "%s_%d"%(matName, 5)]
        numFacePerFace = len(oneFaces)
        self.matStartId = np.asarray([0,
                                      numFacePerFace,
                                      numFacePerFace*2,
                                      numFacePerFace*3,
                                      numFacePerFace*4,
                                      numFacePerFace*5],int)

    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(6):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 6:
                newH = []
                for i in range(6):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        # modify points for each face
        normals = [(0,0,1), (0,0,-1), (-1,0,0), (1,0,0), (0,1,0), (0,-1,0)]
        
        for face_idx in range(6):
            heightField = heightFields[face_idx]
            normal = np.asarray(normals[face_idx])
            offSet = self.pointNumPerFace * face_idx
            
            for i in range(self.pointNumPerFace):
                uv = self.uvs[i]
                # Ensure UV coordinates are in valid range
                u = np.clip(uv[0], 0, 1)
                v = np.clip(uv[1], 0, 1)
                
                xy = np.array([u, v]) * (heightField.shape[1] - 1, heightField.shape[0] - 1)
                h = subPix(heightField, xy[0], xy[1])
                self.points[i + offSet] = self.points[i + offSet] + h * normal

class ShapeGen(Shape):
    """
    A shape generated by the Blender shape_generator addon.
    This class wraps the generation process and makes it compatible with the Shape interface.
    """
    def __init__(self, n_extrusions=2, n_subdivisions=0, target_scale=1.0, rng=None):
        super(ShapeGen, self).__init__()
        self.n_extrusions = n_extrusions
        self.n_subdivisions = n_subdivisions
        self.target_scale = target_scale  # Now should be the MAX axis value, not mean
        self.rng = rng if rng is not None else np.random
        
        # Parameters for shape generation
        self.axis_preferences = [
            self.rng.uniform(0, 0.2),
            self.rng.uniform(0.6, 1),
            self.rng.uniform(0, 1)
        ]
        self.seed = self.rng.randint(1000)
        self.big_shapeseed = self.rng.randint(1000)
        self.big_locseed = self.rng.randint(1000)
        
    def genShape(self, matName="mat"):
        """Generate the abstract shape using the shape_generator addon"""
        super(ShapeGen, self).__init__()
        
        # Delete everything in the scene
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
        for coll in bpy.data.collections:
            bpy.data.collections.remove(coll)
        
        # Generate object using the shape_generator addon
        bpy.ops.mesh.shape_generator(
            random_seed=self.seed,
            mirror_x=0,
            mirror_y=0,
            mirror_z=0,
            favour_vec=self.axis_preferences,
            amount=self.n_extrusions,
            is_subsurf=(self.n_subdivisions > 0),
            subsurf_subdivisions=self.n_subdivisions,
            big_shape_num=1,
            medium_shape_num=0,
            big_random_seed=self.big_shapeseed,
            big_random_scatter_seed=self.big_locseed,
        )
        
        # Join all objects into one
        for obj in bpy.data.objects:
            obj.select_set(True)
        
        if len(bpy.context.selected_objects) > 1:
            bpy.ops.object.join()
        
        # Get the generated object
        obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        
        if obj is None:
            print("No object generated!")
            # Create a simple cube as last resort
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.active_object
        
        # Center the object
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
        bpy.ops.object.location_clear(clear_delta=False)
        
        # Scale to match the maximum dimension of other primitives
        def mesh_max_dimension(ob):
            """Get the maximum dimension of the object's bounding box"""
            from mathutils import Vector
            bbox = [Vector(b) for b in ob.bound_box]
            min_coord = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
            max_coord = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
            dimensions = max_coord - min_coord
            return max(dimensions.x, dimensions.y, dimensions.z)
        
        current_max_dim = mesh_max_dimension(obj)
        if current_max_dim > 0:
            # Scale so the maximum dimension matches the target_scale
            # target_scale should be the maximum axis value from other primitives
            # Add some controlled variation (±10%) to avoid perfectly uniform sizes
            size_variation = self.rng.uniform(0.9, 1.5)
            desired_max_dim = self.target_scale * size_variation
            scale_factor = desired_max_dim / current_max_dim
            obj.scale *= scale_factor
            
            print(f"ShapeGen scaling: current_max_dim={current_max_dim:.3f}, "
                  f"target_scale={self.target_scale:.3f}, "
                  f"desired_max_dim={desired_max_dim:.3f}, "
                  f"scale_factor={scale_factor:.3f}")
        
        # Apply the scale
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Extract mesh data
        mesh = obj.data
        
        # Extract vertices
        self.points = np.array([v.co[:] for v in mesh.vertices])
        
        # Extract faces (convert to 1-based indexing)
        faces_list = []
        for p in mesh.polygons:
            # Handle both triangles and quads
            if len(p.vertices) == 3:
                faces_list.append([v + 1 for v in p.vertices])
            elif len(p.vertices) == 4:
                # Split quad into two triangles
                verts = p.vertices
                faces_list.append([verts[0] + 1, verts[1] + 1, verts[2] + 1])
                faces_list.append([verts[0] + 1, verts[2] + 1, verts[3] + 1])
            else:
                # For n-gons, create a fan triangulation
                verts = p.vertices
                for i in range(1, len(verts) - 1):
                    faces_list.append([verts[0] + 1, verts[i] + 1, verts[i+1] + 1])
        
        self.faces = np.array(faces_list) if faces_list else np.array([[1, 2, 3]])  # Fallback triangle
        
        # Create simple UV coordinates
        # First ensure we have a UV layer
        if not mesh.uv_layers:
            mesh.uv_layers.new(name='UVMap')
        
        uv_layer = mesh.uv_layers.active
        
        # Extract UV coordinates
        uvs_dict = {}
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vertex_idx = mesh.loops[loop_idx].vertex_index
                if vertex_idx not in uvs_dict:
                    uv = uv_layer.data[loop_idx].uv
                    uvs_dict[vertex_idx] = (uv.x, uv.y)
        
        # Create UV array in vertex order
        self.uvs = np.array([uvs_dict.get(i, (0.5, 0.5)) for i in range(len(self.points))])
        
        # For faces UV, use same indices as faces (since we have one UV per vertex)
        self.facesUV = self.faces.copy()
        
        # Set material info
        self.matNames = [matName]
        self.matStartId = np.asarray([0], int)
        
        # Clean up
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
    def applyHeightField(self, heightFields):
        """Apply height field deformation to the shape"""
        if len(self.points) == 0:
            print("no points")
            return False
            
        # For ShapeGen, we'll apply a simple height field based on UV coordinates
        if len(heightFields.shape) == 2:
            heightField = heightFields
        else:
            heightField = heightFields[0] if len(heightFields) > 0 else np.zeros((36, 36))
        
        # Calculate normals for each vertex
        normals = np.zeros_like(self.points)
        
        # Calculate face normals and accumulate at vertices
        for face in self.faces:
            v0, v1, v2 = self.points[face[0]-1], self.points[face[1]-1], self.points[face[2]-1]
            face_normal = np.cross(v1 - v0, v2 - v0)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
            
            for vertex_idx in face:
                normals[vertex_idx-1] += face_normal
        
        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] /= norm
        
        # Apply height field
        for i, (point, uv) in enumerate(zip(self.points, self.uvs)):
            # Ensure UV coordinates are in valid range [0, 1]
            u = np.clip(uv[0], 0, 1)
            v = np.clip(uv[1], 0, 1)
            
            # Convert to height field coordinates
            xy = np.array([u, v]) * (heightField.shape[1] - 1, heightField.shape[0] - 1)
            h = subPix(heightField, xy[0], xy[1])
            self.points[i] = point + normals[i] * h * 0.5  # Scale factor for height




class MultiShape(Shape):
    """
    Shape types:
        0 – Ellipsoid
        1 – Cube
        2 – Cylinder
        3 – ShapeGen (abstract shape from Blender addon)
    """
    def __init__(self,
                 numShape=None,
                 candShapes=(0, 1, 2, 3),  # Now includes ShapeGen
                 shape_counts=None,
                 smoothPossibility=0.1,
                 axisRange=(0.35, 1.55),
                 heightRangeRate=(0, 0.2),
                 rotateRange=(0, 180),
                 translation_control=None,
                 rotation_rng=None,
                 translation_rng=None,
                 size_rng=None,
                 shapegen_extrusions=(2, 4),  # Range for ShapeGen extrusions
                 shapegen_subdivisions=0):     # Subdivisions for ShapeGen
        super().__init__()

        self.shape_counts      = shape_counts
        self.candShapes        = list(candShapes)
        self.smoothPossibility = smoothPossibility
        self.axisRange         = (0.7, 1.2) if axisRange == (0.35, 1.55) else axisRange
        self.heightRangeRate   = heightRangeRate
        self.rotateRange       = rotateRange
        self.translation_control = translation_control if translation_control is not None else 1.0
        self.shapegen_extrusions = shapegen_extrusions
        self.shapegen_subdivisions = shapegen_subdivisions

        self.rotation_rng    = rotation_rng    if rotation_rng    is not None else np.random
        self.translation_rng = translation_rng if translation_rng is not None else np.random
        self.size_rng        = size_rng        if size_rng        is not None else np.random

        # number of primitives
        if self.shape_counts is not None:
            self.numShape = sum(self.shape_counts.values())
        elif numShape is not None:
            self.numShape = int(numShape)
        else:
            self.numShape = 6

        self.materialSubShapeIdx = []

    def _create_shape(self, shape_type, axis_vals, shape_params=None):
            if shape_type == 0:
                return Ellipsoid(*axis_vals)
            if shape_type == 1:
                return Cube(*axis_vals)
            if shape_type == 2:
                return Cylinder(*axis_vals)
            if shape_type == 3:
                target_scale = np.max(axis_vals) * 2.0
                n_extrusions = shape_params.get('n_extrusions', 2) if shape_params else 2
                n_subdivisions = shape_params.get('n_subdivisions', 0) if shape_params else 0
                return ShapeGen(n_extrusions, n_subdivisions, target_scale, rng=self.size_rng)
            raise ValueError(f"Unknown shape type {shape_type}")

    def _get_shape_info(self, shape, position=None):
        if position is None:
            position = np.zeros(3)
        pts   = shape.points + position
        return dict(points      = pts,
                    center      = pts.mean(0),
                    min         = pts.min(0),
                    max         = pts.max(0),
                    dimensions  = pts.max(0) - pts.min(0))

    def _spin_about_point(self, shape, pivot, rng):
        """Rotate a shape in‑place around a pivot, keeping that pivot fixed."""
        ang = rng.uniform(0.0, 360.0, 3)
        shape.translate(-pivot)
        shape.rotate((1, 0, 0), ang[0])
        shape.rotate((0, 1, 0), ang[1])
        shape.rotate((0, 0, 1), ang[2])
        shape.translate(pivot)

    def _sample_random_point(self, shape, rng):
        """Uniform surface sampler."""
        faces = shape.faces
        verts = shape.points

        v0 = verts[faces[:, 0] - 1]
        v1 = verts[faces[:, 1] - 1]
        v2 = verts[faces[:, 2] - 1]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        f_idx = np.searchsorted(np.cumsum(areas), rng.random() * areas.sum())

        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v
        return u * v0[f_idx] + v * v1[f_idx] + w * v2[f_idx]

    def genShape(self, no_hf=False):
        """Generate a multi-shape object with robust tracking of transformations."""
        super().genShape()
        rng_r, rng_t, rng_s = self.rotation_rng, self.translation_rng, self.size_rng

        def _random_axes():
            axis = rng_s.uniform(self.axisRange[0], self.axisRange[1], 3)
            
            if rng_s.rand() < 0.9:  # 90% chance of stretching
                stretch_type = rng_s.rand()
                
                if stretch_type < 0.5:  # Stretch one axis
                    sa = rng_s.randint(0, 3)
                    sf = rng_s.uniform(1.2, 2.2)
                    tf = rng_s.uniform(0.2, 0.6)
                    
                    axis[sa] *= sf
                    axis[[i for i in range(3) if i != sa]] *= tf
                else:  # Stretch two axes
                    thin_axis = rng_s.randint(0, 3)
                    stretch_axes = [i for i in range(3) if i != thin_axis]
                    
                    sf1 = rng_s.uniform(1.2, 2.0)
                    sf2 = rng_s.uniform(1.2, 2.0)
                    tf = rng_s.uniform(0.3, 0.6)
                    
                    axis[stretch_axes[0]] *= sf1
                    axis[stretch_axes[1]] *= sf2
                    axis[thin_axis] *= tf
                
            return axis

        # Define shape specifications
        specs = []
        if self.shape_counts:
            for stype, cnt in self.shape_counts.items():
                specs.extend([dict(type=stype)] * cnt)
        else:
            for _ in range(self.numShape):
                specs.append(dict(type=rng_r.choice(self.candShapes)))

        # Generate shape details
        for spec in specs:
            # All shape types get proper axis values now
            spec['axis'] = _random_axes()
            
            if spec['type'] == 3:  # ShapeGen additional parameters
                spec['n_extrusions'] = rng_s.randint(self.shapegen_extrusions[0], 
                                                     self.shapegen_extrusions[1] + 1)
                spec['n_subdivisions'] = self.shapegen_subdivisions
            
            # Height fields
            hfs = []
            if spec['type'] == 3:  # ShapeGen uses fewer height fields
                minA = spec['axis'].min() * 2.0
                maxH = rng_s.uniform(self.heightRangeRate[0] * minA,
                                    self.heightRangeRate[1] * minA, 1)
            else:
                minA = spec['axis'].min() * 2.0
                maxH = rng_s.uniform(self.heightRangeRate[0] * minA,
                                    self.heightRangeRate[1] * minA, 6)
            
            for m in maxH:
                if no_hf or m == 0 or rng_s.rand() < self.smoothPossibility:
                    hfs.append(np.zeros((36, 36)))
                else:
                    hfg = HeightFieldCreator(maxHeight=(-m, m), rng=rng_s)
                    hfs.append(hfg.genHeightField())
            spec['hfs'] = np.asarray(hfs)
            spec['rot'] = rng_r.uniform(self.rotateRange[0],
                                       self.rotateRange[1], 3)

        # Initialize tracking data for each primitive
        primitive_tracking = []
        for i, spec in enumerate(specs):
            primitive_tracking.append({
                'type': spec['type'],
                'index': i,
                'primitive_id': f"type{spec['type']}_idx{i}",
                'axis_vals': spec['axis'].copy(),
                'initial_rotation': spec['rot'].copy(),
                'initial_translation': np.zeros(3),
                'additional_rotations': [],
                'additional_translations': [],
                'connection_points': [],
                'height_fields': spec['hfs'],
                'final_centroid': None,
                'final_rotation_matrix': None
            })
            
            # Add ShapeGen-specific parameters if applicable
            if spec['type'] == 3:
                primitive_tracking[i]['n_extrusions'] = spec['n_extrusions']
                primitive_tracking[i]['n_subdivisions'] = spec['n_subdivisions']

        # Create actual shape objects
        shapes = []
        for i, spec in enumerate(specs):
            shape_params = None
            if spec['type'] == 3:
                shape_params = {
                    'n_extrusions': spec['n_extrusions'],
                    'n_subdivisions': spec['n_subdivisions']
                }
            
            shp = self._create_shape(spec['type'], spec['axis'], shape_params)
            shp.genShape(f"mat_{i}")
            shp.applyHeightField(spec['hfs'])
            
            # Apply initial rotations
            shp.rotate((1, 0, 0), spec['rot'][0])
            shp.rotate((0, 1, 0), spec['rot'][1])
            shp.rotate((0, 0, 1), spec['rot'][2])
            
            shapes.append(shp)

        # Process the first shape
        anchor0 = self._sample_random_point(shapes[0], rng_t)
        shapes[0].translate(-anchor0)
        
        # Track this translation
        primitive_tracking[0]['additional_translations'].append(-anchor0)
        primitive_tracking[0]['connection_points'].append(np.zeros(3))
        
        self.addShape(shapes[0], 0)
        
        # Initialize assembled shape for subsequent connections
        assembled = Shape()
        assembled.points = np.copy(shapes[0].points)
        assembled.faces = np.copy(shapes[0].faces)
        
        # Process remaining shapes
        for i in range(1, len(shapes)):
            # Find connection point on the assembled shape
            connection_point = self._sample_random_point(assembled, rng_t)
            
            # Find point on the new shape to connect
            shape_point = self._sample_random_point(shapes[i], rng_t)
            
            # Calculate and apply translation
            trans = connection_point - shape_point
            shapes[i].translate(trans)
            
            # Track this translation and connection point
            primitive_tracking[i]['additional_translations'].append(trans)
            primitive_tracking[i]['connection_points'].append(connection_point.copy())
            
            # Generate and track spin rotation angles
            spin_angles = rng_r.uniform(0.0, 360.0, 3)
            primitive_tracking[i]['additional_rotations'].append(spin_angles.copy())
            
            # Apply spin rotation
            self._spin_about_point(shapes[i], connection_point, rng_r)
            
            # Add to self for final output
            self.addShape(shapes[i], i)
            
            # Update assembled shape for next primitive
            points_offset = len(assembled.points)
            new_points = shapes[i].points.copy()
            assembled.points = np.vstack([assembled.points, new_points])
            
            new_faces = shapes[i].faces.copy()
            new_faces = new_faces + points_offset 
            
            if len(assembled.faces) > 0 and len(new_faces) > 0:
                assembled.faces = np.vstack([assembled.faces, new_faces])
            elif len(new_faces) > 0:
                assembled.faces = new_faces
        
        # Track recentering 
        center_before_recenter = np.mean(self.points, axis=0) if len(self.points) > 0 else np.zeros(3)
        self.reCenter()
        center_after_recenter = np.mean(self.points, axis=0) if len(self.points) > 0 else np.zeros(3)
        recenter_translation = center_after_recenter - center_before_recenter
        
        # Apply recenter translation to all tracking
        for i in range(len(primitive_tracking)):
            primitive_tracking[i]['additional_translations'].append(recenter_translation)
        
        # Calculate final centroids and rotation matrices
        for i in range(len(specs)):
            # Find all points belonging to this primitive based on materialSubShapeIdx
            primitive_points = []
            
            if hasattr(self, 'materialSubShapeIdx') and len(self.materialSubShapeIdx) > 0:
                for face_idx, subshape_idx in enumerate(self.materialSubShapeIdx):
                    if subshape_idx == i and face_idx < len(self.faces):
                        # Get points for this face
                        for vertex_idx in self.faces[face_idx]:
                            if 0 <= vertex_idx-1 < len(self.points):
                                primitive_points.append(self.points[vertex_idx-1])
            
            if primitive_points:
                centroid = np.mean(np.array(primitive_points), axis=0)
            else:
                centroid = np.zeros(3)
            
            primitive_tracking[i]['final_centroid'] = centroid
            
            # Calculate final rotation matrix
            try:
                from scipy.spatial.transform import Rotation as R
                
                # Start with identity matrix
                rot_matrix = np.eye(3)
                
                # Apply initial rotations (XYZ order)
                initial_rot = primitive_tracking[i]['initial_rotation']
                rot_x = R.from_euler('x', initial_rot[0], degrees=True).as_matrix()
                rot_y = R.from_euler('y', initial_rot[1], degrees=True).as_matrix() 
                rot_z = R.from_euler('z', initial_rot[2], degrees=True).as_matrix()
                rot_matrix = rot_matrix @ rot_x @ rot_y @ rot_z
                
                # Apply additional rotations from spins
                for add_rot in primitive_tracking[i]['additional_rotations']:
                    rot_x = R.from_euler('x', add_rot[0], degrees=True).as_matrix()
                    rot_y = R.from_euler('y', add_rot[1], degrees=True).as_matrix()
                    rot_z = R.from_euler('z', add_rot[2], degrees=True).as_matrix()
                    rot_matrix = rot_matrix @ rot_x @ rot_y @ rot_z
                
                primitive_tracking[i]['final_rotation_matrix'] = rot_matrix
            except ImportError:
                primitive_tracking[i]['final_rotation_matrix'] = np.eye(3)
                print(f"Warning: scipy not available, using identity matrix for rotation")
        
        # Store the primitive tracking data
        self.primitive_tracking = primitive_tracking
        
        # Return format for compatibility
        primitive_ids, axis_vals_s, translations = [], [], []
        translation1s, rotations = [], []
        rotation1s, height_fields_s = [], []
        final_centroids_list, final_rotation_matrices = [], []
        
        for i, spec in enumerate(specs):
            primitive_ids.append(spec['type'])
            axis_vals_s.append(spec['axis'])
            translations.append(primitive_tracking[i]['additional_translations'][0] if primitive_tracking[i]['additional_translations'] else np.zeros(3))
            translation1s.append(np.zeros(3))
            rotations.append(spec['rot'])
            rotation1s.append(np.zeros(3))
            height_fields_s.append(spec['hfs'])
            final_centroids_list.append(primitive_tracking[i]['final_centroid'])
            final_rotation_matrices.append(primitive_tracking[i]['final_rotation_matrix'])
        
        return (primitive_ids, axis_vals_s, translations, translation1s, 
                rotations, rotation1s, height_fields_s,
                final_centroids_list, final_rotation_matrices)


def load_shape_generator_addon():
    """Load and enable the shape_generator addon"""
    print('-------LOADING SHAPE GENERATOR------')
    
    # Try to load the addon directly from the operators.py file
    addon_file_path = os.path.join(os.getcwd(), 'shape_generator', 'operators.py')
    
    if not os.path.exists(addon_file_path):
        print(f"Addon file not found at {addon_file_path}. Please ensure the addon is installed.")
        return False
    
    try:
        # First, try to import the operators module directly
        import sys
        shape_gen_dir = os.path.join(os.getcwd(), 'shape_generator')
        if shape_gen_dir not in sys.path:
            sys.path.append(shape_gen_dir)
        
        # Try to import and register the operator directly
        try:
            from operators import ShapeGeneratorOperator
            bpy.utils.register_class(ShapeGeneratorOperator)
            print('Successfully registered ShapeGeneratorOperator')
            return True
        except ImportError:
            print("Could not import ShapeGeneratorOperator directly, trying addon installation...")
        
        # If direct import fails, try the addon installation method
        bpy.ops.preferences.addon_install(filepath=addon_file_path)
        
        # Try to enable with just the module name
        try:
            bpy.ops.preferences.addon_enable(module='shape_generator')
        except:
            # If that fails, try enabling just the operators module
            try:
                bpy.ops.preferences.addon_enable(module='operators')
            except:
                print("Could not enable addon through preferences")
                return False
        
        bpy.ops.wm.save_userpref()
        print('-------FINISHED LOADING SHAPE GENERATOR------')
        return True
        
    except Exception as e:
        print(f"Error installing addon: {e}")
        
        # As a fallback, try to manually register the operator
        try:
            # Read and execute the operators.py file directly
            with open(addon_file_path, 'r') as f:
                addon_code = f.read()
            
            # Execute in the current namespace
            exec(addon_code, globals())
            
            # Try to register the ShapeGeneratorOperator if it exists
            if 'ShapeGeneratorOperator' in globals():
                bpy.utils.register_class(globals()['ShapeGeneratorOperator'])
                print('Successfully registered ShapeGeneratorOperator via direct execution')
                return True
        except Exception as fallback_error:
            print(f"Fallback registration also failed: {fallback_error}")
        
        return False


def createVarObjShapes(outFolder, shapeIds, uuid_str='', shape_counts=None,
                       sub_obj_nums=[1,2,3,4,5,6,7,8,9], sub_obj_num_poss=[1,2,3,7,10,7,3,2,1],
                       bMultiObj=False, bPermuteMat=True, candShapes=[0,1,2,3],
                       bScaleMesh=False, bMaxDimRange=[0.3,0.5], smooth_probability=1.0,
                       no_hf=False, texture_images=None, texture_rng=None,
                       translation_control=None, rotation_rng=None, translation_rng=None, size_rng=None,
                       shapegen_extrusions=(2, 4), shapegen_subdivisions=0, enable_placement_diversity=False,
                       min_centroid_distance=0.3, max_placement_attempts=10, warp_id=""):
    """
    Create shapes with robust tracking of primitives and transformations.
    Now includes ShapeGen as shape type 3.
    """
    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)

    output_paths = []
    shapes_parameters = []

    warp_cache_file = None
    if enable_placement_diversity and warp_id:
        warp_cache_file = os.path.join(tempfile.gettempdir(), f"warp_centroids_{warp_id}.pkl")

    if shape_counts is not None:
        fixed_sub_obj_num = sum(shape_counts.values())
        fixed_shape_counts = shape_counts
    else:
        # Randomly select sub_obj_num
        sub_obj_bound = np.reshape(sub_obj_num_poss, -1).astype(float)
        sub_obj_bound = sub_obj_bound / np.sum(sub_obj_bound)
        sub_obj_bound = np.cumsum(sub_obj_bound)

        if sub_obj_bound[-1] != 1.0:
            print("Incorrect bound")
            sub_obj_bound[-1] = 1.0

        counts = np.zeros(len(sub_obj_nums))
        chooses = np.random.uniform(0, 1.0, len(shapeIds))

    for ii, i in enumerate(shapeIds):
        shape_parameters = {'uuid_str': uuid_str}

        success = False
        attempts = 0

        for attempt in range(max_placement_attempts):
            attempts += 1

            if shape_counts is not None:
                sub_obj_num = fixed_sub_obj_num
                shape_parameters['sub_obj_num'] = sub_obj_num
                shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]
                print(f'i: {i}, sub_obj_num: {sub_obj_num}')
                ms = MultiShape(shape_counts=shape_counts,
                                smoothPossibility=smooth_probability,
                                axisRange=(0.7, 1.2),
                                translation_control=translation_control,
                                rotation_rng=rotation_rng,
                                translation_rng=translation_rng,
                                size_rng=size_rng,
                                shapegen_extrusions=shapegen_extrusions,
                                shapegen_subdivisions=shapegen_subdivisions)
            else:
                # Randomly select sub_obj_num
                choose = chooses[ii]
                sub_obj_num = sub_obj_nums[-1]
                for iO in range(len(sub_obj_bound)):
                    if choose < sub_obj_bound[iO]:
                        sub_obj_num = sub_obj_nums[iO]
                        counts[iO] += 1
                        break
                shape_parameters['sub_obj_num'] = sub_obj_num
                shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]
                print(f'i: {i}, sub_obj_num: {sub_obj_num}')
                ms = MultiShape(numShape=sub_obj_num,
                                candShapes=candShapes,
                                smoothPossibility=smooth_probability,
                                axisRange=(0.7, 1.2),
                                translation_control=translation_control,
                                rotation_rng=rotation_rng,
                                translation_rng=translation_rng,
                                size_rng=size_rng,
                                shapegen_extrusions=shapegen_extrusions,
                                shapegen_subdivisions=shapegen_subdivisions)


            sub_objs_vals = list(ms.genShape(no_hf=no_hf))
            
            if bPermuteMat:
                ms.permuteMatIds()

            if enable_placement_diversity and warp_cache_file:
                if hasattr(ms, 'primitive_tracking'):
                    is_diverse, reason = check_placement_diversity_simple(
                        ms.primitive_tracking, warp_cache_file, min_centroid_distance
                    )
                    
                    if is_diverse:
                        print(f"✓ Placement {ii+1} accepted on attempt {attempt+1}: {reason}")
                        success = True
                        break
                    else:
                        print(f"✗ Placement {ii+1} attempt {attempt+1} rejected: {reason}")
                        # Regenerate with different seeds
                        np.random.seed(np.random.randint(0, 100000))
                        random.seed(np.random.randint(0, 100000))
                        continue
                else:
                    print("Warning: No primitive tracking, accepting without diversity check")
                    success = True
                    break
            else:
                # No diversity checking, accept immediately
                success = True
                break

        if not success:
            print(f"Failed to generate diverse placement {ii+1} after {max_placement_attempts} attempts")
            continue

        shape_parameters['placement_attempts'] = attempts

        # Define the output file path
        filename = f'object_{ii:03d}.blend'
        output_path = Path(outFolder) / filename
        output_paths.append(str(output_path.resolve()))

        max_dim, material_ids = ms.genObj(
            str(output_path),
            bMat=True,
            bComputeNormal=True,
            bScaleMesh=bScaleMesh,
            bMaxDimRange=bMaxDimRange,
            texture_images=texture_images,
            texture_rng=texture_rng
        )
        shape_parameters['max_dim'] = max_dim

        # Extract all primitive tracking data if available
        primitive_tracking = getattr(ms, 'primitive_tracking', None)
        
        # Adjust material_ids to match the number of sub-objects if needed
        if not material_ids or len(material_ids) != sub_obj_num:
            material_ids = list(range(sub_obj_num))
        
        # Extract the final centroids and rotation matrices
        final_centroids = sub_objs_vals[7] if len(sub_objs_vals) > 7 else []
        final_rotations = sub_objs_vals[8] if len(sub_objs_vals) > 8 else []
        
        # Add these to sub_objs_vals for backward compatibility
        sub_objs_vals = list(sub_objs_vals)
        while len(sub_objs_vals) < 9:
            sub_objs_vals.append([])
        sub_objs_vals.append(material_ids)

        # Map parameter names to values from sub_objs_vals
        param_keys = ['primitive_id', 'axis_vals', 'translation', 'translation1', 
                     'rotation', 'rotation1', 'height_fields', 'final_centroids', 
                     'final_rotations', 'material_id']
        
        # Ensure all shapes have valid parameter values
        for iS in range(sub_obj_num):
            for i_key, key in enumerate(param_keys):
                if i_key < len(sub_objs_vals):
                    sub_obj_val = sub_objs_vals[i_key]
                    if isinstance(sub_obj_val, list) and len(sub_obj_val) > iS:
                        val = sub_obj_val[iS]
                    else:
                        val = 0
                    
                    # Convert numpy arrays to lists for JSON serialization
                    shape_parameters['sub_objs'][iS][key] = val.tolist() if isinstance(val, np.ndarray) else val
        
        # Add detailed primitive tracking if available
        if primitive_tracking:
            for iS in range(min(sub_obj_num, len(primitive_tracking))):
                for key, val in primitive_tracking[iS].items():
                    if key not in shape_parameters['sub_objs'][iS]:
                        # Convert numpy arrays to lists for JSON serialization
                        if isinstance(val, np.ndarray):
                            shape_parameters['sub_objs'][iS][key] = val.tolist()
                        elif isinstance(val, list) and all(isinstance(x, np.ndarray) for x in val):
                            shape_parameters['sub_objs'][iS][key] = [x.tolist() for x in val]
                        else:
                            shape_parameters['sub_objs'][iS][key] = val
        
        # Create primitive_data list with enhanced identification
        primitive_data = []
        for iS in range(sub_obj_num):
            primitive_type = shape_parameters['sub_objs'][iS].get('primitive_id', 0)
            
            # Create a unique identifier that combines type and index
            primitive_id = f"type{primitive_type}_idx{iS}"
            
            # Get centroid and rotation data
            centroid = shape_parameters['sub_objs'][iS].get('final_centroids', [0, 0, 0])
            rotation_matrix = shape_parameters['sub_objs'][iS].get('final_rotations', np.eye(3).tolist())
            
            # Create primitives with full metadata for matching
            primitive_data.append({
                'type': primitive_type,
                'shape_index': iS,
                'primitive_id': primitive_id,
                'centroid': centroid,
                'rotation_matrix': rotation_matrix,
                'axis_vals': shape_parameters['sub_objs'][iS].get('axis_vals', [1, 1, 1])
            })
            
            # Add ShapeGen-specific parameters if type is 3
            if primitive_type == 3:
                primitive_data[-1]['n_extrusions'] = shape_parameters['sub_objs'][iS].get('n_extrusions', 2)
                primitive_data[-1]['n_subdivisions'] = shape_parameters['sub_objs'][iS].get('n_subdivisions', 0)
        
        # Sort primitive data by type and then by index for consistent matching
        primitive_data.sort(key=lambda x: (x.get('type', 0), x.get('shape_index', 0)))
        
        # Add to shape parameters
        shape_parameters['primitive_data'] = primitive_data
        shapes_parameters.append(shape_parameters)

    # Save shape parameters to pickle files
    for i, output_path in enumerate(output_paths):
        pickle_path = output_path.replace('.blend', '.pkl')
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(shapes_parameters[i], f)
            print(f"Saved shape parameters to {pickle_path}")
        except Exception as e:
            print(f"Error saving to {pickle_path}: {e}")

    return output_paths, shapes_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create shapes")
    parser.add_argument('--output_dir', default='outputs', help='output directory')
    parser.add_argument('--num_shapes', default=1, type=int, help='number of shapes to create')
    parser.add_argument('--uuid_str', default='', type=str, help='uuid to use for the shape')
    parser.add_argument('--seed', default=0, type=int, help='seed for random number generation')
    parser.add_argument('--smooth_probability', default=1.0, type=float, help='probability of smoothing the height field')
    parser.add_argument('--shape_counts', type=str,
                        help='Comma-separated list of shape counts, e.g., cube:2,ellipsoid:3,shapegen:1')
    parser.add_argument('--no_hf', default=True, action='store_true', help='do not use height field')
    parser.add_argument('--texture_folder', type=str, help='Folder containing texture images')
    parser.add_argument('--texture_seed', type=int, default=0, help='Seed for texture selection')
    parser.add_argument('--translation_control', type=float, default=0.4,
                        help='Control the translation of shapes (0.0 to 1.0)')
    parser.add_argument('--rotation_seed', type=int, default=None,
                        help='Seed for controlling the rotation of shapes')
    parser.add_argument('--sub_obj_nums', type=str, default='1,2,3,4,5,6,7,8,9', help='Possible numbers of sub-objects')
    parser.add_argument('--sub_obj_num_poss', type=str, default='1,2,3,7,10,7,3,2,1', help='Corresponding probabilities for numbers of sub-objects')
    parser.add_argument('--cand_shapes', type=str, default='0,1,2,3', help='Candidate shapes to select from (now includes 3 for ShapeGen)')
    parser.add_argument('--translation_seed', type=int, default=None,
                        help='Seed for controlling the translation of shapes')
    parser.add_argument('--size_seed', type=int, default=None,
                        help='Seed for controlling the size of shapes')
    parser.add_argument('--shapegen_min_extrusions', type=int, default=3,
                        help='Minimum number of extrusions for ShapeGen shapes')
    parser.add_argument('--shapegen_max_extrusions', type=int, default=5,
                        help='Maximum number of extrusions for ShapeGen shapes')
    parser.add_argument('--shapegen_subdivisions', type=int, default=0,
                        help='Number of subdivisions for ShapeGen shapes (0-5)')

    parser.add_argument('--enable_placement_diversity', action='store_true',
                        help='Enable placement diversity checking within warp')
    parser.add_argument('--min_centroid_distance', type=float, default=0.3,
                        help='Minimum distance between corresponding centroids')
    parser.add_argument('--max_placement_attempts', type=int, default=10,
                        help='Maximum attempts to generate diverse placement')
    parser.add_argument('--warp_id', type=str, default='',
                        help='Unique identifier for current warp')
    
    args = parser.parse_args()

    seed_everything(args.seed)

    # Load the shape generator addon
    addon_loaded = load_shape_generator_addon()

    args.sub_obj_nums = [int(x) for x in args.sub_obj_nums.split(',')]
    args.sub_obj_num_poss = [int(x) for x in args.sub_obj_num_poss.split(',')]
    args.cand_shapes = [int(x) for x in args.cand_shapes.split(',')]

    if args.shape_counts:
        shape_counts = {}
        shape_counts_list = args.shape_counts.split(',')
        for item in shape_counts_list:
            shape_name, count = item.split(':')
            count = int(count)

            if shape_name.lower() == 'ellipsoid':
                shape_type = 0
            elif shape_name.lower() == 'cube':
                shape_type = 1
            elif shape_name.lower() == 'cylinder':
                shape_type = 2
            elif shape_name.lower() == 'shapegen':
                shape_type = 3
            else:
                raise ValueError(f"Unknown shape name: {shape_name}")
            shape_counts[shape_type] = count
    else:
        shape_counts = None

    if args.texture_folder:
        texture_images = find_texture_images(args.texture_folder)
        if texture_images:
            texture_rng = np.random.RandomState(args.texture_seed)
        else:
            texture_images = None
            texture_rng = None
    else:
        texture_images = None
        texture_rng = None

    if args.rotation_seed is not None:
        rotation_rng = np.random.RandomState(args.rotation_seed)
    else:
        rotation_rng = np.random
        
    if args.translation_seed is not None:
        translation_rng = np.random.RandomState(args.translation_seed)
    else:
        translation_rng = np.random

    if args.size_seed is not None:
        size_rng = np.random.RandomState(args.size_seed)
    else:
        size_rng = np.random  

    start_time = time.time()
    out_dir = args.output_dir
    num_shapes = args.num_shapes

    output_paths, shapes_parameters = createVarObjShapes(
        out_dir,
        range(num_shapes),
        uuid_str=args.uuid_str,
        shape_counts=shape_counts,
        sub_obj_nums=args.sub_obj_nums,
        sub_obj_num_poss=args.sub_obj_num_poss,
        candShapes=args.cand_shapes,
        bMultiObj=False,
        bPermuteMat=False,
        bScaleMesh=True,
        bMaxDimRange=[0.3, 0.45],
        smooth_probability=args.smooth_probability,
        no_hf=args.no_hf,
        texture_images=texture_images,
        texture_rng=texture_rng,
        translation_control=args.translation_control,  
        rotation_rng=rotation_rng,
        translation_rng=translation_rng,
        size_rng=size_rng,
        shapegen_extrusions=(args.shapegen_min_extrusions, args.shapegen_max_extrusions),
        shapegen_subdivisions=args.shapegen_subdivisions,
        enable_placement_diversity=args.enable_placement_diversity,
        min_centroid_distance=args.min_centroid_distance,
        max_placement_attempts=args.max_placement_attempts,
        warp_id=args.warp_id
    )
    shape_generation_time = time.time()
    print('Saved shapes to', out_dir)
    print(f'TIME - merged_shape_generation.py: shape_generation_time: {shape_generation_time - start_time:.2f}s')