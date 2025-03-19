import numpy as np
import trimesh
from pyrender import Scene, Mesh, PerspectiveCamera, OffscreenRenderer
from sklearn.neighbors import KDTree


def get_camera_pose(room_size):
    room = trimesh.creation.box(extents=[room_size, room_size, room_size])
    scene = trimesh.Scene(room)
    camera_pose = scene.camera.look_at(points=room.vertices)
    return camera_pose


def init_renderer(resolution):
    # Render 3d object
    width = resolution[0]
    height = resolution[1]
    r = OffscreenRenderer(viewport_width=width, viewport_height=height)
    return r


def render_mesh(mesh: trimesh.Trimesh, camera_pose, resolution=512, ray_tracing=True, renderer=None):
    if ray_tracing:
        rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        depth, tri_img, index_tri, sign, p_image = trimesh_ray_tracing(
            mesh, camera_pose, resolution=resolution, rayintersector=rayintersector
        )
        mask = p_image[..., 0] == np.inf
        depth = 255 - depth
        depth[mask] = 0

        return depth, p_image
    else:
        renderer = init_renderer((resolution, resolution)) if renderer is None else renderer
        mesh_py = Mesh.from_trimesh(mesh, smooth=False)
        scene = Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
        scene.add(mesh_py)
        # camera = OrthographicCamera(xmag=1, ymag=1)
        camera = PerspectiveCamera(yfov=np.pi / 3.0)

        scene.add(camera, pose=camera_pose)
        color, depth = renderer.render(scene)  # SKIP_CULL_FACES = 1024
        return depth, color


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):
    # this is done to correct the mistake in way trimesh raycasting works.
    # in general this cannot be done.
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [resolution, resolution]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov, fov

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # for each hit, find the distance along its vector
    index_tri, index_ray, points = rayintersector.intersects_id(origins, vectors, multiple_hits=False, return_locations=True)
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(mesh.face_normals[index_tri], vectors[index_ray])

    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]
    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = (
        np.ones(
            [scene.camera.resolution[0], scene.camera.resolution[1], 3],
            dtype=np.float32,
        )
        * np.inf
    )
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)

    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    # show the resulting image
    return a, b, index_tri, sign, p_image


def find_match(source, target, k=1):
    tree = KDTree(source)
    d, indices = tree.query(target, k=k)
    return d[:, 0], indices[:, 0]


def find_correspondence_bw_images(triangle_ids1, triangle_ids2, thresh, return_outside=False):
    """Find correspondence between two images based on the cartesian coordinates"""
    if len(triangle_ids1.shape) == 2:
        triangle_ids1 = np.expand_dims(triangle_ids1, 2)
        triangle_ids2 = np.expand_dims(triangle_ids2, 2)

    # x_1, y_1 = np.where(triangle_ids1[:, :, 0] > -1)
    x_1, y_1 = np.where(triangle_ids1[:, :, 0] != np.inf)
    triangle_index_1 = triangle_ids1[x_1, y_1]
    # x_2, y_2 = np.where(triangle_ids2[:, :, 0] > -1)
    x_2, y_2 = np.where(triangle_ids2[:, :, 0] != np.inf)
    triangle_index_2 = triangle_ids2[x_2, y_2]
    d, indices = find_match(triangle_index_1, triangle_index_2)
    matched_indices_2 = np.where(d < thresh)[0]

    matched_indices_1 = indices[matched_indices_2]

    matched_x_2 = x_2[matched_indices_2]
    matched_y_2 = y_2[matched_indices_2]

    matched_x_1 = x_1[matched_indices_1]
    matched_y_1 = y_1[matched_indices_1]

    return (
        matched_x_1,
        matched_y_1,
        matched_x_2,
        matched_y_2,
        triangle_ids1[matched_x_1, matched_y_1][:, 0],
        triangle_ids2[matched_x_2, matched_y_2][:, 0],
    )
