import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from glob import glob
from depth_anything_v2 import dpt
from SuperGlue.models.matching import Matching
from dataclasses import dataclass
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly 
import os
import pandas as pd

plotly.offline.init_notebook_mode(connected=True)
RESOLUTION : int = 518 * 518

# OUTLINE: 
#   RUNTIME INIT UTILS
#   HELPERS FOR PREPROCESSORS
#   MAIN PREPROCESSORS
#   VISUALIZATION

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# ---------------------------------- RUNTIME INIT UTILS -----------------------------------

def get_models():
    """
    Initializes and returns the depth estimation model and feature matching model.

    This function sets up the depth estimation model using the DepthAnythingV2 architecture
    and loads the pre-trained weights based on the specified encoder and dataset. It also
    configures and initializes the SuperGlue feature matching model.

    Returns:
        tuple: A tuple containing:
            - model (DepthAnythingV2): The depth estimation model.
            - matching (Matching): The feature matching model.
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vits' 
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = dpt.DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'./models/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    matching = Matching(config).eval().to(torch.device('cpu'))

    return model, matching

@dataclass(frozen=True)
class iPhone15ProMainCamera():
    n: float = 4284 # px
    center: float = 4284 / 2 # idx
    focal_length: float = 24 # mm
    sensor_size: float = 1/1.28 # in
    camera_dpi: int = n / sensor_size # px / in
    focal_length_px: float = focal_length * camera_dpi / 25.4 #px

# ---------------------------------- RUNTIME INIT UTILS -----------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# ------------------------------- HELPERS FOR PREPROCESSOR --------------------------------


def get_rays(N=518, camera_params=iPhone15ProMainCamera()):
    """
    Generates a grid of rays emanating from the camera's viewpoint.

    The function creates a square grid of rays based on the specified resolution N,
    scaled according to the camera parameters. Each ray is normalized to unit length.

    Parameters:
    - N (int): The resolution of the ray grid. Determines how many rays are generated.
    - camera_params (iPhoneCamera): An instance of the iPhoneCamera class containing
      camera specifications.

    Returns:
    - np.ndarray: A 2D array of shape (N*N, 3), where each row represents a normalized ray
      vector emanating from the camera's viewpoint.
    """
    scale = N / camera_params.n 
    C = camera_params.center * scale
    f = camera_params.focal_length_px * scale

    rays = [np.array([(x - C), (y - C), f]) for y in range(0,N) for x in range(0,N)]
    rays = [ray / np.linalg.norm(ray) for ray in rays]
    return np.array(rays)

def get_colors(img, N=518):
    """
    Extracts the RGB color values from an image and flattens them into a 2D array.

    This function reads the RGB color values from an image and flattens them into
    a 2D array of shape (N^2, 3), where each row represents an RGB color value. 
    Each value will map 1-1 to the corresponding ray vector in the ray grid.

    Parameters:
    - img (np.ndarray): The input image.
    - N (int): The resolution of the image. Defaults to 518.

    Returns:
    - np.ndarray: A 2D array of shape (N^2, 3) containing RGB color values.
    """
    return np.array([img[y][x] for y in range(N) for x in range(N)])

def rays_to_angles(rays):
    """
    Converts a set of ray vectors into a 2xN^2 numpy array containing azimuth and elevation angles.

    This function calculates the azimuth and elevation angles for each ray vector
    based on their components. The azimuth angle is calculated as the arctan of the
    x-component over the z-component. The elevation angle is calculated as the arcsin
    of the y-component. The results are stored in a 2xN^2 numpy array where the first row
    contains all azimuth angles and the second row contains all elevation angles.

    Parameters:
    - rays (np.ndarray): A 2D array of shape (M, 3), where M is the number of rays,
      and each row represents a ray vector.

    Returns:
    - np.ndarray: A 2xN^2 numpy array where the first row contains azimuth angles,
      and the second row contains elevation angles, both in degrees.
    """
    azimuth = np.arctan2(rays[:,0], rays[:,2]) * (180 / np.pi)
    elevation = np.arcsin(rays[:,1]) * (180 / np.pi)
    return np.array([azimuth, elevation])

def img(path: str):
    """
    Reads an image from the given path, converts it to RGB, and resizes it to 518x518.

    Args:
        path (str): The path to the image file.

    Returns:
        np.ndarray: The processed image.
    """
    image = cv2.imread(path)  # H, W, C
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (518,518), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def get_images(prefix: str):
    """
    Retrieves and processes all images with the given prefix from the './Images/' directory.
    All images are converted to RGB, and resized to 518x518.

    Args:
        prefix (str): The prefix of the image filenames.

    Returns:
        list: A list of processed images.
    """
    return [img(f) for f in sorted(glob(f"./images/{prefix}_*.jpg"))]

def preprocess_image_depth(image: np.ndarray):
    """
    Normalizes and transposes the given image for depth processing.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The preprocessed image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image-mean)/std
    image = np.transpose(image, (2,0,1)) # C, H, W
    return image

def preprocess_image_matcher(image: np.ndarray) -> torch.Tensor:
    """
    Converts the given image to grayscale, normalizes it, and converts it to a PyTorch tensor.

    Args:
        image (np.ndarray): The input image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    tensor = torch.from_numpy(image).float().to(torch.device('cpu')).unsqueeze(0).unsqueeze(0)
    return tensor

def postprocess_matches(pred: dict, num_matches: int = -1):
    """
    Post-processes the predicted matches to filter and sort them by confidence.

    Args:
        pred (dict): The dictionary containing prediction results.
        num_matches (int, optional): The number of matches to return. Defaults to -1 (all matches).

    Returns:
        tuple: Filtered and sorted keypoints and matching scores.
    """
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}    
    kpts0, kpts1, matches, conf = pred['keypoints0'], pred['keypoints1'], pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    idx = [x for x in range(len(mconf))]
    idx = sorted(idx, key=lambda x: mconf[x])[::-1][:num_matches]
    mkpts0 = np.array(mkpts0)[idx]
    mkpts1 = np.array(mkpts1)[idx]
    mconf = np.array(mconf)[idx]
    return mkpts0, mkpts1, mconf

def project_point(_x, _y, depth, N=518, camera_params=iPhone15ProMainCamera()):
    """
    Projects a 2D point into 3D space using depth information and camera parameters.

    Args:
        _x (int): The x-coordinate of the point.
        _y (int): The y-coordinate of the point.
        depth (np.ndarray): The depth map.
        N (int, optional): The scaling factor. Defaults to 518.
        camera_params (iPhoneCamera, optional): The camera parameters. Defaults to iPhoneCamera().

    Returns:
        tuple: The 3D coordinates (x, y, z) of the projected point.
    """
    scale = N / camera_params.n 
    C = camera_params.center * scale
    f = camera_params.focal_length_px * scale

    z = depth[int(_y)][int(_x)]
    x = -((_x - C) * z / f)
    y = -((_y - C) * z / f)
    return x,y,z 

def get_depth(image: np.ndarray, model: torch.nn.Module):
    input_image = torch.from_numpy(image).unsqueeze(0).to(torch.float32).to(torch.device("cpu"))
    with torch.no_grad():
        depth = model.forward(input_image).squeeze()
        return depth
    
def get_matches(inputs: list[np.ndarray], model: Matching, num_matches: int = 30):
    matches = []
    for i in range(len(inputs) - 1):
        with torch.no_grad():
            pred = model({'image0': inputs[i], 'image1': inputs[i+1]})
            matches.append(postprocess_matches(pred=pred, num_matches=num_matches))
    return matches

def project_matches_to_3d(matches, depths):
    """
    Projects 2D matches to 3D coordinates using depth information.

    Args:
        matches (list of tuples): A list of tuples where each tuple contains two sets of points and a match score.
        depths (list of float): A list of depth values corresponding to each match.

    Returns:
        list of tuples: A list of tuples where each tuple contains two sets of 3D points.
    """
    matches_3d = []
    for i, (pts0, pts1, _) in enumerate(matches):
        pts0_3d = np.array([project_point(a,b, depths[i]) for a,b in pts0])
        pts1_3d = np.array([project_point(a,b, depths[i+1]) for a,b in pts1])
        matches_3d.append((pts0_3d, pts1_3d))
    return matches_3d

def get_transform(_pts0, _pts1):
    """
    Computes the rotation matrix and translation vector that aligns two sets of 3D points.

    Args:
        _pts0 (numpy.ndarray): The first set of 3D points.
        _pts1 (numpy.ndarray): The second set of 3D points.

    Returns:
        tuple: A tuple containing the rotation matrix (numpy.ndarray) and the translation vector (numpy.ndarray).
    """
    centroid0 = np.mean(_pts0, axis=0)
    centroid1 = np.mean(_pts1, axis=0)
    
    pts0 = _pts0 - centroid0
    pts1 = _pts1 - centroid1
    H = np.dot(pts0.T, pts1)

    U, S, Vt = np.linalg.svd(H)

    R : np.ndarray = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t : np.ndarray = np.dot(centroid0, R.T) - centroid1
    return R, t

def decompose_rotation_matrix(R):
    """
    Decomposes a rotation matrix into Euler angles (in degrees).

    Args:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        numpy.ndarray: A 1D array containing the Euler angles (in degrees) corresponding to the rotation matrix.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([
        np.degrees(x), 
        np.degrees(y), # elevation
        np.degrees(z)  # azimuth
        ])

def compute_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix that rotates vector v1 to vector v2.

    Args:
        v1 (np.ndarray): A 3D direction vector.
        v2 (np.ndarray): A 3D direction vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    if np.allclose(v1, v2):
        R = np.eye(3)
        print(f"Rotation: \n{R}")
        return R

    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)

    # skew-symmetric cross-product matrix
    K = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])

    # Rodrigues' rotation formula
    I = np.eye(3)
    R = I + K + K @ K * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2))
    print(f"Rotation: \n{R}")

    return R

def compute_positions(transforms: list[tuple[np.ndarray, np.ndarray]]):
    positions = [np.array([0,0,0])]
    for t in transforms:
        positions.append(positions[-1] + t[1])
    return np.array(positions)

def compute_ray_directions(transforms: list[tuple[np.ndarray, np.ndarray]]):
    ray_direction_vectors = [get_rays()]
    for t in transforms:
        ray_direction_vectors.append(np.dot(ray_direction_vectors[-1], t[0].T))
    return np.array(ray_direction_vectors)

# ------------------------------- HELPERS FOR PREPROCESSOR --------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# ----------------------------------- MAIN PREPROCESSORS -----------------------------------

def training_preprocessor(image_prefix: str, model: torch.nn.Module, matching: Matching, force_reload=False):
    data_path = f'./data/{image_prefix}.pt'
    if os.path.exists(data_path) and not force_reload:
        return torch.load(data_path)

    images = get_images(image_prefix)
    match_inputs = [preprocess_image_matcher(x) for x in images]
    depth_inputs = [preprocess_image_depth(x) for x in images]
    depths = [get_depth(image=x, model=model) for x in depth_inputs]
    matches = get_matches(inputs=match_inputs, model=matching, num_matches=-1)
    matches_3d = project_matches_to_3d(matches, depths)
    transforms = [get_transform(m[0], m[1]) for m in matches_3d]
    positions = compute_positions(transforms)
    ray_direction_vectors = compute_ray_directions(transforms)
    colors = np.array([get_colors(x) for x in images])

    num_images, _, _ = colors.shape
    colors = colors.reshape(num_images * RESOLUTION, 3)
    ray_direction_vectors = ray_direction_vectors.reshape(num_images * RESOLUTION, 3)
    positions = np.repeat(positions, RESOLUTION, axis=0)
    data = np.concatenate((positions, ray_direction_vectors, colors), axis=1)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    torch.save(data_tensor, data_path)
    return data_tensor

def inference_preprocessor(position: np.ndarray, direction: np.ndarray):
    R = compute_rotation_matrix(np.array([0,0,1]), direction)
    rays = np.dot(get_rays(), R.T)
    positions = np.repeat(np.array([position]), RESOLUTION, axis=0)
    data = np.concatenate((positions, rays), axis=1)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor

# ----------------------------------- MAIN PREPROCESSORS -----------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# ------------------------------------- VISUALIZATION -------------------------------------

            
def create_trace(matches, colors, opacity=0.8):
    x = [point[0] for point in matches]
    y = [point[1] for point in matches]
    z = [point[2] for point in matches]

    if isinstance(colors[0], (list, tuple)) and len(colors[0]) == 3:
        marker_color = ['rgb({},{},{})'.format(*color) for color in colors]
    else:
        marker_color = colors
    
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker={
            'size': 1,
            'opacity': opacity,
            'color': marker_color
        }
    )

def depth_to_pcd(depth: torch.Tensor, image: np.ndarray):
    pcd = []
    colors = []
    for _y in range(0,518,2):
        for _x in range(0,518,2):
            pcd.append(np.array(project_point(_x,_y, depth)))
            colors.append(image[_y][_x])
    return np.array(pcd), np.array(colors)

def create_pcd_trace(depth, image):
    points, colors = depth_to_pcd(depth, image)
    trace = create_trace(points, colors)  
    return trace

def create_pcd_fig(idx, depths, images):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

    trace1 = create_pcd_trace(depths[idx], images[idx])
    trace2 = create_pcd_trace(depths[idx+1], images[idx+1])

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    fig.update_layout(height=500, width=1200)
    return fig

def plot_matches(idx, images, matches):
    imgs = (images[idx], images[idx+1])
    mkpts0 = matches[idx][0]
    mkpts1 = matches[idx][1]
    n = len(mkpts0)    
    color = matplotlib.cm.gist_rainbow(np.linspace(start=0, stop=1.0, num=n))
    np.random.shuffle(color)

    linecolor = [(r, g, b, 0.5) for r, g, b, _ in color]

    fig, ax = plt.subplots(1, 2, figsize=(8,4), dpi=100)
    for i in range(2):
        ax[i].imshow(imgs[i])
        ax[i].set_aspect('equal')
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])

    plt.tight_layout()

    norm = fig.transFigure.inverted()
    fig_coords_mkpts0 = norm.transform(ax[0].transData.transform(mkpts0))
    fig_coords_mkpts1 = norm.transform(ax[1].transData.transform(mkpts1))

    fig.lines.clear()

    for i in range(len(fig_coords_mkpts0)):
        line = matplotlib.lines.Line2D(
            xdata=[fig_coords_mkpts0[i, 0], fig_coords_mkpts1[i, 0]], 
            ydata=[fig_coords_mkpts0[i, 1], fig_coords_mkpts1[i, 1]],  
            transform=fig.transFigure,  
            color=linecolor[i],  
            linewidth=1
        )
        fig.lines.append(line)

    ax[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=6)
    ax[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=6)
    return fig

def visualize_rays(data_tensor):
    positions = data_tensor[:, :3]
    ray_directions = data_tensor[:, 3:6]

    colors = list(mcolors.CSS4_COLORS.values())[:len(positions)]

    fig = go.Figure()
    for i in range(0, len(positions), 400):
        position = positions[i]
        ray_direction = ray_directions[i]
        color = colors[i]

        trace = go.Scatter3d(
            x=[position[0], position[0] + ray_direction[0]],
            y=[position[1], position[1] + ray_direction[1]],
            z=[position[2], position[2] + ray_direction[2]],
            mode='lines',
            line=dict(color=color, width=1)
        )

        fig.add_trace(trace)

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

def visualize_camera_poses(data):
    """
        Visualizes camera poses based on the provided data.

        Parameters:
        data (list): The output of the training_preprocessor function.

        Returns:
        None
    """
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'xd', 'yd', 'zd', 'r', 'g', 'b'])
    df['pos'] = list(zip(df['x'], df['y'], df['z']))
    df['dir'] = list(zip(df['xd'], df['yd'], df['zd']))
    df['rgb'] = list(zip(df['r'], df['g'], df['b']))
    df = df[['pos', 'dir', 'rgb']]

    fig = go.Figure()
    positions = df['pos'].unique()
    colors = list(mcolors.CSS4_COLORS.values())[65:][:len(positions)]
    for i, pos in enumerate(positions):
        for dir in df[df['pos'] == pos]['dir'][::2000]:
            try:
                trace = go.Scatter3d(
                    x=[pos[0], pos[0] + dir[0]],
                    y=[pos[1], pos[1] + dir[1]],
                    z=[pos[2], pos[2] + dir[2]],
                    mode='lines',
                    line=dict(color=colors[i], width=1)
                )

                fig.add_trace(trace)
            except:
                print(pos, dir)
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='white', showbackground=False, color='white'),
            yaxis=dict(gridcolor='white', showbackground=False, color='white'),
            zaxis=dict(gridcolor='white', showbackground=False, color='white')
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        width=800,
        height=800
    )

    fig.show()