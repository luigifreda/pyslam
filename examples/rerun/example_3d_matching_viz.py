import rerun as rr
import numpy as np

# Helper function to create a 3D plane for an image based on camera parameters
def create_image_plane(image, transformation, plane_size=(1.0, 0.75)):
    # Define the 4 corners of the image plane (before transformation) in 3D
    h, w = plane_size  # Width and height of the plane in world units

    # The corners of the plane (centered around origin before applying transformation)
    corners = np.array([[-w/2, -h/2, 0],  # Bottom-left
                        [ w/2, -h/2, 0],  # Bottom-right
                        [ w/2,  h/2, 0],  # Top-right
                        [-w/2,  h/2, 0],  # Top-left
                        [-w/2, -h/2, 0]]) # Bottom-left

    # Apply transformation (rotation and translation) to the corners
    transformed_corners = np.dot(transformation[:3, :3], corners.T).T + transformation[:3, 3]
    
    return transformed_corners


def log_points(name, points, color=[255, 0, 0], labels=None):
    rr.log(f'{name}', rr.Points3D(points,colors=color,labels=labels))

# Main function to visualize the 3D matching results, including images as 3D planes
def visualize_3d_matching_with_images(img1, img2, keypoints1, keypoints2, points3d, transformation1, transformation2):
    rr.init("3D Matching with Images", spawn=True)
    
    # Log images as textures
    rr.log("image1", rr.Image(img1))
    rr.log("image2", rr.Image(img2))
    
    # Define the size of the image planes in world units (adjust this based on actual scale)
    plane_size = (2.0, 1.5)  # Example size

    # Create 3D planes for the images (using transformations for position and orientation)
    plane_corners_img1 = create_image_plane(img1, transformation1, plane_size)
    plane_corners_img2 = create_image_plane(img2, transformation2, plane_size)

    # Log the corners of the image planes as points
    log_points("image1/plane_corners", plane_corners_img1, color=[255, 0, 0], labels=["BL", "BR", "TR", "TL"])
    log_points("image2/plane_corners", plane_corners_img2, color=[0, 255, 0], labels=["BL", "BR", "TR", "TL"])

    # Draw lines between the corners to form the plane (image1)
    #rr.log_line_strip("image1/plane", plane_corners_img1[[0, 1, 2, 3, 0]], color=[255, 0, 0])
    rr.log(f'image1/plane', rr.LineStrips3D([plane_corners_img1], colors=[255, 0, 0]))

    # Draw lines between the corners to form the plane (image2)
    #rr.log_line_strip("image2/plane", plane_corners_img2[[0, 1, 2, 3, 0]], color=[0, 255, 0])
    rr.log(f'image2/plane', rr.LineStrips3D([plane_corners_img2], colors=[0, 255, 0]))

    # Log keypoints and 3D matching lines
    for i, (kp1, kp2) in enumerate(zip(keypoints1, keypoints2)):
        rr.log(f"image1/keypoint_{i}", rr.Points2D(kp1, labels=f"kp{i}_img1", colors=[255, 0, 0]))
        rr.log(f"image2/keypoint_{i}", rr.Points2D(kp2, labels=f"kp{i}_img2", colors=[0, 255, 0]))

    # Apply the transformation to the 3D points
    points3d_transformed = np.dot(transformation2[:3, :3], points3d.T).T + transformation2[:3, 3]

    # Log original and transformed 3D points
    for i, (point_orig, point_trans) in enumerate(zip(points3d, points3d_transformed)):
        rr.log(f"points_3d/original_{i}", rr.Points3D(point_orig, labels=f"3D_pt{i}_orig", colors=[0, 0, 255]))
        rr.log(f"points_3d/transformed_{i}", rr.Points3D(point_trans, labels=f"3D_pt{i}_trans", colors=[255, 255, 0]))

    # Draw lines connecting the original and transformed 3D points
    for i, (point_orig, point_trans) in enumerate(zip(points3d, points3d_transformed)):
        #rr.log_line_strip(f"points_3d/line_{i}", [point_orig, point_trans], color=[255, 0, 255])
        rr.log(f"points_3d/line_{i}", rr.LineStrips3D([point_orig, point_trans],colors=[255, 0, 255]))

# Dummy data for the example
img1 = np.zeros((480, 640, 3), dtype=np.uint8)  # Example blank image
img2 = np.zeros((480, 640, 3), dtype=np.uint8)  # Example blank image

keypoints1 = np.array([[100, 150], [200, 250], [300, 350]])  # 2D keypoints in img1
keypoints2 = np.array([[105, 155], [205, 255], [310, 355]])  # 2D keypoints in img2
points3d = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])       # Corresponding 3D points

# Transformation between img1 and img2 (example: 4x4 identity matrix for img1 and img2)
transformation1 = np.eye(4)  # Identity (img1 is at origin)
transformation2 = np.eye(4)  # Identity (img2 at origin, you can modify this)

# Call the visualization function
visualize_3d_matching_with_images(img1, img2, keypoints1, keypoints2, points3d, transformation1, transformation2)