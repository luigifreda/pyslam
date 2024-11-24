import numpy as np

import sys 
sys.path.append("../../")

from utils_geom import qvec2rotmat, rotmat2qvec



if __name__ == "__main__":
    # Test quaternion
    qvec = [0.1, 0.2, 0.3, 0.9]
    qvec = qvec / np.linalg.norm(qvec)  # Ensure it's normalized

    # Convert quaternion to rotation matrix and back
    R = qvec2rotmat(qvec)
    qvec_recovered = rotmat2qvec(R)

    # Check consistency
    print(f'Original quaternion: {qvec}')
    print(f'Rotation matrix:\n{R}')
    RxRT = np.dot(R, R.T)
    print(f'R * R.T:\n{RxRT}, squared norm: {np.linalg.norm(RxRT)**2}, det: {np.linalg.det(RxRT)}')
    print(f'Recovered quaternion:" {qvec_recovered}')
    
    print(f'Error % in recovered quaternion: {100 * np.linalg.norm(qvec - qvec_recovered)/np.linalg.norm(qvec)}')