# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path
import time
import numpy as np

from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
import torch


from config import Config
from dataset import dataset_factory
config = Config()

dataset = dataset_factory(config.dataset_settings)
# torch.set_float32_matmul_precision('high')
LightGlue.pruning_keypoint_thresholds['cuda']
# torch.set_grad_enabled(False)
images = Path("/home/aniel/pyslam/LightGlue/assets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
print(device)
extractor = SuperPoint(max_num_keypoints=512,nms_radius=5).eval().to(device)  # load the extractor

torch.cuda.empty_cache()
configs = {
    "LightGlue-full": {
        "depth_confidence": -1,
        "width_confidence": -1,
    },
    "LightGlue-adaptive": {},  # No specific settings, defaults will be used
}
conf = configs["LightGlue-adaptive"]
matcher = LightGlue(features="superpoint",n_layers=2).eval().to(device)
# matcher.pruning_keypoint_thresholds = {
#                 k: -1 for k in matcher.pruning_keypoint_thresholds
#             }
# import torch._dynamo

# torch._dynamo.reset() 
# matcher.compile(mode='reduce-overhead')
def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    
    return torch.tensor(image / 255.0, dtype=torch.float,device="cuda")


#image0 = load_image(images / "DSC_0411.JPG",resize=[160, 120])
# image1 = load_image(images / "DSC_0410.JPG",resize=[160, 120])
# time0 = time.time()
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))

# print((1/(time.time()-time0)))


# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension

# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# print(kpts0)
import cv2

# Set up the video capture
video_path = '/home/aniel/pyslam/thirdparty/superpoint/assets/nicu3.mp4'
cap = cv2.VideoCapture(video_path)
ret, previous_frame = cap.read()
previous_frame = cv2.resize(previous_frame, (1226,370))
previous_features = extractor.extract(numpy_image_to_torch(previous_frame),resize=None)
print(previous_frame.shape)
import time
img_id=0
while dataset.isOk():

    current_frame = dataset.getImage(img_id)
    time0=time.time()
    # ret, current_frame = cap.read()
    # if not ret:
    #     break
    current_frame = cv2.resize(current_frame, (1226,370))
    torch.cuda.synchronize()
    with torch.inference_mode():
    
        current_features = extractor.extract(numpy_image_to_torch(current_frame),resize=None)
        
        matches = matcher({"image0": previous_features, "image1": current_features})
    torch.cuda.synchronize()
    #print(matches)
    feats0, feats1, matches = [
        rbd(x) for x in [previous_features, current_features, matches]
    ]  # remove batch dimension

    kpts0, kpts1, matchess = feats0["keypoints"], feats1["keypoints"], matches["matches"]
    m_kpts0, m_kpts1 = kpts0[matchess[..., 0]], kpts1[matchess[..., 1]]
    kp=kpts1.cpu().numpy()
    for k in kp:
        #print(k)
        try:
            cv2.circle(current_frame,(int(k[0]),int(k[1])),3,(255),-1)
        except:
            pass
    #print(kp)
    
    # Process matches here (e.g., for tracking, stabilization, etc.)
    
    
    #viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    # Update previous_frame for the next iteration
    img_id +=1
    previous_features = current_features
    cv2.imshow("frame",current_frame)
    cv2.waitKey(2)
    print(1/(time.time()-time0))
cap.release()
