import unittest
import numpy as np
import cv2

import sys 
sys.path.append("../../")
import pyslam.config as config


from pyslam.utilities.utils_dust3r import Dust3rImagePreprocessor

    
# Unit test class
class TestDust3rImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        # Create a dummy image: 330x310 with 3 channels (simulate a color image)
        self.dummy_img1 = np.full((330, 310, 3), 100, dtype=np.uint8)
        # Create another dummy image: 400x400 (square image)
        self.dummy_img2 = np.full((400, 400, 3), 150, dtype=np.uint8)
        self.imgs = [self.dummy_img1, self.dummy_img2]
        

    def test_preprocess_images_224(self):
        proc = Dust3rImagePreprocessor(inference_size=224, verbose=self.verbose)
        processed = proc.preprocess_images(self.imgs)

        # Check that metadata was stored for each image
        self.assertEqual(len(proc.metadata), 2)
        # Check that processed images have the correct shape (square, 224x224 or less depending on crop)
        for p in processed:
            # p['img'] shape is (1, H, W, C) because we added a new axis.
            img_processed = p['img'][0]
            _, H, W = img_processed.shape[:3]
            self.assertEqual(H, W)
            self.assertTrue(H <= 224)
            
    def test_preprocess_images_512(self):
        proc = Dust3rImagePreprocessor(inference_size=512, verbose=self.verbose)
        processed = proc.preprocess_images(self.imgs)

        # Check that metadata was stored for each image
        self.assertEqual(len(proc.metadata), 2)
        # Check that processed images have the correct shape (square, 512x512 or less depending on crop)
        for p in processed:
            # p['img'] shape is (1, H, W, C) because we added a new axis.
            img_processed = p['img'][0]
            _, H, W = img_processed.shape[:3]
            #self.assertEqual(H, W)
            self.assertTrue(W <= 512)
            self.assertTrue(H <= 512)

    def test_scale_down_keypoints(self):
        proc = Dust3rImagePreprocessor(inference_size=224, verbose=self.verbose)
        # Preprocess one image
        proc.preprocess_images([self.dummy_img1])

        # Assume a keypoint in the original image at (100, 150)
        original_kps = [(100, 150)]
        # Transform to processed image coordinates
        scaled_kps = proc.scale_down_keypoints(original_kps, image_idx=0)

        # Now invert the transformation using our stored metadata:
        meta = proc.metadata[0]
        orig_H, orig_W = meta['original_shape']
        resized_H, resized_W = meta['resized_shape']
        crop_x, crop_y = meta['crop_offset']

        # Compute expected scaling
        scale_x = resized_W / orig_W
        scale_y = resized_H / orig_H
        expected = np.array([(100 * scale_x - crop_x, 150 * scale_y - crop_y)])

        # Allow a small tolerance due to rounding
        np.testing.assert_allclose(scaled_kps, expected, rtol=1e-2)

    def test_rescale_keypoints(self):
        proc = Dust3rImagePreprocessor(inference_size=224, verbose=self.verbose)
        proc.preprocess_images([self.dummy_img1])

        # Let’s simulate keypoints from the processed (cropped) image.
        # For instance, center of the processed image.
        processed_meta = proc.metadata[0]
        cropped_shape = processed_meta['cropped_shape']
        kp_proc = [(cropped_shape[1] / 2, cropped_shape[0] / 2)]
        
        # Now convert these keypoints back to the original image coordinates.
        keypoints_raw = proc.rescale_keypoints(kp_proc, image_idx=0)
        
        # For a symmetric center crop, the center should remain roughly the same.
        orig_shape = processed_meta['original_shape']
        expected_center = (orig_shape[1] / 2, orig_shape[0] / 2)
        
        np.testing.assert_allclose(keypoints_raw[0], expected_center, rtol=0.1)
        
    def test_scale_down_rescale_keypoints(self):
        proc = Dust3rImagePreprocessor(inference_size=224, verbose=self.verbose)
        proc.preprocess_images([self.dummy_img1])
        H,W = self.dummy_img1.shape[:2]
        # Generate a list of random points in the original image
        num_points = 100
        list_original_kps = np.random.rand(num_points, 2) * np.array([W, H])
        original_kps = list_original_kps.tolist()
        # Transform to processed image coordinates
        scaled_kps = proc.scale_down_keypoints(original_kps, image_idx=0)
        # Now invert the transformation using our stored metadata:        
        keypoints_raw = proc.rescale_keypoints(scaled_kps, image_idx=0)
        # Check the full list of keypoints
        np.testing.assert_allclose(keypoints_raw, original_kps, rtol=0.1)

if __name__ == '__main__':
    unittest.main()
