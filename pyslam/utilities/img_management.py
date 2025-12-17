"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2
import math

from .geometry import add_ones, homography_matrix
from .drawing import draw_random_img

import traceback


# combine two images horizontally
def combine_images_horizontally(img1, img2):
    if img1.ndim <= 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if img2.ndim <= 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img3[:h1, :w1, :3] = img1
    img3[:h2, w1 : w1 + w2, :3] = img2
    return img3


# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks
def img_blocks(img, row_divs, col_divs):
    rows, cols = img.shape[:2]
    # print('img.shape: ', img.shape)
    xs = np.uint32(
        np.rint(np.linspace(0, cols, num=col_divs + 1))
    )  # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs + 1)))
    # print('img_blocks xs: ', xs)
    # print('img_blocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], y1, x1  # return block, row, col


def mask_block(mask, x1, x2, y1, y2):
    if mask is None:
        return None
    else:
        return mask[y1:y2, x1:x2]


# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks
def img_mask_blocks(img, mask, row_divs, col_divs):
    rows, cols = img.shape[:2]
    # print('img.shape: ', img.shape)
    xs = np.uint32(
        np.rint(np.linspace(0, cols, num=col_divs + 1))
    )  # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs + 1)))
    # print('img_blocks xs: ', xs)
    # print('img_blocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], mask_block(
                mask, x1, x2, y1, y2
            ), y1, x1  # return block, row, col


# Pad an image
def pad_img(img: np.ndarray, padding: int, color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Pad an image with 'padding' along each side (height and width)
    and fill the padding with 'color'.

    Parameters:
    - img:  Image of shape [H, W, C=3] with channels as RGB (same
            as the 'color' channels)
    - padding:      Padding 'P' (int) for each dimension (applied
                    on both ends of axis)
    - color:    The RGB color of the padding

    Returns:
    - _img:     Image of shape [H+2P, W+2P, C=3]
    """
    if type(color) == list:
        color = tuple(color)
    assert len(color) == 3, "Color should be (R, G, B) value"
    color = np.array(color)
    # ret_img = np.pad(img, [(padding, padding), (padding, padding),
    #             (0, 0)], constant_values=[(color, color),
    #                         (color, color), (0, 0)])
    ret_img = np.ones((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding, 3), np.uint8) * color
    ret_img[padding:-padding, padding:-padding] = img
    return ret_img.astype(img.dtype)


# create a generator over an image to produce a pyramid of images in the scale space by using the input scale factor
# N.B: check the newer Pyramid class in pyramid.py!
def pyramid(image, scale=1.2, minSize=(30, 30), gauss_filter=True, sigma0=1.0):
    level = 0
    inv_scale = 1.0 / scale

    # from https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L212
    # \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sigma_nominal = 0.5  # no filtering on the original image from https://www.vlfeat.org/api/sift.html#sift-tech-ss
    # sigma0 = 1.0       # N.B.: SIFT use 1.6 for this value
    sigma_prev = sigma_nominal

    sigma_total = math.pow(scale, level) * sigma0
    print("level %d, sigma_total: %f" % (level, sigma_total))
    sigma_cur = math.sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev)
    sigma_prev = sigma_cur

    if gauss_filter:
        image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma_cur)

    # yield the original image
    yield image, level

    while True:
        level += 1

        sigma_total = math.pow(scale, level) * sigma0
        print("level %d, sigma_total: %f" % (level, sigma_total))
        sigma_cur = math.sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev)
        sigma_prev = sigma_cur

        if gauss_filter:
            blur = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma_cur)
            image = cv2.resize(
                blur, (0, 0), fx=inv_scale, fy=inv_scale
            )  # ,interpolation = cv2.INTER_NEAREST)
        else:
            image = cv2.resize(
                image, (0, 0), fx=inv_scale, fy=inv_scale
            )  # ,interpolation = cv2.INTER_NEAREST)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image, level


# N.B.: if you want the mask indexs, you can return  mask_idxs = (mask.ravel() == 1)
def mask_from_polygon(size, pts):
    pts = pts.astype(np.int32)  # reshape(-1,1,2)
    mask = np.zeros(size[:2], np.uint8)
    mask = cv2.fillConvexPoly(mask, pts, 255)
    return mask


# rotate an image by adjusting the output image size in order to contain the rotated image
# angle in degrees
def rotate_img(img, center=None, angle=0, scale=1):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    img_box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    # print('img_box:',img_box)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # grab sin and cos from matrix
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h * cos))
    # adjust the rotation matrix to take into account translation (in the new image)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_img_box = (M @ add_ones(img_box).T).T
    # print('rotated_img_box:',rotated_img_box)
    img_out = cv2.warpAffine(img, M, (new_w, new_h))
    return img_out, rotated_img_box, M


# transform an image by rotating and translating the camera (camera-x along image-x, camera-y along image-y, camera-z along the optical axis)
# the image is assumed to lie on the plane Z=1 (in front of the camera at distance d=1 along the optical axis);
# we compute the homography induced by the plane Z=1 when the camera is moved from [I|0] to [R|t] (see homography_matrix());
# adjust_frame => adjust the frame or not in order to contain the transformed image, in this case tx,ty are useless
# tx=0.5 correspond to half image width (see homography_matrix());
# angles input are in degrees
def transform_img(img, rotx, roty, rotz, tx=0, ty=0, scale=1, adjust_frame=True):
    roll = rotx * math.pi / 180.0
    pitch = roty * math.pi / 180.0
    yaw = rotz * math.pi / 180.0
    # N.B.: in the computed homography_matrix we set d=1 (see homography_matrix())
    # u=fx*X/Z => on Z=d=1 one has u=fx*X/1
    # if we shift the camera of tz along Z, then one has u'=fx*X/(1-tz)
    # hence we have a zoom_factor = 1/(1-tz) => tz = (zoom_factor - 1)/zoom_factor
    tz = (scale - 1) / scale
    (h, w) = img.shape[:2]
    center = np.float32([w / 2, h / 2, 1])
    img_box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    # print('img_box:',img_box)
    H = homography_matrix(img, roll, pitch, yaw, tx, ty, tz)
    # print('H:',H)
    transformed_img_box = H @ add_ones(img_box).T
    transformed_img_box = (transformed_img_box[:2] / transformed_img_box[2]).T
    transformed_center = (H @ center.T).T
    # print('transformed_img_box:',transformed_img_box)
    if adjust_frame:
        # adjust the frame in order to contain the transformed image
        min_u = math.floor(transformed_img_box[:, 0].min())
        max_u = math.ceil(transformed_img_box[:, 0].max())
        min_v = math.floor(transformed_img_box[:, 1].min())
        max_v = math.ceil(transformed_img_box[:, 1].max())
        new_w = max_u - min_u
        new_h = max_v - min_v
        if H[2, 2] != 0:
            H = H / H[2, 2]
        T = np.array([[1, 0, -min_u], [0, 1, -min_v], [0, 0, 1]])
        H = T @ H
        transformed_img_box = H @ add_ones(img_box).T
        transformed_img_box = (transformed_img_box[:2] / transformed_img_box[2]).T
        transformed_center = (H @ center.T).T
    else:
        # simulate the camera pose change
        new_w = w
        new_h = h
    img_out = cv2.warpPerspective(img, H, (new_w, new_h))
    return img_out, transformed_img_box, H


# add 'disturbing' background on `img` outside the given bounding `img_box`
def add_background(img, img_box, img_background=None):
    if img_background is None:
        # create random image
        img_background = draw_random_img(img.shape)
    else:
        # check if we have to resize img_background
        if img_background.shape != img.shape:
            # print('resizing img background')
            (h, w) = img.shape[:2]
            img_background = cv2.resize(img_background, (w, h))
            # check if we have to convert to gray image
            if img.ndim == 2:
                img_background = cv2.cvtColor(img_background, cv2.COLOR_RGB2GRAY)
        # print('img.shape:',img.shape,', img_background.shape:',img_background.shape)
    mask = mask_from_polygon(img.shape, img_box)
    inverse_mask = cv2.bitwise_not(mask)
    img_background = cv2.bitwise_or(img_background, img_background, mask=inverse_mask)
    # combine foreground+background
    final = cv2.bitwise_or(img, img_background)
    return final


def proc_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


# create a scaled image of uint8 from a image of floats
def img_from_floats(img_flt, img_max=None, img_min=None, eps=1e-9):
    assert img_flt.dtype in [np.float32, np.float64, np.float16, np.double, np.single]
    img_max = np.max(img_flt) if img_max is None else img_max
    img_min = np.min(img_flt) if img_min is None else img_min
    if img_max is not None or img is not None:
        img_flt = np.clip(img_flt, img_min, img_max)
    img_range = img_max - img_min
    if img_range < eps:
        img_range = 1
    img = (img_flt - img_min) / img_range * 255.0
    return img.astype(np.uint8)


# remove borders from img
def remove_borders(image, borders):
    shape = image.shape
    new_im = np.zeros_like(image)
    if len(shape) == 4:
        shape = [shape[1], shape[2], shape[3]]
        new_im[:, borders : shape[0] - borders, borders : shape[1] - borders, :] = image[
            :, borders : shape[0] - borders, borders : shape[1] - borders, :
        ]
    elif len(shape) == 3:
        new_im[borders : shape[0] - borders, borders : shape[1] - borders, :] = image[
            borders : shape[0] - borders, borders : shape[1] - borders, :
        ]
    else:
        new_im[borders : shape[0] - borders, borders : shape[1] - borders] = image[
            borders : shape[0] - borders, borders : shape[1] - borders
        ]
    return new_im


# keep the same shape (same channels) of input image
def get_dark_gray_image(img, dark_factor=0.4):
    res = None
    if img.ndim == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_gray_image = (gray_image * dark_factor).astype(np.uint8)
        res = cv2.merge([dark_gray_image, dark_gray_image, dark_gray_image])
    else:
        gray_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dark_gray_image = (gray_image * dark_factor).astype(np.uint8)
        res = dark_gray_image
    return res


# See https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
def convert_float_to_colored_uint8_image(float_img, color_map=cv2.COLORMAP_AUTUMN):
    # Normalize the float image to [0, 255]
    normalized_gray_image = cv2.normalize(
        float_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    # Convert the image to uint8 (necessary for display)
    uint8_img = np.uint8(normalized_gray_image)
    colored_img = cv2.applyColorMap(uint8_img, color_map)
    return colored_img


# Convert a float value in the range [0, 1] to a color using a colormap.
# see https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
def float_to_color_array(values, colormap=cv2.COLORMAP_AUTUMN):
    values = np.where(values < 0, 0, values)
    values = np.where(values > 1, 1, values)
    # Convert the float value to a 1x1 grayscale image (in range [0, 255])
    gray_values = np.uint8(values * 255.0)
    gray_image = np.array(gray_values, dtype=np.uint8)
    # Apply the colormap to the grayscale image
    colored_image = cv2.applyColorMap(gray_image, colormap)
    return colored_image.reshape(-1, 3)


def float_to_color(value, colormap=cv2.COLORMAP_AUTUMN):
    if not 0 <= value <= 1:
        if value > 1.0001:
            print(f"[float_to_color]: The input value {value} was expected to be between 0 and 1.")
        value = max(0.0, min(value, 1.0))
    # Convert the float value to a 1x1 grayscale image (in range [0, 255])
    gray_value = np.uint8(value * 255)
    gray_image = np.array([gray_value], dtype=np.uint8)
    # Apply the colormap to the grayscale image
    colored_image = cv2.applyColorMap(gray_image, colormap)
    # Return the color in BGR format
    color = tuple(int(c) for c in colored_image[0, 0])
    return color


class ImgWriter:
    kFont = cv2.FONT_HERSHEY_SIMPLEX
    kFontScale = 0.7
    kFontColor = (255, 255, 255)
    kBgColor = (0.2, 0.2, 0.2)
    kFontThickness = 1
    kFontLineType = cv2.LINE_AA

    def __init__(
        self,
        font_scale=kFontScale,
        font_color=kFontColor,
        font_thickness=kFontThickness,
        font_line_type=kFontLineType,
    ):
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.font_thickness_bg = font_thickness + 1
        self.font_line_type = font_line_type

    def write(self, img, text, pos):
        cv2.putText(
            img,
            text,
            pos,
            self.kFont,
            self.font_scale,
            ImgWriter.kBgColor,
            self.font_thickness_bg,
            self.font_line_type,
        )
        cv2.putText(
            img,
            text,
            pos,
            self.kFont,
            self.font_scale,
            self.font_color,
            self.font_thickness,
            self.font_line_type,
        )


# visualize loop closure candidates in a single image
class LoopCandidateImgs:
    def __init__(self):
        self.candidates = None
        self.map_color = {}
        self.current_count = 0
        self.max_count = 0
        self.img_size = None
        self.img_writer = ImgWriter()

    def add(self, img_loop, img_id, score=None):
        font_pos = (50, 50)
        text = f"id: {img_id}" if score is None else f"id: {img_id}, s: {score:.2f}"
        self.img_writer.write(img_loop, text, font_pos)
        if img_loop is not None:
            self.img_size = img_loop.shape
            img_rows = self.img_size[0]
            if self.candidates is None:
                self.candidates = img_loop
            else:
                img_rows = self.img_size[0]
                if self.max_count == 0:
                    self.candidates = img_loop
                elif self.current_count < self.max_count:
                    self.candidates[
                        self.current_count * img_rows : (self.current_count + 1) * img_rows, :
                    ] = img_loop
                else:
                    self.candidates = np.vstack((self.candidates, img_loop))
            self.map_color[self.current_count] = True
            self.current_count += 1
            self.max_count = max(self.max_count, self.current_count)

    def reset(self):
        if self.candidates is not None:
            img_rows = self.img_size[0]
            # make all the old candidates gray
            for i in range(self.max_count):
                if i in self.map_color and self.map_color[i]:
                    temp = self.candidates[i * img_rows : (i + 1) * img_rows, :]
                    self.candidates[i * img_rows : (i + 1) * img_rows, :] = get_dark_gray_image(
                        temp
                    )
                    self.map_color[i] = False
        self.current_count = 0


class ImageTable:
    border_width = 1

    def __init__(self, num_columns: int = 3, resize_scale: float = 1.0):
        """
        Initializes the ImageTable instance.

        Args:
            num_columns (int): Number of columns in the image table.
            resize_scale (float): Scale to resize added images.
        """
        self.num_columns = num_columns
        self.resize_scale = resize_scale
        self.images = []
        self.table_image = None

    def image(self):
        return self.table_image

    def add(self, image: np.ndarray):
        """
        Adds a new image to the table after resizing it.

        Args:
            image (np.ndarray): The image to add (as a NumPy array).
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: ImageTable: Image is not a valid numpy array, skipping.")
            return False

        try:
            # Resize the image
            height, width = image.shape[:2]
            if self.resize_scale != 1.0:
                new_size = (int(width * self.resize_scale), int(height * self.resize_scale))
                if new_size[0] <= 0 or new_size[1] <= 0:
                    print(
                        f"Error: ImageTable: Invalid new size {new_size[0]}x{new_size[1]}, skipping."
                    )
                    return False
                # print(f'ImageTable: Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]}')
                resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_image = image
            self.images.append(resized_image)
            return True
        except Exception as e:
            print(f"Error: ImageTable: Failed to add image: {e}")
            print(traceback.format_exc())
        return False

    def reset(self):
        """
        Resets the image table, clearing all added images.
        """
        self.images = []
        self.table_image = None

    def render(self) -> np.ndarray:
        """
        Renders the table as a single composite image.

        Returns:
            np.ndarray: The composite image.
        """
        if len(self.images) == 0:
            raise ValueError("No images to render.")

        try:

            border_width = self.border_width
            ndim = self.images[0].ndim
            fill_value = [255, 255, 255] if ndim == 3 else 255

            # Add a small border around each image
            bordered_images = [
                cv2.copyMakeBorder(
                    img,
                    border_width,
                    border_width,
                    border_width,
                    border_width,
                    cv2.BORDER_CONSTANT,
                    value=fill_value,
                )
                for img in self.images
            ]

            # Determine the size of each row
            rows = []
            for i in range(0, len(bordered_images), self.num_columns):
                row_images = bordered_images[i : i + self.num_columns]

                # Pad the row with blank images if necessary
                if len(row_images) < self.num_columns:
                    height, width = row_images[0].shape[:2]
                    blank_image = (
                        np.zeros((height, width, ndim), dtype=row_images[0].dtype)
                        if ndim == 3
                        else np.zeros((height, width), dtype=row_images[0].dtype)
                    )
                    blank_image = cv2.copyMakeBorder(
                        blank_image,
                        border_width,
                        border_width,
                        border_width,
                        border_width,
                        cv2.BORDER_CONSTANT,
                        value=fill_value,
                    )
                    row_images.extend([blank_image] * (self.num_columns - len(row_images)))

                # Ensure all images in the row have the same height
                max_height = max(img.shape[0] for img in row_images)
                row_images = [
                    (
                        cv2.copyMakeBorder(
                            img,
                            0,
                            max_height - img.shape[0],
                            0,
                            0,
                            cv2.BORDER_CONSTANT,
                            value=fill_value,
                        )
                        if img.shape[0] < max_height
                        else img
                    )
                    for img in row_images
                ]

                # Concatenate images horizontally to form a row
                ndim = row_images[0].ndim
                for j, col in enumerate(row_images):
                    if col.ndim != ndim:
                        print(f"changing elem {i},{j} ndim from: {col.ndim} to {ndim}")
                        row_images[j] = np.mean(col, axis=2)
                row = np.hstack(row_images)
                rows.append(row)

            # Concatenate rows of different heights vertically to form the final table
            max_height = max(img.shape[0] for img in rows)
            max_width = max(img.shape[1] for img in rows)
            table_image = np.vstack(
                [
                    (
                        cv2.copyMakeBorder(
                            row,
                            0,
                            max_height - row.shape[0],
                            0,
                            max_width - row.shape[1],
                            cv2.BORDER_CONSTANT,
                            value=fill_value,
                        )
                        if row.shape[0] < max_height or row.shape[1] < max_width
                        else row
                    )
                    for row in rows
                ]
            )
            self.table_image = table_image
            return table_image

        except Exception as e:
            print(f"Error: ImageTable: Failed to render table: {e}")
            print(traceback.format_exc())


# Adapted from https://github.com/nburrus/stereodemo/blob/main/stereodemo/utils.py
class ImagePadder:
    def __init__(self, multiple, mode):
        self.multiple = multiple
        self.mode = mode

    def pad_width(self, size: int, multiple: int):
        return 0 if size % multiple == 0 else multiple - (size % multiple)

    def pad(self, im: np.ndarray):
        # H,W,C
        rows = im.shape[0]
        cols = im.shape[1]
        self.rows_to_pad = self.pad_width(rows, self.multiple)
        self.cols_to_pad = self.pad_width(cols, self.multiple)
        if self.rows_to_pad == 0 and self.cols_to_pad == 0:
            return im
        return np.pad(im, ((0, self.rows_to_pad), (0, self.cols_to_pad), (0, 0)), mode=self.mode)

    def unpad(self, im: np.ndarray):
        w = im.shape[1] - self.cols_to_pad
        h = im.shape[0] - self.rows_to_pad
        return im[:h, :w, :]
