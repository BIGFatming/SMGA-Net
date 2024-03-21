import cv2
import torch
import numpy as np
from models.peppa_landmark.torch_detector import Detector
from mtcnn_models.mtcnn import MTCNN
import random


def mosaic(img):
    img_copy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    mosaic_area_max = int(height * width)
    mosaic_area_min = int((height * width) / 8)
    x_0 = random.randint(0, int(width / 2))
    x_1 = random.randint(x_0 + int(width / 16), width)
    mosaic_width = x_1 - x_0
    height_min = int(mosaic_area_min / mosaic_width)
    height_max = int(mosaic_area_max / mosaic_width)
    y_0 = random.randint(0, int(height / 2))
    y_1 = random.randint(y_0 + height_min, y_0 + height_max)
    mosaic_area = img_copy[y_0:y_1, x_0:x_1]
    original_shape = (mosaic_area.shape[1], mosaic_area.shape[0])
    mosaic_intensity = random.randint(5, 10)
    mosaic_area = mosaic_area[::mosaic_intensity, ::mosaic_intensity]
    mosaic_area = cv2.resize(mosaic_area, original_shape)
    img_copy[y_0:y_1, x_0:x_1] = mosaic_area
    return img_copy


def gamma_trans(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


def random_gamma(img):
    img_copy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    gamma_area_max = int(height * width)
    gamma_area_min = int((height * width) / 8)
    x_0 = random.randint(0, int(width / 2))
    x_1 = random.randint(x_0 + int(width / 16), width)
    gamma_width = x_1 - x_0
    height_min = int(gamma_area_min / gamma_width)
    height_max = int(gamma_area_max / gamma_width)
    y_0 = random.randint(0, int(height / 2))
    y_1 = random.randint(y_0 + height_min, y_0 + height_max)
    gamma_area = img_copy[y_0:y_1, x_0:x_1]
    gamma_area = gamma_trans(gamma_area.astype(np.uint8), random.uniform(0.2, 1.8))
    img_copy[y_0:y_1, x_0:x_1] = gamma_area
    return img_copy


def motion_blur(img, degree=30, angle=360):
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def random_motion(img):
    img_copy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    blur_area_max = int(height * width)
    blur_area_min = int((height * width) / 8)
    x_0 = random.randint(0, int(width / 2))
    x_1 = random.randint(x_0 + int(width / 16), width)
    blur_width = x_1 - x_0
    height_min = int(blur_area_min / blur_width)
    height_max = int(blur_area_max / blur_width)
    y_0 = random.randint(0, int(height / 2))
    y_1 = random.randint(y_0 + height_min, y_0 + height_max)
    blur_area = img_copy[y_0:y_1, x_0:x_1]
    blur_area = motion_blur(blur_area, degree=random.randint(10, 30), angle=random.randint(0, 360))
    img_copy[y_0:y_1, x_0:x_1] = blur_area
    return img_copy


def faceImg_trans(img):
    flag = random.random()
    if flag < 1 / 4:
        return random_gamma(img)
    elif 1 / 4 <= flag <= 2 / 4:
        return random_motion(img)
    elif 2 / 4 <= flag <= 3 / 4:
        return random_gamma(random_motion(img))
    else:
        return img


def partImg_trans(right_eye, left_eye, glabella, mouth):
    flag = random.random()
    if flag < 1 / 4:
        right_eye_trans = random_gamma(right_eye)
        left_eye_trans = random_gamma(left_eye)
        glabella_trans = random_gamma(glabella)
        mouth_trans = random_gamma(mouth)
    elif 1 / 4 <= flag < 2 / 4:
        right_eye_trans = random_motion(right_eye)
        left_eye_trans = random_motion(left_eye)
        glabella_trans = random_motion(glabella)
        mouth_trans = random_motion(mouth)
    elif 2 / 4 <= flag < 3 / 4:
        right_eye_trans = random_gamma(random_motion(right_eye))
        left_eye_trans = random_gamma(random_motion(left_eye))
        glabella_trans = random_gamma(random_motion(glabella))
        mouth_trans = random_gamma(random_motion(mouth))
    else:
        right_eye_trans = right_eye
        left_eye_trans = left_eye
        glabella_trans = glabella
        mouth_trans = mouth
    upper_ = np.concatenate((left_eye_trans, right_eye_trans), axis=1)
    lower_ = np.concatenate((mouth_trans, glabella_trans), axis=1)
    composition_ = np.concatenate((upper_, lower_), axis=0)
    return composition_


def get_face_and_composition(img, faceImg, lmks, size, self_supervise=False):
    # right_eye_region
    right_eye_right = min(lmks[36][0], lmks[37][0], lmks[38][0],
                          lmks[39][0], lmks[40][0], lmks[41][0])
    right_eye_top = min(lmks[36][1], lmks[37][1], lmks[38][1],
                        lmks[39][1], lmks[40][1], lmks[41][1])
    right_eye_left = max(lmks[36][0], lmks[37][0], lmks[38][0],
                         lmks[39][0], lmks[40][0], lmks[41][0])
    right_eye_bottom = max(lmks[36][1], lmks[37][1], lmks[38][1],
                           lmks[39][1], lmks[40][1], lmks[41][1])
    right_eye_p1 = (right_eye_right - 10, right_eye_top - 10)
    right_eye_p2 = (right_eye_left + 10, right_eye_bottom + 10)
    right_eye = img[right_eye_p1[1]:right_eye_p2[1], right_eye_p1[0]:right_eye_p2[0]]
    if right_eye.size == 0:
        right_eye = np.zeros((int(size / 2), int(size / 2), 3))

    # left_eye_region
    left_eye_right = min(lmks[42][0], lmks[43][0], lmks[44][0],
                         lmks[45][0], lmks[46][0], lmks[47][0])
    left_eye_top = min(lmks[42][1], lmks[43][1], lmks[44][1],
                       lmks[45][1], lmks[46][1], lmks[47][1])
    left_eye_left = max(lmks[42][0], lmks[43][0], lmks[44][0],
                        lmks[45][0], lmks[46][0], lmks[47][0])
    left_eye_bottom = max(lmks[42][1], lmks[43][1], lmks[44][1],
                          lmks[45][1], lmks[46][1], lmks[47][1])
    left_eye_p1 = (left_eye_right - 10, left_eye_top - 10)
    left_eye_p2 = (left_eye_left + 10, left_eye_bottom + 10)
    left_eye = img[left_eye_p1[1]:left_eye_p2[1], left_eye_p1[0]:left_eye_p2[0]]
    if left_eye.size == 0:
        left_eye = np.zeros((int(size / 2), int(size / 2), 3))

    # glabella region
    glabella_right = right_eye_left
    glabella_left = left_eye_right
    glabella_top = min(left_eye_top, right_eye_top)
    glabella_bottom = lmks[28][1]
    glabella_p1 = (glabella_right - 5, glabella_top - 40)
    glabella_p2 = (glabella_left - 5, glabella_bottom + 5)
    glabella = img[glabella_p1[1]:glabella_p2[1], glabella_p1[0]:glabella_p2[0]]
    if glabella.size == 0:
        glabella = np.zeros((int(size / 2), int(size / 2), 3))

    # mouth region
    mouth_right = min(lmks[48][0], lmks[49][0], lmks[50][0], lmks[51][0], lmks[52][0],
                      lmks[53][0], lmks[54][0], lmks[55][0], lmks[56][0], lmks[57][0],
                      lmks[58][0], lmks[59][0], lmks[60][0], lmks[61][0], lmks[62][0],
                      lmks[63][0], lmks[64][0], lmks[65][0], lmks[66][0], lmks[67][0])
    mouth_top = min(lmks[48][1], lmks[49][1], lmks[50][1], lmks[51][1], lmks[52][1],
                    lmks[53][1], lmks[54][1], lmks[55][1], lmks[56][1], lmks[57][1],
                    lmks[58][1], lmks[59][1], lmks[60][1], lmks[61][1], lmks[62][1],
                    lmks[63][1], lmks[64][1], lmks[65][1], lmks[66][1], lmks[67][1])
    mouth_left = max(lmks[48][0], lmks[49][0], lmks[50][0], lmks[51][0], lmks[52][0],
                     lmks[53][0], lmks[54][0], lmks[55][0], lmks[56][0], lmks[57][0],
                     lmks[58][0], lmks[59][0], lmks[60][0], lmks[61][0], lmks[62][0],
                     lmks[63][0], lmks[64][0], lmks[65][0], lmks[66][0], lmks[67][0])
    mouth_bottom = max(lmks[48][1], lmks[49][1], lmks[50][1], lmks[51][1], lmks[52][1],
                       lmks[53][1], lmks[54][1], lmks[55][1], lmks[56][1], lmks[57][1],
                       lmks[58][1], lmks[59][1], lmks[60][1], lmks[61][1], lmks[62][1],
                       lmks[63][1], lmks[64][1], lmks[65][1], lmks[66][1], lmks[67][1])
    mouth_p1 = (mouth_right - 10, mouth_top - 10)
    mouth_p2 = (mouth_left + 10, mouth_bottom + 10)
    mouth = img[mouth_p1[1]:mouth_p2[1], mouth_p1[0]:mouth_p2[0]]
    if mouth.size == 0:
        mouth = np.zeros((int(size / 2), int(size / 2), 3))

    faceImg = cv2.resize(faceImg, (size, size), interpolation=cv2.INTER_LINEAR)
    right_eye = cv2.resize(right_eye, (int(size / 2), int(size / 2)), interpolation=cv2.INTER_LINEAR)
    left_eye = cv2.resize(left_eye, (int(size / 2), int(size / 2)), interpolation=cv2.INTER_LINEAR)
    glabella = cv2.resize(glabella, (int(size / 2), int(size / 2)), interpolation=cv2.INTER_LINEAR)
    mouth = cv2.resize(mouth, (int(size / 2), int(size / 2)), interpolation=cv2.INTER_LINEAR)

    upper = np.concatenate((left_eye, right_eye), axis=1)
    lower = np.concatenate((mouth, glabella), axis=1)
    composition = np.concatenate((upper, lower), axis=0)

    if not self_supervise:
        return faceImg, composition
    if self_supervise:
        faceImg_ = faceImg_trans(faceImg.copy())
        composition_ = partImg_trans(right_eye=right_eye, left_eye=left_eye, glabella=glabella, mouth=mouth)
        return faceImg, faceImg_, composition, composition_


class Men:
    def __init__(self, device):
        self.face_detector = MTCNN(keep_all=True, device=device)
        self.lmk_detector = Detector(device=device)
        self.pre_bbox = None

    def transform(self, imgs, size, self_supervise=False):
        imgs = np.array(imgs)
        if len(imgs.shape) == 3:  # image
            img = imgs
            bboxes, _ = self.face_detector.detect(img)
            if bboxes is not None:
                bbox = bboxes[0]
                bbox = bbox.astype(np.int16)
                bbox = np.maximum(bbox, 0)
                faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if faceImg.size == 0:
                    bbox = self.pre_bbox
                    faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                else:
                    self.pre_bbox = bbox
                lmks, _ = self.lmk_detector.detect(img, bbox)
                lmks = lmks.astype(np.int16)

                if not self_supervise:
                    faceImg, composition = get_face_and_composition(img=img, faceImg=faceImg, lmks=lmks, size=size)
                else:
                    faceImg, faceImg_, composition, composition_ = get_face_and_composition(img=img, faceImg=faceImg,
                                                                                            lmks=lmks,
                                                                                            self_supervise=True,
                                                                                            size=size)

            if bboxes is None and self.pre_bbox is not None:
                bbox = self.pre_bbox
                faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                lmks, _ = self.lmk_detector.detect(img, bbox)
                lmks = lmks.astype(np.int16)

                if not self_supervise:
                    faceImg, composition = get_face_and_composition(img=img, faceImg=faceImg, lmks=lmks, size=size)
                else:
                    faceImg, faceImg_, composition, composition_ = get_face_and_composition(img=img, faceImg=faceImg,
                                                                                            lmks=lmks,
                                                                                            self_supervise=True,
                                                                                            size=size)

            if bboxes is None and self.pre_bbox is None:
                return None, None

            if not self_supervise:
                faceImg = torch.from_numpy(faceImg).permute(2, 0, 1).float()
                composition = torch.from_numpy(composition).permute(2, 0, 1).float()
                return faceImg, composition
            else:
                faceImg = torch.from_numpy(faceImg).permute(2, 0, 1).float()
                faceImg_ = torch.from_numpy(faceImg_).permute(2, 0, 1).float()
                composition = torch.from_numpy(composition).permute(2, 0, 1).float()
                composition_ = torch.from_numpy(composition_).permute(2, 0, 1).float()
                return faceImg, faceImg_, composition, composition_

        if len(imgs.shape) == 4:  # image sequence
            face_imgs = []
            part_imgs = []
            face_imgs_ = []
            part_imgs_ = []
            for img in imgs:
                bboxes, _ = self.face_detector.detect(img)
                if bboxes is not None:
                    bbox = bboxes[0]
                    bbox = bbox.astype(np.int16)
                    bbox = np.maximum(bbox, 0)
                    faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if faceImg.size == 0:
                        bbox = self.pre_bbox
                        faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    else:
                        self.pre_bbox = bbox
                    lmks, _ = self.lmk_detector.detect(img, bbox)
                    lmks = lmks.astype(np.int16)

                    if not self_supervise:
                        faceImg, composition = get_face_and_composition(img=img, faceImg=faceImg, lmks=lmks, size=size)
                        face_imgs.append(faceImg)
                        part_imgs.append(composition)
                    else:
                        faceImg, faceImg_, composition, composition_ = get_face_and_composition(img=img,
                                                                                                faceImg=faceImg,
                                                                                                lmks=lmks,
                                                                                                self_supervise=True,
                                                                                                size=size)
                        face_imgs.append(faceImg)
                        part_imgs.append(composition)
                        face_imgs_.append(faceImg_)
                        part_imgs_.append(composition_)

                if bboxes is None and self.pre_bbox is not None:
                    bbox = self.pre_bbox
                    faceImg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    lmks, _ = self.lmk_detector.detect(img, bbox)
                    lmks = lmks.astype(np.int16)

                    if not self_supervise:
                        faceImg, composition = get_face_and_composition(img=img, faceImg=faceImg, lmks=lmks, size=size)
                        face_imgs.append(faceImg)
                        part_imgs.append(composition)
                    else:
                        faceImg, faceImg_, composition, composition_ = get_face_and_composition(img=img,
                                                                                                faceImg=faceImg,
                                                                                                lmks=lmks,
                                                                                                self_supervise=True,
                                                                                                size=size)
                        face_imgs.append(faceImg)
                        part_imgs.append(composition)
                        face_imgs_.append(faceImg_)
                        part_imgs_.append(composition_)

                if bboxes is None and self.pre_bbox is None:
                    return None, None

            face_imgs = np.array(face_imgs)
            part_imgs = np.array(part_imgs)
            face_imgs_ = np.array(face_imgs_)
            part_imgs_ = np.array(part_imgs_)
            if not self_supervise:
                faceImg = torch.from_numpy(face_imgs).permute(0, 3, 1, 2).float()
                composition = torch.from_numpy(part_imgs).permute(2, 0, 1).float()
                return faceImg, composition
            else:
                faceImg = torch.from_numpy(face_imgs).permute(0, 3, 1, 2).float()
                faceImg_ = torch.from_numpy(face_imgs_).permute(0, 3, 1, 2).float()
                composition = torch.from_numpy(part_imgs).permute(0, 3, 1, 2).float()
                composition_ = torch.from_numpy(part_imgs_).permute(0, 3, 1, 2).float()
                return faceImg, faceImg_, composition, composition_

# def puzzle(img):
#     img_copy = img.copy()
#     height = img.shape[0]
#     width = img.shape[1]
#     fragment0 = img_copy[0:int(height / 2), 0:int(width / 2)]
#     fragment1 = img_copy[0:int(height / 2), int(width / 2):width]
#     fragment2 = img_copy[int(height / 2):height, 0:int(width / 2)]
#     fragment3 = img_copy[int(height / 2):height, int(width / 2):width]
#     fragments = [fragment0, fragment1, fragment2, fragment3]
#     np.random.shuffle(fragments)
#     new = np.zeros_like(img_copy)
#     new[0:int(height / 2), 0:int(width / 2)] = fragments[0]
#     new[0:int(height / 2), int(width / 2):width] = fragments[1]
#     new[int(height / 2):height, 0:int(width / 2)] = fragments[2]
#     new[int(height / 2):height, int(width / 2):width] = fragments[3]
#     return new
