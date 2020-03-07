import numpy as np
import os
import glob
import cv2
import math
import concurrent.futures as cf
import augmentation_transforms as transforms
from random import*
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import cpu_count
from tqdm import tqdm

SUPPORTED_FILE_EXTENSIONS = ["png", "jpeg", "jpg", "bmp"]

#------------------------------------------------------------------------------------------------------------------------
def padding(image, pad_size):

    org_size = image.shape[:2]
    assert org_size[0] <= pad_size[0] and org_size[1] <= pad_size[1]
    assert np.all(np.array([org_size, pad_size]) % 2 == 0)
    image = _padding_img(image, pad_size)
    return image

def _padding_img(image, pad_size):

    org_size = image.shape[:2]
    image = np.copy(image)
    pad_w = (pad_size[1] - org_size[1]) // 2
    pad_h = (pad_size[0] - org_size[0]) // 2
    img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),'constant', constant_values=0)
    return image

def crop_shape(image, crop_size):
    assert np.all(np.array(crop_size) % 2 == 0)
    org_size = image.shape[:2]
    if org_size[0] % 2 != 0:
        image = image[1:,:,:]
    if org_size[1] % 2 != 0:
        image = image[:,1:,:]
    org_size = image.shape[:2]
    if org_size[0] > crop_size[0]:
        diff = org_size[0] - crop_size[0]
        image = image[diff//2:org_size[0]-diff//2,:,:]
    if org_size[1] > crop_size[1]:
        diff = org_size[1] - crop_size[1]
        image = image[:,diff//2:org_size[1]-diff//2,:]
    return image

def random_crop(image, crop_width, crop_height):
    h, w = image.shape[:2]
    if h == crop_width and w == crop_height:
        return image
    top = np.random.randint(0, h - crop_width)
    left = np.random.randint(0, w - crop_height)
    bottom = top + crop_height
    right = left + crop_width
    image = image[top:bottom, left:right]
    return image

def gaussain_blur(image, max_filiter_size = 3, sigma_min = 1, sigma_max = 1) :
	image = image.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		sigma = random.uniform(sigma_min, sigma_max)
		out_image = cv2.GaussianBlur(image, (filter_size, filter_size), sigma)
	return out_image

def gaussain_noise(image, mean = 0, var = 0.1) :
	image = image.astype(np.uint8)
	h, w, c = image.shape
	sigma = var ** 0.5
	gauss = np.random.normal(mean, sigma, (h, w, c))
	gauss = gauss.reshape(h, w, c).astype(np.uint8)
	out_image = image + gauss
	return out_image

def read_data(dir_image, dir_label, crop_size, crop=True, no_label=False):
    images = []
    labels = []
    names = []
    image_paths = []

    image_paths.extend(glob.glob(os.path.join(dir_image, '*.*')))

    for path in image_paths:
        image_name = os.path.splitext(os.path.basename(path))[0]
        image_extension = path.split(".")[-1]
        
        if image_extension.lower() not in SUPPORTED_FILE_EXTENSIONS:
            continue

        names.append(image_name)
        image = cv2.imread(path)#, cv2.IMREAD_UNCHANGED)

        if crop:
            if crop_size[0] > 0 and crop_size[1] > 0 and image.shape[0] > crop_size[0] > 0 and image.shape[1] > crop_size[1]:
                image = crop_shape(image, crop_size)
                image = padding(image, crop_size)

        if len(image) == 2:
            h, w = image.shape[:2]
            image.reshape(h, w, 1)
        images.append(image)

        if no_label:
            labels.append(None)
            continue

        label_path = os.path.join(dir_label, '{}.{}'.format(image_name, image_extension))
        label = cv2.imread(label_path)#, cv2.IMREAD_UNCHANGED)

        if crop:
            if crop_size[0] > 0 and crop_size[1] > 0 and label.shape[0] > crop_size[0] > 0 and label.shape[1] > crop_size[1]:
                label = crop_shape(label, crop_size)
                label = padding(label, crop_size)

        if len(label) == 2:
            h, w = label.shape[:2]
            label.reshape(h, w, 1)
        labels.append(label)
        
    if no_label == False:
        return images, labels, names

    return images, None, names


def read_list_data(image_root, image_list, crop_size, crop=True, resize=False):
    
    images = []
    
    for file_name in image_list:
        image_file_path = os.path.join(image_root, file_name);

        image = cv2.imread(image_file_path)#, cv2.IMREAD_UNCHANGED)

        if crop:
            if crop_size[0] > 0 and crop_size[1] > 0 and image.shape[1] > crop_size[0] and image.shape[0] > crop_size[1]:
                image = crop_shape(image, crop_size)
                image = padding(image, crop_size)

        if resize:
            if not crop_size[0] == image.shape[1] or not crop_size[1] == image.shape[0]:
                image = cv2.resize(image, dsize=(crop_size[0], crop_size[1]), interpolation=cv2.INTER_LANCZOS4)
        
        if len(image) == 2:
            h, w = image.shape[:2]
            image.reshape(h, w, 1)

        images.append(image)
    return images

#-----------------------------------------------------------------------------------------------------------------------
# Text file load
def read_class_txt(path):
    
    class_list = []
    
    if os.path.isfile(path):
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                
                line_list = line.split()
                class_list.append(line_list[0])
    return class_list

def read_image_path_txt(path, class_num):
    
        files_list = []
        labels_list = []
        dict_data = {}

        if os.path.isfile(path):
            with open(path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
     
                    line_list = line.split(',')
                    files_list.append(line_list[0])
                    labels_list.append(line_list[2])
                
            if files_list and labels_list :
                files_list, labels_list = shuffle_list(files_list, labels_list)
        return files_list, labels_list


def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return list(zip(*l))

def stack_bayer(image_rgb):
	temp_bayer = np.zeros([image_rgb.shape[0], image_rgb.shape[1]], dtype = np.uint8)
	for j in range(temp_bayer.shape[0]):
		for i in range(temp_bayer.shape[1]):
			if (i % 2 == 0) and (j % 2 == 0):
				temp_bayer[j, i] = image_rgb[j, i, 0]
			elif (i % 2 == 1) and (j % 2 == 1):
				temp_bayer[j, i] = image_rgb[j, i, 2] 
			else:
				temp_bayer[j, i] = image_rgb[j, i, 1]
	bayer = cv2.cvtColor(temp_bayer, cv2.COLOR_BAYER_BG2BGR)
	return bayer

#-----------------------------------------------------------------------------------------------------------------------
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def add_particle(image, count=10, size_range=(1,10), isBright_defect=False):

    h,w = image.shape[:2]

    for index in range(count):
        size = random.randint(size_range[0], size_range[1])
        mask_size = int(size * 3.0)

        kernel_size = size//2 + 1
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        
        # opencv reference
        sigma_min = 0.3*(kernel_size/2.0 - 1.0)+0.8
        sigma = random.uniform(sigma_min, sigma_min + 5.0)
        offset = 10

        if len(image.shape) == 3:
            mask = np.zeros((mask_size + offset, mask_size + offset, 3), dtype=np.int16)
        else:
            mask = np.zeros((mask_size + offset, mask_size + offset), dtype=np.int16)

        color_rnd = random.randint(30, 70)
        shape_color = (255 - color_rnd, 255 - color_rnd, 255 - color_rnd)
        cv2.circle(mask, (mask.shape[0]//2, mask.shape[1]//2), size//2, shape_color, -1, lineType=cv2.LINE_AA)

        if size > 100:
            # Apply transformation on image
            mask = elastic_transform(mask, mask.shape[1] * 2, mask.shape[1] * 0.1, mask.shape[1] * 0.005)
        #seletect_rnd = random.randint(0, 3)

        #if seletect_rnd == 0:
        #    sinDistort(mask)
        #elif seletect_rnd == 1:
        #    randomDistort2(mask)

        x = random.randint(mask_size + offset, w - (mask_size + offset))
        y = random.randint(mask_size + offset, h - (mask_size + offset))

        roi_image = image[y:y+mask_size+offset,x:x+mask_size+offset]
        mask_image = mask & roi_image
        mask_image = cv2.GaussianBlur(mask_image, (kernel_size, kernel_size), sigma) * random.uniform(0.5, 1.5)

        if isBright_defect:
            mask_image = mask_image * -1.0
        mask_image = mask_image.astype(np.int16)
        roi_image = roi_image.astype(np.int16)
        #particle_image = abs(roi_image - mask_image)
        particle_image = np.subtract(roi_image, mask_image)

        particle_image[particle_image<0] = 0
        particle_image[particle_image>255] = 255
        image[y:y+mask_size+offset,x:x+mask_size+offset] = particle_image
    return image

def dataset_thread_generator(images, crop_size, image_count):
    
    input_image = []
    input_label = []
    results = []

    if len(images) == 0:
        return input_image, input_label

    try:
        workers = 4 #cpu_count()
    except NotImplementedError:
        workers = 1
    
    chunk = image_count // workers

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(workers):
            results.append(executor.submit(dataset_generator, images, crop_size, chunk))

        for future in cf.as_completed(results):
            res = future.result()
            input_image.extend(res[0])
            input_label.extend(res[1])

    # using Debugging
    #dataset_generator(images, image_count, crop_size)

    return input_image, input_label

def dataset_add_particle(image, count, crop_size):
    input_label = []
    input_image = []
    for no in range(0, count):
        crop_image = random_crop(image, crop_size, crop_size)
        # append label iamge
        input_label.append(crop_image)

        crop_image = add_particle(crop_image, count=10, size_range=(1,10), isBright_defect=False)
        #cv2.imwrite('d:/test/{}_test.bmp'.format(no), crop_image)
        # append input image
        input_image.append(crop_image)

    return input_image, input_image

def dataset_generator_PAD(images, crop_size, image_count, blur_enable, blur_kernel, blur_sigma):
    input_image = []
    input_label = []

    for image in images:
        for no in range(0, image_count):
            crop_image = random_crop(image, crop_size, crop_size)
            # append label iamge
            input_label.append(crop_image)

            crop_image = add_particle(crop_image.copy(), count=5, size_range=(3,10), isBright_defect=False)
            crop_image = add_particle(crop_image.copy(), count=5, size_range=(3,10), isBright_defect=True)
            #cv2.imwrite('d:/test/{}_test.bmp'.format(no), crop_image)
            # append input image
            input_image.append(crop_image)

    return input_image, input_label

def dataset_generator(images, crop_size, image_count):
    input_image = []
    input_label = []
    scale = 1

    for image in images:
        for no in range(0, image_count):
            crop_size = math.ceil(crop_size//scale) * scale
            crop_image = random_crop(image, crop_size, crop_size)
            # append label iamge
            input_label.append(crop_image)

            # append input image
            height, width = crop_image.shape[:2]
            resize_image = cv2.resize(crop_image, (width//scale, height//scale))
            input_image.append(resize_image)
    return input_image, input_label

#-----------------------------------------------------------------------------------------------------------------------

def dataAugmentation_run(index_start, index_end, mode, dict_rotation, dict_blur_gaussian, dict_noise_gaussian, input_images, label_images):
    dataAugmentation = DataAugmentation(dict_rotation=dict_rotation, dict_blur_gaussian=dict_blur_gaussian, dict_noise_gaussian=dict_noise_gaussian)
    dataAugmentation(index_start, index_end, mode, input_images, label_images)

def dataAugmentation_run_thread(mode, dict_rotation, dict_blur_gaussian, dict_noise_gaussian, input_images, label_images):
    try:
        workers = 1 #cpu_count()
    except NotImplementedError:
        workers = 1

    image_count = len(input_images)
    if image_count == 0 or workers > image_count:
        return
    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(workers):
            index_start = i * (image_count//workers)
            index_end = (i + 1) * (image_count//workers)
            if i == workers - 1:
                index_end = image_count
            results.append(executor.submit(dataAugmentation_run, index_start, index_end, mode, dict_rotation, dict_blur_gaussian, dict_noise_gaussian, input_images, label_images))

        for future in cf.as_completed(results):
            res = future.result()


class DataAugmentation(object):
    def __init__(self, dict_rotation, dict_blur_gaussian, dict_noise_gaussian):
        # rotation
        self.transform_rotation = [transforms.Skip()]
        if dict_rotation['enable_CW90']:
            self.transform_rotation.append(transforms.RandomRotation(degrees=90))
        if dict_rotation['enable_CCW90']:
            #self.transform_rotation.append(transforms.RandomResizedCrop(size=96, scale=(0.7, 1)))
            self.transform_rotation.append(transforms.RandomResizedCrop(size=128, scale=(0.7, 1), ratio=(1, 1))) # 20200115 hcw
        if dict_rotation['enable_h_flip']:
            self.transform_rotation.append(transforms.RandomHorizontalFlip())
        if dict_rotation['enable_v_flip']:
            self.transform_rotation.append(transforms.RandomVerticalFlip())

        # etc
        self.transform = [transforms.Skip()]
        if dict_blur_gaussian['enable']:
            self.transform.append(transforms.RandomGaussianBlur(dict_blur_gaussian['max_kernel_size'], dict_blur_gaussian['min_sigma'], dict_blur_gaussian['max_sigma']))
        if dict_noise_gaussian['enable']:
            self.transform.append(transforms.RandomGaussianNoise(max_var=dict_noise_gaussian['var']))

    def __call__(self, index_start, index_end, mode, input_images, label_images=None):

        if mode == 'CLASSIFICATION':
            label_images = None

        for i in range(index_start, index_end):
            #rotation_augm = transforms.RandomChoice(self.transform_rotation)
            rotation_augm = transforms.RandomOrder(self.transform_rotation)
            input_images[i] = rotation_augm(input_images[i])
            if label_images is not None:
                label_images[i] = rotation_augm(label_images[i])

            if 0.5 < random():
                random_augm = transforms.RandomOrder(self.transform) # 섞어서 모두 진행
            else:
                random_augm = transforms.RandomChoice(self.transform) # 랜덤으로 한개 선택

            input_images[i] = random_augm(input_images[i])

            if mode == 'DM':
                bayer_trans = transforms.RandomMakeBayerImage()
                input_images[i] = bayer_trans(input_images[i])

    #def __call__(self, mode, input_images, label_images=None):
    #    for index, image in enumerate(input_images):
    #        rotation_augm = transforms.RandomChoice(self.transform_rotation) 
    #        input_images[index] = rotation_augm(image)
    #        if label_images is not None:
    #            label_images[index] = rotation_augm(label_images[index])

    #        if 0.5 < random.random():
    #            random_augm = transforms.RandomOrder(self.transform) # 섞어서 모두 진행
    #        else:
    #            random_augm = transforms.RandomChoice(self.transform) # 랜덤으로 한개 선택

    #        input_images[index] = random_augm(input_images[index])

    #        if mode == 'DM':
    #            bayer_trans = transforms.RandomMakeBayerImage()
    #            input_images[index] = bayer_trans(input_images[index])

#-----------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape: (N, H, W, num_classes (include background)).
        """

        images = np.array(images, dtype=np.uint8)

        if labels is not None:
            labels = np.array(labels, dtype=np.uint8)
            assert images.shape[0] == labels.shape[0],\
                ('Number of examples mismatch, between images and labels')
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels  # NOTE: this can be None, if not given.
        # image/label indices(can be permuted)
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def sample_batch(self, batch_size, shuffle=True):
        """
        Return sample examples from this dataset.
        :param batch_size: int, size of a sample batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        if shuffle:
            indices = np.random.choice(self._num_examples, batch_size)
        else:
            indices = np.arange(batch_size)
        batch_images = self._images[indices]
        if self._labels is not None:
            batch_labels = self._labels[indices]
        else:
            batch_labels = None
        return batch_images, batch_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next 'batch_size' examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number
        # of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self._images[indices_rest_part]
            images_new_part = self._images[indices_new_part]
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)
            if self._labels is not None:
                labels_rest_part = self._labels[indices_rest_part]
                labels_new_part = self._labels[indices_new_part]
                batch_labels = np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self._images[indices]
            if self._labels is not None:
                batch_labels = self._labels[indices]
            else:
                batch_labels = None

        return batch_images, batch_labels