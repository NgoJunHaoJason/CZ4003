'''
Functions for performing Otsu's algorithm.

3 variations:

1. the standard way Otsu's algorithm on the whole image
2. segmenting the image into blocks, and performing Otsu's algorithm on each block
3. use sliding window on the image, and use Otsu's algorithm within the window
'''
import cv2
import numpy as np


def threshold(image: np.ndarray) -> np.ndarray:
    '''
    Otsu algorithm, done the standard way. (global threshold)
    
    The mask that is obtained can be used to binarize the image, by doing: 
    np.where(image > threshold_mask, 255, 0)

    :params:
    - image (np.ndarray): the image to find the threshold of

    :return:
    - mask of threshold (np.ndarray)
    '''
    histogram = cv2.calcHist(
        images=[image],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
    )

    num_pixels = image.size

    gray_level_probabilities = [
        bin_count[0] / num_pixels
        for bin_count in histogram
    ]
    
    best_threshold = -1
    min_intraclass_variance = float('inf')

    for threshold in range(len(histogram)):  # 0 to 255
        lower_group_probability = sum(gray_level_probabilities[:threshold+1])
        upper_group_probability = sum(gray_level_probabilities[threshold+1:])

        if lower_group_probability == 0 or upper_group_probability == 0:
            continue  # otherwise, divide by 0

        lower_group_mean = sum([
            gray_level * probability  # gray level == pixel value
            for gray_level, probability in enumerate(gray_level_probabilities[:threshold+1])
        ]) / lower_group_probability

        upper_group_mean = sum([
            gray_level * probability
            for gray_level, probability in enumerate(gray_level_probabilities[threshold+1:])
        ]) / upper_group_probability

        lower_group_variance = sum([
            ((gray_level - lower_group_mean)**2) * probability
            for gray_level, probability in enumerate(gray_level_probabilities[:threshold+1])
        ]) / lower_group_probability

        upper_group_variance = sum([
            ((gray_level - upper_group_mean)**2) * probability
            for gray_level, probability in enumerate(gray_level_probabilities[threshold+1:])
        ]) / upper_group_probability

        intraclass_variance = lower_group_probability * lower_group_variance + \
            upper_group_probability * upper_group_variance

        if intraclass_variance < min_intraclass_variance:
            min_intraclass_variance = intraclass_variance
            best_threshold = threshold

    return np.full_like(image, fill_value=best_threshold)



def segmented_threshold(
    image: np.ndarray,
    num_vertical_segments: int,
    num_horizontal_segments: int,
) -> np.ndarray:
    '''
    Otsu threshold, done on segments of the image.
    
    Total number of segments = num_vertical_segments * num_horizontal_segments

    The mask that is obtained can be used to binarize the image, by doing: 
    np.where(image > threshold_mask, 255, 0)

    :params:
    - image (np.ndarray): the image to find the threshold of
    - num_vertical_segments (int): number of times to segment vertically; at least 1
    - num_horizontal_segments (int): number of times to segment horizontally; at least 1

    :return:
    - mask of threshold (np.ndarray)
    '''
    if num_vertical_segments < 1:
        raise ValueError('There must be at least 1 segment in vertical direction')
    if num_horizontal_segments < 1:
        raise ValueError('There must be at least 1 segment in horizontal direction')

    image_height, image_length = image.shape

    # take ceiling to ensure entire image is covered
    segment_height = image_height // num_vertical_segments + 1
    segment_length = image_length // num_horizontal_segments + 1

    threshold_mask = np.zeros_like(image)

    for vertical_offset in range(0, image_height, segment_height):
        for horizontal_offset in range(0, image_length, segment_length):
            image_segment = image[
                vertical_offset: vertical_offset + segment_height,
                horizontal_offset: horizontal_offset + segment_length
            ]

            # standard Otsu's algorithm
            segmented_threshold_mask = threshold(image_segment)

            threshold_mask[
                vertical_offset: vertical_offset + segment_height,
                horizontal_offset: horizontal_offset + segment_length
            ] = segmented_threshold_mask
            
    return threshold_mask


def sliding_window_threshold(
    image: np.ndarray,
    window_height: int,
    window_length: int,
    vertical_stride: int,
    horizontal_stride: int,
) -> np.ndarray:
    '''
    Otsu threshold, done by using a sliding window over the image.

    The mask that is obtained can be used to binarize the image, by doing: 
    np.where(image > threshold_mask, 255, 0)

    :params:
    - image (np.ndarray): the image to find the threshold of
    - window_height (int): height of the window i.e. vertical width
    - window_length (int): length of the window i.e. horizontal width
    - vertical_stride (int): number of pixel-wise steps to take in the vertical direction
    - horizontal_stride (int): number of pixel-wise steps to take in the horizontal direction

    :return:
    - mask of threshold (np.ndarray)
    '''
    if window_height < 1:
        raise ValueError('Window must have a height of at least 1 pixel')
    if window_length < 1:
        raise ValueError('Window must have a length of at least 1 pixel')
    if vertical_stride < 1:
        raise ValueError('Stride in the vertical direction must be of at least 1 pixel')
    if horizontal_stride < 1:
        raise ValueError('Stride in the horizontal direction must be of at least 1 pixel')

    image_height, image_length = image.shape

    threshold_mask = np.zeros_like(image, dtype=np.float64)
    num_repetitions = np.zeros_like(image)

    for vertical_offset in range(0, image_height - window_height + vertical_stride, vertical_stride):
        for horizontal_offset in range(0, image_length - window_length + horizontal_stride, horizontal_stride):
            window = image[
                vertical_offset: vertical_offset + window_height,
                horizontal_offset: horizontal_offset + window_length
            ]

            # standard Otsu's algorithm
            window_threshold_mask = threshold(window)

            threshold_mask[
                vertical_offset: vertical_offset + window_height,
                horizontal_offset: horizontal_offset + window_length
            ] += np.mean(window_threshold_mask)

            num_repetitions[
                vertical_offset: vertical_offset + window_height,
                horizontal_offset: horizontal_offset + window_length
            ] += 1
            
    return threshold_mask / num_repetitions  # get average
