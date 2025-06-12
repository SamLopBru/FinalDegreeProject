def samplingSlices(slices, num_slices=27):
    total_slices = len(slices)

    if total_slices >= 100 and total_slices < 110:
        slices = slices[3:total_slices-3]
    elif total_slices >=  110 and total_slices<120:
        slices = slices[6:total_slices-6]

    first_turn = [slice for i, slice in enumerate(slices) if i%3 ==0]
    second_turn = [slice for i, slice in enumerate(slices) if (i+2)%3 ==0]
    third_turn = [slice for i, slice in enumerate(slices) if (i+1)%3 ==0]

    selected_slices = first_turn + second_turn + third_turn
    selected_slices = selected_slices[:num_slices]

    return selected_slices

def get_original_index(sampled_index, total_slices, num_sampled=27):
    """
    Returns the index in the original set of a sampled slice.
    """
    # Determine how many slices were cut
    if 100 <= total_slices < 110:
        start_crop = 3
    elif 110 <= total_slices < 120:
        start_crop = 6
    else:
        start_crop = 0  

    cropped_total = total_slices - 2 * start_crop  

    first_indices = [i for i in range(cropped_total) if i % 3 == 0]
    second_indices = [i for i in range(cropped_total) if (i + 2) % 3 == 0]
    third_indices = [i for i in range(cropped_total) if (i + 1) % 3 == 0]

    full_order = first_indices + second_indices + third_indices
    full_order = full_order[:num_sampled] 

    if sampled_index >= len(full_order):
        raise IndexError("The sampled index is outside the range of selected slices.")

    original_in_cropped = full_order[sampled_index]
    original_index = original_in_cropped + start_crop  

    return original_index

def unresizePascalVOC(bbox_resized, original_size, new_size):
    """
    Given a Pascal VOC bounding box ([x_min, y_min, x_max, y_max]) in the resized image,
    returns the bbox in Pascal VOC format but scaled to the original size.

    Args:
    - bbox_resized: [x_min, y_min, x_max, y_max] (in the resized image).
    - original_size: (original_width, original_height) of the original image
    - new_size: (new_width, new_height) of the resized size.

    Returns:
    - [x_min, y_min, x_max, y_max] in original image
    """
    new_width, new_height = new_size
    original_width, original_height = original_size

    x_min_resized, y_min_resized, x_max_resized, y_max_resized = bbox_resized

    x_min = x_min_resized * (original_width / new_width)
    y_min = y_min_resized * (original_height / new_height)
    x_max = x_max_resized * (original_width / new_width)
    y_max = y_max_resized * (original_height / new_height)

    # Clip for ensure that the coordinates are within the original image dimensions
    x_min = max(0, min(x_min, original_width))
    y_min = max(0, min(y_min, original_height))
    x_max = max(0, min(x_max, original_width))
    y_max = max(0, min(y_max, original_height))

    return [x_min, y_min, x_max, y_max]



