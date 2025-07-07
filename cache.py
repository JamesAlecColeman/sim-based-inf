def check_cache(mesh_info_dict, keys_to_read):
    """ Read values from the mesh cache dict

    Args:
        mesh_info_dict (dict): mesh cache e.g. {"endo_mask": [0, 1, 0, ...], "lv_mask": [1, 1, 0, ...], ...}
        keys_to_read (list of keys): to be read from mesh cache e.g. ["endo_mask", "lv_mask", ...]

    Returns:
        values_read (tuple of values): values read from cache, or None if key is not present
    """
    values_read = []

    for key in keys_to_read:
        if key in mesh_info_dict:
            values_read.append(mesh_info_dict[key])
        else:
            values_read.append(None)

    return tuple(values_read)


def clear_cache(mesh_info_dict, keys_to_clear):
    """ Delete entries from cache

    Args:
        mesh_info_dict (dict): mesh cache e.g. {"endo_mask": [0, 1, 0, ...], "lv_mask": [1, 1, 0, ...], ...}
        keys_to_clear (list of keys): to be deleted from mesh cache e.g. ["endo_mask"]

    Returns:
        mesh_info_dict (dict): post-deletion mesh cache e.g. {"endo_mask": [0, 1, 0, ...], ...}
    """
    for key in keys_to_clear:
        if key in mesh_info_dict:
            del mesh_info_dict[key]

    return mesh_info_dict