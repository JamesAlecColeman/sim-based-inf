def check_cache(mesh_info_dict, keys_to_read):
    """ Attempts to load value in cache corresponding to each key, None if it isn't in the cache """
    values_read = []

    for key in keys_to_read:
        if key in mesh_info_dict:
            values_read.append(mesh_info_dict[key])
        else:
            values_read.append(None)

    return tuple(values_read)


def clear_cache(mesh_info_dict, keys_to_clear):

    for key in keys_to_clear:
        if key in mesh_info_dict:
            del mesh_info_dict[key]

    return mesh_info_dict