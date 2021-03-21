from copy import deepcopy


def sort_format(supported_formats, format, name):

    newformat = deepcopy(format)

    if newformat is None:
        for detect in supported_formats:
            if detect in name.lower():
                newformat = detect
                break

    if newformat is None:
        newformat = supported_formats[0]

    newformat = newformat.lower()
    newname = f"{name}"
    if f"{newformat}" != name[-len(newformat) :]:
        newname += f".{newformat}"

    return newformat, newname
