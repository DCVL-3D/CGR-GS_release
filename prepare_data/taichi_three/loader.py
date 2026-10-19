import numpy as np

def _append(faces, indices):
    if len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) == 3:
        faces.append(indices)
    else:
        assert False, len(indices)

def readobj(path, scale=1):
    vi = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == 'v':
            vi.append(fields)
        elif type == 'vt':
            vt.append(fields)
        elif type == 'vn':
            vn.append(fields)

    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        if type != 'f':
            continue

        indices = [[int(_) - 1 if _ != '' else 0 for _ in field.split('/')] for field in fields]

        if len(indices) == 4:
            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[2], indices[3], indices[0]])
        elif len(indices) == 3:
            faces.append(indices)
        else:
            assert False, len(indices)

    ret = {}
    ret['vi'] = None if len(vi) == 0 else np.array(vi).astype(np.float32) * scale
    ret['vt'] = None if len(vt) == 0 else np.array(vt).astype(np.float32)
    ret['vn'] = None if len(vn) == 0 else np.array(vn).astype(np.float32)
    ret['f'] = None if len(faces) == 0 else np.array(faces).astype(np.int32)
    return ret
