import numpy as np
import nibabel as nib
import pydicom
from os.path import join, exists, basename, split
from os import listdir, environ
from skimage.draw import polygon
import pickle

_names_sorting_warning_printed = False
_multiple_roi_dcms_warning_printed = False


def load_pkl(path_to_file):
    with open(path_to_file, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pkl(data, path_to_file):
    if not path_to_file.endswith('.pkl'):
        path_to_file += '.pkl'
    with open(path_to_file, 'wb') as file:
        pickle.dump(data, file)


def read_nii(nii_file):
    img = nib.load(nii_file)
    spacing = img.header['pixdim'][1:4]
    return img.get_fdata(), spacing


def read_nii_files(nii_files):
    '''
    read_nii_files(*args)

    reads nii files that belong togehter, e.g. image channels and a
    corresponding segmentation.

    Parameters
    ----------
    *args : list
        full pathes to nii files to be read

    Raises
    ------
    ValueError
        if images found do not have the same voxel spacing

    Returns
    -------
    image
        4d tensor with images stacked up in first (channel) dimension
    spacing : len 3
        voxel spacing in x, y, z direction in mm

    '''
    im, spacing = read_nii(nii_files[0])
    out_volumes = [im]
    for nii_file in nii_files[1:]:
        out = read_nii(nii_file)
        if np.any(out[1] != spacing):
            raise ValueError('Spacing didn\'t match for '
                             + nii_files[0] + ' and ' + nii_file + '. Got '
                             + str(spacing) + ' and ' + str(out[1]))
        out_volumes.append(out[0])
    return np.stack(out_volumes), spacing


def read_data_tpl_from_nii(folder, case):
    data_tpl = {}

    if not exists(folder):
        folder = join(environ['OV_DATA_BASE'], 'raw_data', folder)

    if not exists(folder):
        raise FileNotFoundError('Can\'t read from folder {}. It doesn\'t exist.'.format(folder))

    if isinstance(case, int):
        case = 'case_%03d.nii.gz' % case

    if not isinstance(case, str):
        raise TypeError('Input \'case\' must be string, not {}'.format(type(case)))

    # first let's read the data info
    if exists(join(folder, 'data_info.pkl')):
        data_info = load_pkl(join(folder, 'data_info.pkl'))
        if case[:-7] in data_info:
            data_tpl.update(data_info[case[:-7]])

    image_folders_ex = [join(folder, imf) for imf in ['images', 'imagesTr', 'imagesTs']
                        if exists(join(folder, imf))]
    if len(image_folders_ex) == 0:
        raise FileNotFoundError('Didn\'t find any image folder in {}.'.format(folder))

    image_files = []
    for image_folder in image_folders_ex:
        matching_files = [join(folder, image_folder, file) for file in
                          listdir(join(folder, image_folder))
                          if file.startswith(case[:-7])]
        if len(matching_files) > 0 and len(image_files) > 0:
            raise FileExistsError('Found images for in multiple image folders at path {} for '
                                  'case {}.'.format(folder, case))
        image_files = matching_files

    if len(image_files) == 0:
        raise FileNotFoundError('No image files found for case {}.'.format(case))
    elif len(image_files) == 1:
        raw_image_file = image_files[0]
        im, spacing = read_nii(raw_image_file)
    else:
        raw_image_file = image_files
        im_data = [read_nii(file) for file in raw_image_file]
        ims = [im for im, spacing in im_data]
        spacings = [spacing for im, spacing in im_data]

        if not np.all([np.all(spacings[0] == sp) for sp in spacings[1:]]):
            raise ValueError('Found unequal spacings when reading the image files {}'
                             ''.format(image_files))
        im = np.stack(ims)
        spacing = spacings[0]

    data_tpl['image'] = im
    data_tpl['spacing'] = spacing
    data_tpl['raw_image_file'] = raw_image_file

    label_folders_ex = [join(folder, lbf) for lbf in ['labels', 'labelsTr', 'labelsTs']
                        if exists(join(folder, lbf))]
    if len(label_folders_ex) == 0:
        return data_tpl

    label_files = []
    for label_folder in label_folders_ex:
        matching_files = [join(folder, image_folder, file) for file in
                          listdir(join(folder, image_folder))
                          if file.startswith(case[:-7])]
        if len(matching_files) > 0 and len(label_files) > 0:
            raise FileExistsError('Found labels for in multiple label folders at path {} for '
                                  'case {}.'.format(folder, case))
        label_files = matching_files
    if len(label_files) == 0:
        return data_tpl
    elif len(label_files) == 1:
        lb, spacing = read_nii(label_files[0])
        if not np.all(spacing == data_tpl['spacing']):
            raise ValueError('Found not matching spacings for case {}.'.format(case))
        data_tpl['label'] = lb
        data_tpl['raw_label_file'] = label_files[0]

    return data_tpl


def _is_im_dcm_ds(ds):
    attrs = ['pixel_array', 'ImagePositionPatient', 'PixelSpacing',
             'RescaleSlope', 'RescaleIntercept']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def _is_roi_dcm_ds(ds):
    attrs = ['StructureSetROISequence', 'ROIContourSequence']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def read_dcms(dcm_folder, reverse=True, names=None):
    '''
    read_dcms(dcms, dcmrt=None, reverse=True, names=None)

    Reads dicom files for axial images and dcmrt files that contain ROIS.
    If is assumed that both the image dcms and the dcmrt dicoms are files in
    \'dcm_folder\'. If not dcmrt files is found an empty segmentation is
    returned, else one segmentation array is created for each dcmrt file.
    Image dcms should be of axial reconstructions with the attributes
    pixel_array, ImagePositionPatient, PixelSpacing, RescaleSlope,
    RescaleIntercept, roi dcms should have the attributes
    StructureSetROISequence and ROIContourSequence

    Parameters
    ----------
    dcm_folder : str
        full path of dcm_folder to dcms to be read
    reverse : bool, optional
        For right handed coordinate systems like Siemens uses them the top
        slice has the highest z value, so in this case we sort the dcms in
        descending (reverse) order with respect to their z coordinate.
        The default is True.
    names : list
        names that are contained in the ROI file. Will be used to encode the
        ROIs in the output seg, e.g. the ROI names[0] will be encoded as 1
        in the output and so on. If the list is not given it is checked if
        all names start with a number, otherwise all occuring names is sorted
        alphabetically. Comparisson of names if case insensitive

    Returns
    -------
    image, spacing

    '''
    global _names_sorting_warning_printed, _multiple_roi_dcms_warning_printed
    # read the image and sort it with respect to the z coordinates
    dcms = [join(dcm_folder, dcm) for dcm in listdir(dcm_folder)
            if dcm.endswith('.dcm')]
    dcms.sort()
    imdss = []
    roidss = []
    roidcms = []
    for dcm in dcms:
        ds = pydicom.dcmread(dcm)
        if _is_im_dcm_ds(ds):
            imdss.append(ds)
        elif _is_roi_dcm_ds(ds):
            roidss.append(ds)
            roidcms.append(dcm)
        else:
            raise TypeError(dcm + ' is neither image nor roi dcm.')
    if len(roidss) > 1 and not _multiple_roi_dcms_warning_printed:
        print('Found multiple ROI dcms in folder '+dcm_folder+'. '
              'One segmentation is returned for each file ordered '
              'alphabetically by filename.')
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]
    imdss = [ds for _, ds in sorted(zip(z_im, imdss), reverse=reverse)]
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]

    # now get the spacing
    ps = np.array(imdss[0].PixelSpacing).astype(float)
    z_sp = np.abs(np.median(np.diff(z_im)))
    spacing = [*ps, z_sp]

    # convert the image in HU
    a = imdss[0].RescaleSlope
    b = imdss[0].RescaleIntercept
    im = np.stack([imds.pixel_array*a+b for imds in imdss],
                  -1).astype(np.int16)
    im[im < -1024] = -1024

    for roids, roidcm in zip(roidss, roidcms):
        seg = np.zeros_like(im)

        pos_r = float(imdss[0].ImagePositionPatient[1])
        pos_c = float(imdss[0].ImagePositionPatient[0])
        names_found = [s.ROIName.lower() for s in
                       roids.StructureSetROISequence]
        names_found = np.unique(names_found).tolist()
        names_found.sort()
        if names is None:
            names = names_found
        else:
            names = [name.lower() for name in names]
            for name in names_found:
                if name not in names:
                    raise ValueError('Name error in '+roidcm+'. Found ROI with'
                                     ' name '+name+' which was not given in'
                                     ' the name list.')
        # now let's look at all ROIS
        for i in range(len(roids.ROIContourSequence)):
            name = roids.StructureSetROISequence[i].ROIName.lower()
            num = names.index(name)
            for s in roids.ROIContourSequence[i].ContourSequence:
                c = s.ContourData
                # list of polygone corners
                nodes = np.array(c).reshape((-1, 3))
                ad = np.abs(z_im-nodes[0, 2])
                # z index of the slice the contour is marked in
                z_index = np.argmin(np.abs(ad))
                if np.max(ad) > 0.1:
                    ValueError('z axis of difference larger than .1mm found'
                               ' between dcmrt and dcms.')
                r = (nodes[:, 1] - pos_r) / spacing[1]
                # from patient coordinate system to index of the image
                c = (nodes[:, 0] - pos_c) / spacing[0]
                rr, cc = polygon(r, c)
                seg[rr, cc, z_index] = num

        # we check if all names start with a number
        # in that case we're converting the segmentations to carries these
        # numbers
        if np.all([name[0].isdigit() for name in names]):
            nums = []
            for name in names:
                i = 1
                while name[:i].isdigit() and i <= len(name):
                    i += 1
                nums.append(name[:i-1])
            seg_new = np.zeros_like(seg)
            for i, num in enumerate(nums):
                seg_new[seg == i] = num
            seg = seg_new
        # if not the integers will refer to the alphabetical sorting
        elif not _names_sorting_warning_printed:
            print('Warning: Alphabetical sorting of ROI names is applied.'
                  ' This is only relevant for multiclass problems. If no '
                  'consistent names are chosen in the ROI files or if some '
                  'scans do not show all ROIs errors might appear in the '
                  'segmentations. Recommendation: hand ROI names as a list '
                  'to read_dcms or rename ROIS to start with an integer.')

    data_tpl = {}
    data_tpl['image'] = im
    data_tpl['raw_image_file'] = dcm_folder
    if len(roidcms) > 0:
        data_tpl['label'] = seg
        if len(roidcms) == 1:
            roidcms = roidcms[0]
        data_tpl['raw_label_file'] = roidcms
    data_tpl['spacing'] = spacing
    for key, attr in zip(['pat_id', 'date'], ['PatientID', 'AcquisitionDate']):
        if hasattr(ds, attr):
            data_tpl[key] = ds.__getattr__(attr)
    data_tpl['dataset'] = basename(split(dcm_folder)[0])

    return data_tpl


def save_nii(im, out_file, spacing=None, img=None):
    '''
    save_nii(im, out_file, spacing=None, img=None)

    saves image as nii file by either overwriting image data from another nii
    file or by giving the spacing and creating a new nii file

    Parameters
    ----------
    im : 3d array
        image data.
    out_file : str
        path to where the image should be stored
    spacing : len 3
        voxel spacing in mm, optional
    img : nifti image
        Image to be used for overwriting.

    Returns
    -------
    None.

    '''
    if not out_file.endswith('.nii.gz'):
        out_file = out_file + '.nii.gz'

    if spacing is None and img is None:
        raise ValueError('Voxel spacing or another nifti image must be given'
                         'as input when writing a new nifti file.')
    elif spacing is None:
        im_nii = nib.Nifti1Image(im, img.affine, img.header)
    else:
        im_nii = nib.Nifti1Image(im, np.eye(4))
        im_nii.header['pixdim'][1:4] = spacing
    nib.save(im_nii, out_file)
