import zipfile
import os

def extract_dicom(zip_paths, scan_id, output_dir):
    """
    Extracts the .dcm files corresponding to the specified scan
    
    :param zip_paths: list of paths to zip files, where .dcm files are stored.
    :param scan_id: id of the scan.
    :param output_dir: output directory where .dcm files are extracted.
    """
    extracted = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if scan_id in member and not member.endswith('/'):
                    filename = os.path.basename(member)
                    target_path = os.path.join(output_dir, filename)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    extracted += 1
    return extracted