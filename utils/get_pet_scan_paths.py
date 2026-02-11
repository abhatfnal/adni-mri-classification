"""
Scans a large ADNI .zip archive and prints the paths to raw, high-quality
FDG-PET and AV45-PET scan directories, filtering out common utility scans.
"""
import zipfile
import sys
import re

def get_pet_scan_paths_final(zip_path):
    """
    Finds unique series directories for high-quality, raw PET scans by inspecting
    directory names and excluding known low-quality or utility scan types.
    """
    candidate_paths = set()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if member.endswith('.dcm'):
                parts = member.split('/')
                if len(parts) > 3:
                    series_path = '/'.join(parts[:3]) + '/'
                    candidate_paths.add(series_path)

    # --- Filtering Logic ---
    # Keywords for scans we WANT to process
    desired_keywords = ['FDG', 'AV45', 'FLORBETAPIR']
    # Keywords for scans we want to EXCLUDE
    undesired_keywords = [
        'EARLY', 'AC', 'ATTEN', 'CTAC', 'LOCALIZER', 
        'PHANTOM', 'SCOUT', 'DYNAMIC', 'DYN'
    ]
    
    final_paths = []
    for path in sorted(list(candidate_paths)):
        series_name = path.split('/')[2].upper()
        
        # Must contain a desired keyword
        has_desired = any(key in series_name for key in desired_keywords)
        if not has_desired:
            continue
            
        # Must NOT contain any undesired keywords
        is_undesired = any(key in series_name for key in undesired_keywords)
        if is_undesired:
            continue
            
        final_paths.append(path)
                            
    return final_paths

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_pet_scan_paths.py <path_to_zip_file>")
        sys.exit(1)
        
    zip_filepath = sys.argv[1]
    paths = get_pet_scan_paths_final(zip_filepath)
    for path in paths:
        print(path)

