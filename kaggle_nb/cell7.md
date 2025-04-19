# Create fmutils.py module for file operations
%%writefile fmutils.py
import os

class fmutils:
    @staticmethod
    def get_all_files(path):
        """Get all files in a directory"""
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, returning empty list.")
            return []
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    @staticmethod
    def get_basename(path):
        """Get the basename of a path"""
        return os.path.basename(path)
    
    @staticmethod
    def numericalSort(value):
        """Helper function for numerical sorting"""
        parts = value.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return value 