import subprocess
from pathlib import Path
import shutil


def clean_folder_files(fpath, keep=".edf"):
    keep_files = [fp for fp in fpath.glob(f"*{keep}")]
    all_files = [fp for fp in fpath.glob("*")]
    rm_files = [fp for fp in all_files if fp not in keep_files]
    [fp.unlink() for fp in rm_files]


if __name__ == "__main__":
    """
    Run this script 3 times. 
    1. The clean flags are False. The bashCmd's last flag should be "/Process", this will run the processing command
    2. The clean flags are False. The bashCmd's last flag should be "/Archive", this will archive the processed results
    3. Turn the clean_files to True. This will clean up the extra files in the folders
    """
    clean_files = True
    clean_folders = False
    if clean_files:
        clean_folder_files(Path("D:/ScalpData/persystSpikeDetection/9-1"))
    else:
        # Path to PSCLI executable
        PSCLI = "C:/Program Files (x86)/Persyst/Insight/PSCLI.exe"
        # regex path where the EDF files are
        searchStr = "D:\\ScalpData\\persystSpikeDetection\\9-1\\*.edf"

        bashCmd = [PSCLI, f"/SourceFile={searchStr}", "/FileType=EDF90", "/Archive"]
        process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
