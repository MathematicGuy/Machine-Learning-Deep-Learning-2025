import os
import subprocess
import gdown  # Ensure installed: pip install gdown

# if not os.path.exists("FPT_train.csv"):
#     gdown.download(id="1l2TtEaWrp4yieMDWE4Cmehnf5mLx3rop", output="FPT_train.csv")

if not os.path.exists("FPT_train.csv"):
    subprocess.run(["gdown", "--id", "1l2TtEaWrp4yieMDWE4Cmehnf5mLx3rop"], capture_output=True, text=True, check=True)