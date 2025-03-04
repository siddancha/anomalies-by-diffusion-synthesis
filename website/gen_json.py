import numpy as np

from pathlib import Path

json_txt = ""

website_path = Path('/home/sancha/repos/diffunc/website')
home = Path(__file__).parent
folders = sorted(list(home.glob('*/')))
folders = [folder for folder in folders if folder.is_dir()]
for folder_path in folders:
    rel_folder_path = folder_path.relative_to(website_path)
    files = list(Path(folder_path).glob('*.png'))
    try:
        ori_file = [file for file in files if 'ori' in file.name][0]
        gen_file = [file for file in files if 'gen.png' in file.name][0]
        mask_file = [file for file in files if 'unc.png' in file.name][0]
    except Exception:
        import pdb; pdb.set_trace()

    json_txt += \
f"""{{
  \"input\":
    \"{ori_file.relative_to(website_path)}\",
  \"edited\":
    \"{gen_file.relative_to(website_path)}\",
  \"anomaly_mask":
    \"{mask_file.relative_to(website_path)}\",
  \"caption":
    \"TODO:\"
}},
"""

print(json_txt)
    