from google.colab import drive
import sys
drive.mount('/content/drive', force_remount=True)
sys.path.insert(0,"/content/drive/MyDrive/Rakuten")

!pip install unidecode
!pip install fasttext
!pip install googletrans

import shutil 
#shutil.unpack_archive('/content/drive/MyDrive/Rakuten/data/raw_rebased/zipfiles/myimages.zip', '/content/drive/MyDrive/Rakuten/data/raw_rebased')

import os
_, _, files = next(os.walk("/content/drive/MyDrive/Rakuten/data/raw_rebased/images/image_train/"))
file_count = len(files)
assert file_count == 84916
file_count