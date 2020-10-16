
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from glob import glob
import os
from tqdm.auto import tqdm

# save raw images need to be process
raw_data_dir = r'C:/bd_ai/dli/celeba/img_align_celeba'
# save processed images
processed_data_dir = r'C:/bd_ai/dli/celeba/celeba_processed'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#list_imgs = glob(os.path.join(raw_data_dir, "*.jpg"))
list_imgs = os.listdir(raw_data_dir)
# print(list_imgs)
mtcnn = MTCNN(margin=10, select_largest=True, post_process=False)
for imgfile in tqdm(list_imgs):
    img = plt.imread(os.path.join(raw_data_dir, imgfile))
    face = mtcnn(img)
    # print(face)
    if face is not None:
        #os.makedirs(processed_data_dir, exist_ok=True)
        face = face.permute(1, 2, 0).int().numpy()
        plt.imsave(os.path.join(processed_data_dir, imgfile), face.astype(np.uint8))
