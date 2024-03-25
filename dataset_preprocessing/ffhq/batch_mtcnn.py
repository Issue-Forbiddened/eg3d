import argparse
import cv2
import os
from mtcnn import MTCNN
import random
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import threading  # Make sure to import threading

# 使用锁来确保线程安全
lock = Lock()
# 使用字典来存储每个线程的MTCNN实例
detectors = {}

def get_detector():
    # Use threading.get_ident() to get the current thread ID
    thread_id = threading.get_ident()
    if thread_id not in detectors:
        with lock:
            if thread_id not in detectors:  # Double-check to avoid race conditions
                detectors[thread_id] = MTCNN()
    return detectors[thread_id]


def process_image(img_data):
    root, img = img_data
    # 获取当前线程的MTCNN检测器
    detector = get_detector()

    if 'mirror' in img:
        return

    src = os.path.join(root, img)
    out_detection = os.path.join(root, "detections")  # 修改此处，使用当前文件的根目录
    os.makedirs(out_detection, exist_ok=True)  # 确保目标目录存在

    if img.endswith(".jpg"):
        dst = os.path.join(out_detection, img.replace(".jpg", ".txt"))
    elif img.endswith(".png"):
        dst = os.path.join(out_detection, img.replace(".png", ".txt"))
    else:
        return  # 如果不是jpg或png文件，就跳过

    if not os.path.exists(dst):
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        if len(result) > 0:
            index = 0
            if len(result) > 1:  # if multiple faces, take the biggest face
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

            if result[index]["confidence"] > 0.9:
                keypoints = result[index]['keypoints']
                with open(dst, "w") as outLand:
                    outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['nose'][0])) + " " + str(float(keypoints['nose'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
            else:
                print(f"{src}: face not detected")
                # remove the file
                os.remove(src)

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
args = parser.parse_args()
in_root = args.in_root

image_data_list = []

# 使用os.walk遍历所有子目录
for root, dirs, files in os.walk(in_root):
    imgs = sorted([x for x in files if x.endswith(".jpg") or x.endswith(".png")])
    random.shuffle(imgs)
    for img in imgs:
        image_data_list.append((root, img))

# 设置线程数
num_threads = 16

# 使用ThreadPoolExecutor来实现多线程处理
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    list(tqdm(executor.map(process_image, image_data_list), total=len(image_data_list)))

# python batch_mtcnn.py --in_root /data/nersemble_free_all_extracted_frames
