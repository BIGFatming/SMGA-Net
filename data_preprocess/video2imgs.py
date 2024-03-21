import os
import cv2
from tqdm import tqdm

num_tra_dataset = ['001', '002', '005', '006', '008',
                   '009', '012', '013', '015', '020',
                   '023', '024', '031', '032', '033',
                   '034', '035', '036']

num_val_dataset = ['004', '022', '026', '030']

gla_lu = ['glasses', 'night_noglasses', 'nightglasses', 'noglasses', 'sunglasses']

driver_tra_status = ['nonsleepyCombination', 'sleepyCombination', 'slowBlinkWithNodding', 'yawning']
driver_val_status = ['_glasses_mix', '_nightglasses_mix', '_nightnoglasses_mix', '_noglasses_mix',
                     '_sunglasses_mix']

tra_dataset_path = "/home/hym/DiskFiles/data/NTHU-DDD/Training_Evaluation_Dataset/Training Dataset/"
val_dataset_path = "/home/hym/DiskFiles/data/NTHU-DDD/Training_Evaluation_Dataset/Evaluation Dataset/"

test_dataset_path = '/home/hym/DiskFiles/data/NTHU-DDD/Testing_Dataset/test_video_mp4/'


if __name__ == '__main__':
    # for vid_name in os.listdir(test_dataset_path):
    #     print(test_dataset_path + vid_name)
    #     cap = cv2.VideoCapture(test_dataset_path + vid_name)
    #     name, ext = os.path.splitext(vid_name)
    #     if not os.path.exists(test_dataset_path + name):
    #         os.makedirs(test_dataset_path + name)
    #     elif os.path.exists(test_dataset_path + name):
    #         continue
    #     if cap.isOpened():
    #         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         for cnt in range(total):
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
    #             ret, frame = cap.read()
    #             if ret:
    #                 cv2.imwrite(test_dataset_path + name + '/' + str(cnt) + '.jpg', frame)
    #     else:
    #         continue
    #     print(test_dataset_path + vid_name + ' finished')

    for i in range(len(num_tra_dataset)):
        path = tra_dataset_path + '/' + num_tra_dataset[i]
        for j in range(len(gla_lu)):
            for k in range(len(driver_tra_status)):
                vid = path + '/' + gla_lu[j] + '/' + driver_tra_status[k] + '.avi'
                if os.path.exists(vid):
                    cap = cv2.VideoCapture(vid)
                    if not os.path.exists(path + '/' + gla_lu[j] + '/' + driver_tra_status[k]):
                        os.makedirs(path + '/' + gla_lu[j] + '/' + driver_tra_status[k])
                    elif os.path.exists(path + '/' + gla_lu[j] + '/' + driver_tra_status[k]):
                        continue
                    if cap.isOpened():
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        for cnt in range(total):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
                            ret, frame = cap.read()
                            if ret:
                                cv2.imwrite(path + '/' + gla_lu[j] + '/' + driver_tra_status[k] + '/' + str(cnt) +
                                            '.jpg', frame)
                else:
                    continue
                print(path + '/' + gla_lu[j] + '/' + driver_tra_status[k] + '.avi' + ' finished')

    for i in range(len(num_val_dataset)):
        path = val_dataset_path + '/' + num_val_dataset[i]
        for j in range(len(driver_val_status)):
            vid = path + '/' + num_val_dataset[i] + driver_val_status[j] + '.mp4'
            if os.path.exists(vid):
                cap = cv2.VideoCapture(vid)
                if not os.path.exists(path + '/' + driver_val_status[j]):
                    os.makedirs(path + '/' + driver_val_status[j])
                elif os.path.exists(path + '/' + driver_val_status[j]):
                    continue
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for k in range(total):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(path + '/' + driver_val_status[j] + '/' + str(k) + '.jpg', frame)
            else:
                continue
            print(path + '/' + num_val_dataset[i] + driver_val_status[j] + '.mp4' + ' finished')
