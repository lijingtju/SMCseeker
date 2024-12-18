from PIL import Image
from public.parseArgs import ParseArgs
import pandas as pd
import os
import numpy as np
import random
from tools.tools import evaluateRule
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import copy
import re
import time


class ImageEnhance():
    def __init__(self, args: ParseArgs) -> None:
        self.args = args
        self.datadir = os.path.join(args.datadir, args.dataset)
        self.dataPath = None
        fileNames = os.listdir(self.datadir)
        for fileName in fileNames:
            if "{}.".format(self.args.dataset) in fileName:
                self.dataPath = os.path.join(self.datadir, fileName)
                break
        self.fileExtension = os.path.splitext(self.dataPath)[1].lower()
        self.dfData = None
        self.dataDicts = None
        self.loadDatas()
        self.waitSaveQueue = queue.Queue()
        self.needProcessDataAmount = 0
        self.processedDataAmount = 0
        self.processedDatas = []
        self.stopFlag = False
        self.finallyAmount = 0


    def loadDatas(self):
        if self.fileExtension == '.csv':
            self.dfData = pd.read_csv(self.dataPath)
        elif self.fileExtension == '.xlsx':
            self.dfData = pd.read_excel(self.dataPath)
        else:
            raise ValueError("Unsupported file type. Only .csv and .xlsx are supported.")
        self.dataDicts = self.dfData.to_dict(orient='records')
    
    def spNoise(self, img:np.ndarray) -> np.ndarray:
        imgShape = img.shape
        assert len(imgShape) == 2 or len(imgShape) == 3, "Unsupported {} shape. Only 2 or 3 are supported.".format(imgShape)
        randMatrix = np.random.rand(imgShape[0], imgShape[1])
        if self.args.sp_noise_scale == 0:
            return np.array(Image.fromarray(img).resize((self.args.image_size,self.args.image_size)), dtype=np.uint8)
        positiveMask = randMatrix >= (1-self.args.sp_noise_scale/2)
        negativeMask = randMatrix <= self.args.sp_noise_scale/2
        
        positive = np.array([222,25,25], dtype=np.uint8)
        negative = np.array([17,99,219], dtype=np.uint8)

        if len(imgShape) == 2:
            img[positiveMask] = positive[0]  
            img[negativeMask] = negative[0]
        elif len(imgShape) == 3:
            img[positiveMask] = positive  
            img[negativeMask] = negative
        return np.array(Image.fromarray(img).resize((self.args.image_size,self.args.image_size)), dtype=np.uint8)
    
    def flipHorizontal(self, img:np.ndarray) -> np.ndarray:
        img = np.array(Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT))
        return np.array(Image.fromarray(img).resize((self.args.image_size,self.args.image_size)), dtype=np.uint8)

    def flipVertical(self, img:np.ndarray) -> np.ndarray:
        img = np.array(Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM))
        return np.array(Image.fromarray(img).resize((self.args.image_size,self.args.image_size)), dtype=np.uint8)

    def _rotation(self, img:np.ndarray, angle:float) -> np.ndarray:
        pil_img = Image.fromarray(img)
        rotated_img = pil_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        return np.array(rotated_img, dtype=np.uint8)
        # return np.array(Image.fromarray(img).resize((self.args.image_size,self.args.image_size)), dtype=np.uint8)
    
    def rotation(self, img:np.ndarray, bound:tuple) -> np.ndarray:
        angle = random.randint(a=int(bound[0]), b=int(bound[1]))
        return self._rotation(img, angle=angle)
    
    @staticmethod
    def saveImage(img:np.ndarray, savePath:str) -> None:
        Image.fromarray(img).save(savePath)
    
    def saveTask(self) -> None:
        if not os.path.exists(os.path.join(self.args.enhance_save_path, "processed")):
            os.makedirs(os.path.join(self.args.enhance_save_path, "processed"))
        startTime = time.time()
        index = 0
        with ProcessPoolExecutor(max_workers=self.args.worker) as executor:
            futures = []
            while (not self.stopFlag) or (not self.waitSaveQueue.empty()):
                if self.waitSaveQueue.empty():
                    time.sleep(0.5)
                    continue
                imgDict, img = self.waitSaveQueue.get()
                imgDict = copy.deepcopy(imgDict)
                imgIndex = self.processedDataAmount+1
                imgName = "{}.png".format(imgIndex)
                imgSavePath = os.path.join(self.args.enhance_save_path, "processed", imgName)

                future = executor.submit(ImageEnhance.saveImage, img=img, savePath=imgSavePath)
                futures.append(future)
                # self.saveImage(img, imgSavePath)

                imgDict["origin_index"] = imgDict["index"]
                imgDict["index"] = imgIndex
                imgDict["name"] = imgName
                self.processedDatas.append(imgDict)
                self.processedDataAmount += 1

                index += 1
                scale = index/self.finallyAmount
                needTime = (self.finallyAmount - index) *  ( (time.time() - startTime) / index)
                print("\r[{}/{}] {} {} | {} % 🚗 [{} s]    ".format(index, self.finallyAmount, "="*int(scale*20), " " if index == self.finallyAmount else ">",int(scale*100), int(needTime)), end='')

            for future in futures:
                future.result()


    def loadImage(self, imgName) -> np.ndarray:
        imgPath = os.path.join(self.datadir, "raw", imgName)
        return np.array(Image.open(imgPath).convert("RGB"), dtype=np.uint8)

    def strategy(self, imgDict:dict) -> None:
        index = imgDict["index"]
        imgName = imgDict["name"]
        imgDict["img_type"] = "origin"
        img = self.loadImage(imgName=imgName)
        if self.args.preserve_original_data:
            self.waitSaveQueue.put((copy.deepcopy(imgDict), img.copy()))
        if not evaluateRule(self.args.enhance_rule, imgDict):
            return
        for operation in self.args.enhance_list:
            if operation == 'flip_horizontal':
                resImg = self.flipHorizontal(img=img.copy())
                imgDict["img_type"] = "flip_horizontal"
            elif operation == 'flip_vertical':
                resImg = self.flipVertical(img=img.copy())
                imgDict["img_type"] = "flip_vertical"
            elif operation.startswith('rotation_'):
                match = re.match(r'rotation_\[(\d+)-(\d+)\]', operation)
                if match:
                    min_angle = int(match.group(1))
                    max_angle = int(match.group(2))
                    resImg = self.rotation(img=img.copy(), bound=(min_angle, max_angle))
                    imgDict["img_type"] = "{}".format(operation)
                else:
                    raise ValueError("Invalid rotation Angle: {}".format(str(match)))
            elif operation == 'sp_noise':
                resImg = self.spNoise(img=img.copy())
                imgDict["img_type"] = "sp_noise"
            else:
                raise ValueError("Invalid enhancement operation: {}".format(operation))
            self.waitSaveQueue.put((copy.deepcopy(imgDict), resImg.copy()))


    def run(self):
        waitProcessDatas = []
        self.needProcessDataAmount = len(self.dataDicts)
        for index, x in enumerate(self.dataDicts):
            if evaluateRule(self.args.enhance_rule, x):
                waitProcessDatas.append(x)
            print("\rProgress of data statistics: {}/{}".format(index+1, self.needProcessDataAmount), end='')
        print("")
        # waitProcessDatas = [x for x in self.dataDicts if evaluateRule(self.args.enhance_rule, x)]
        self.finallyAmount = len(waitProcessDatas) * (len(self.args.enhance_list)+1) if self.args.preserve_original_data else len(waitProcessDatas) * len(self.args.enhance_rule)
        self.finallyAmount += self.needProcessDataAmount - len(waitProcessDatas) if self.args.preserve_original_data else 0
        startTime = time.time()
        with ThreadPoolExecutor(max_workers=self.args.worker+1 if self.args.worker < 10 else 11) as executor:
            futures = []
            saveDataFuture = executor.submit(self.saveTask)
            for index, imgDict in enumerate(self.dataDicts):
                future = executor.submit(self.strategy, imgDict=copy.deepcopy(imgDict))
                futures.append(future)
            
            taskAmount = len(futures)
            for index, future in enumerate(futures):
                result = future.result()
            self.stopFlag = True
            saveDataFuture.result()
        
        df = pd.DataFrame(self.processedDatas)
        df.to_csv(os.path.join(self.args.enhance_save_path, "{}_processed.csv".format(self.args.dataset)), index=False)
        costTime = time.time() - startTime
        print("\n")
        print("Time spent {} s".format(int(costTime)))
        print("original data: {}".format(self.needProcessDataAmount))
        print("Should get the amount of data: {}".format(self.finallyAmount))
        print("The amount of data actually processed :{}".format(self.processedDataAmount))
        








# # 打开图片
# image = Image.open('your_image.jpg')

# # 左右翻转
# flipped_left_right = image.transpose(Image.FLIP_LEFT_RIGHT)

# # 上下翻转
# flipped_top_bottom = image.transpose(Image.FLIP_TOP_BOTTOM)

# # 保存结果
# flipped_left_right.save('flipped_left_right.jpg')
# flipped_top_bottom.save('flipped_top_bottom.jpg')

# # 显示图片
# flipped_left_right.show()
# flipped_top_bottom.show()
