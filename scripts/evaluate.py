from public.parseArgs import ParseArgs
from models.resnet import resnet18AndMultiHeadAttention, ResNet
import torch
import os
from utils.imageDataLoader import ImageDataset
from utils.expMseLoss import ExpMseLoss
import torchvision.transforms as transforms
import pandas as pd
from typing import List, Dict
from tools.tools import evaluateRule, flatten
import time
import os

class Evaluate():
    def __init__(self, args:ParseArgs) -> None:
        self.finetuneTime = int(time.time())
        self.args = args
        self.device, self.device_ids = self.setup_device(n_gpu_use=self.args.gpu)
        self.model = self.loadModel()

        self.evaluateModelName = os.path.basename(self.args.resume)
        self.log_dir = os.path.join(self.args.log_dir, self.args.dataset)
        self.logName = "evaluate_{}_pt_{}_{}".format(self.evaluateModelName, self.args.dataset,self.finetuneTime)
        self.log_path = os.path.join(self.log_dir, "{}.csv".format(self.logName))
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.img_transformer = [transforms.CenterCrop(args.image_size), transforms.ToTensor()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        
        self.evaluate_dataset = ImageDataset(datas=self.loadDataInfos(), datadir=os.path.join(self.args.datadir, self.args.dataset), folder="raw", img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)

        self.evaluate_dataloader = torch.utils.data.DataLoader(self.evaluate_dataset,
                                                   batch_size=self.args.batch,
                                                   shuffle=True,
                                                   num_workers=self.args.worker,
                                                   pin_memory=True)
        self.evaluate_total = len(self.evaluate_dataloader.dataset)
        self.evaluate_steps = len(self.evaluate_dataloader)




        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

    def loadDataInfos(self) -> List[Dict]:
        dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}.csv".format(self.args.dataset))
        dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}.xlsx".format(self.args.dataset))
        
        if os.path.exists(dataPathcsv):
            dfData = pd.read_csv(dataPathcsv)
        elif os.path.exists(dataPathxlsx):
            dfData = pd.read_excel(dataPathxlsx)
        else:
            raise ValueError("Data does not exist:{}".format(os.path.join(self.args.datadir, self.args.dataset, "{}_test".format(self.args.dataset))))
        dataDict = dfData.to_dict(orient='records')
        return dataDict
    
    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        return new_state_dict

    def loadModel(self) -> ResNet:
        if self.args.model_type == "resnet18AndMultiHeadAttention":
            model = resnet18AndMultiHeadAttention(self.args.num_classes)
        else:
            raise ValueError("Unsupported model network:{}".format(self.args.model_type))

        if self.args.resume is not None:
            checkpoint = torch.load(self.args.resume)
            ckp_keys = list(checkpoint['model_state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            for ckp_key in ckp_keys:
                model_sd[ckp_key] = checkpoint['model_state_dict'][ckp_key]
            model_sd = self.remove_module_prefix(model_sd)
            model.load_state_dict(model_sd)
        return model
    

    def setup_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids


    def evaluatePPV(self, preds, labels):
        preds = preds.tolist()
        labels = labels.tolist()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for p,l in zip(preds,labels):
            if p[0] >= self.args.eval_threshold and l[0] >= self.args.eval_threshold:
                TP += 1
            elif p[0] >= self.args.eval_threshold and l[0] < self.args.eval_threshold:
                FP += 1
            elif p[0] < self.args.eval_threshold and l[0] < self.args.eval_threshold:
                TN += 1
            elif p[0] < self.args.eval_threshold and l[0] >= self.args.eval_threshold:
                FN += 1
        return TP/(TP+FP) if (TP+FP) > 0 else 0, TP, FP, TN, FN

    def evaluate(self, dataloader, epoch=-1):
        self.model.eval()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        preds = []
        allLabels = []
        labelName = []
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                images, labels, dataInfos = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(images)
                labels = labels.view(pred.shape).to(torch.float64)
                predCpu = pred.cpu()
                labelsCpu = labels.cpu()
                ppv, tp, fp, tn, fn = self.evaluatePPV(preds=predCpu, labels=labelsCpu)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                preds.append(predCpu.tolist())
                allLabels.append(labelsCpu.tolist())
                labelName.append(dataInfos["name"])
                print("\r🚀🚀🚀 evlauate: {}/{}".format(step+1, self.evaluate_steps), end='')
        ppv = TP/(TP+FP) if TP+FP > 0 else 0
        print("")
        return {"ppv":ppv, "TP":TP, "FP":FP, "TN":TN, "FN":FN, "pred":preds, "labels":allLabels, "labelName":labelName}
        

    def outputLog(self, name:str, epoch:int, info:dict, symbol:str="🚀", isOutputPred:bool = False):
        print("{} [{}] {}      PPV:{} TP:{} FP:{} TN:{} FN:{}".format("{} ".format(symbol) * 3, epoch, name, info["ppv"], info["TP"], info["FP"], info["TN"], info["FN"]))
        tmpPreds = flatten(info["pred"])
        allLabels = flatten(info["labels"])
        labelName = flatten(info["labelName"])
        if isOutputPred:
            resStr = ""
            for i,x in enumerate(tmpPreds):
                if resStr == "":
                    resStr = "   name:{} label:{:.2f} pred:{:.2f}".format(labelName[i], allLabels[i], x)
                else:
                    resStr += "\n   name:{} label:{:.2f} pred:{:.2f}".format(labelName[i], allLabels[i], x)
            print("      🥇🥈🥉\n{}\n      🥇🥈🥉".format(resStr))

        CSVInfo = "MODEL,DATA_SET,PPV,TP,FP,TN,FN,,NAME,LABEL,PRED\n"
        modelName = os.path.basename(self.args.resume)
        for i,x in enumerate(tmpPreds):
            if i == 0:
                CSVInfo += "{},{},{},{},{},{},{},,{},{:.2f},{:.2f}\n".format(modelName, self.args.dataset, info["ppv"], info["TP"], info["FP"], info["TN"], info["FN"], labelName[i], allLabels[i], x)
            else:
                CSVInfo += ",,,,,,,,{},{:.2f},{:.2f}\n".format(labelName[i], allLabels[i], x)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(CSVInfo)

    def run(self):
        symbols = ["💒","🚢","🛳️","⛴️"]
        itRes = self.evaluate(self.evaluate_dataloader, -1)
        self.outputLog(name=self.args.dataset, epoch=-1, info=itRes, symbol=symbols[2], isOutputPred= True)
            