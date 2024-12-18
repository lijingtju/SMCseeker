from public.parseArgs import ParseArgs
from models.resnet import resnet18AndMultiHeadAttention, ResNet, resnet18MM, resnet18AndMultiHeadAttention2Feature
import torch
import os
from utils.imageDataLoader import ImageDataset
from utils.expMseLoss import ExpMseLoss
from utils.bmseLoss import BMCLoss
import torchvision.transforms as transforms
import pandas as pd
from typing import List, Dict
from tools.tools import evaluateRule, flatten
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup_distributed(args):
    dist.init_process_group(
        backend='nccl',           
        init_method='env://',      
        world_size=args["world_size"], 
        rank=args["rank"]             
    )
    torch.cuda.set_device(args.local_rank) 

class Finetune():
    def __init__(self, args:ParseArgs) -> None:
        torch.backends.cudnn.enabled = False

        self.finetuneTime = int(time.time())
        self.args = args
        self.device, self.device_ids = self.setup_device(n_gpu_use=self.args.gpu)
        self.model = self.loadModel()
        if self.args.loss_function == "mse":
            self.lossFunction = torch.nn.MSELoss()
        elif self.args.loss_function == "expMseLoss":
            self.lossFunction = ExpMseLoss(t=self.args.exp_loss_t, alpha=self.args.exp_loss_alpha, clamp_min=self.args.exp_loss_clamp_min, clamp_max=self.args.exp_loss_clamp_max)
        elif self.args.loss_function == "BMC":
            self.lossFunction = BMCLoss(init_noise_sigma=8.0, device=self.deviceen)
        else:
            raise ValueError("Unsupported loss function: {}".format(self.args.loss_function))
        self.save_model_dir = os.path.join(self.args.save_model_dir, self.args.dataset,"{}_{}_{}".format(self.args.model_flag, self.args.dataset, self.finetuneTime))
        self.log_dir = os.path.join(self.args.log_dir, self.args.dataset)
        self.logName = "{}_{}_{}".format(self.args.model_flag, self.args.dataset, self.finetuneTime)
        self.log_path = os.path.join(self.log_dir, "{}.csv".format(self.logName))
        self.log_info_path = os.path.join(self.log_dir, "{}.txt".format(self.logName))
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Save loss function logs
        self.lossLogPath = os.path.join(self.log_dir, "loss_{}.csv".format(self.logName))
        if self.args.default_save_loss_log:
            lossLogTableHead = "EPOCH,STEP,LOSS,PPV,TP,FP,TN,FN\n"
            with open(self.lossLogPath, 'w', encoding='utf-8') as f:
                f.write(lossLogTableHead)

        self.img_transformer = [transforms.CenterCrop(args.image_size), transforms.ToTensor()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        
        self.train_dataset = ImageDataset(datas=self.loadDataInfos("train"), datadir=os.path.join(self.args.datadir, self.args.dataset), folder="train", img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)
        self.val_dataset = ImageDataset(datas=self.loadDataInfos("val"), datadir=os.path.join(self.args.datadir, self.args.dataset), folder="val", img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)
        self.test_dataset = ImageDataset(datas=self.loadDataInfos("test"), datadir=os.path.join(self.args.datadir, self.args.dataset), folder="test", img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.args.batch,
                                                   shuffle=True,
                                                   num_workers=self.args.worker,
                                                   pin_memory=True)
        self.train_total = len(self.train_dataloader.dataset)
        self.train_steps = len(self.train_dataloader)

        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=self.args.batch,
                                                    shuffle=False,
                                                    num_workers=self.args.worker,
                                                    pin_memory=True)
        self.val_total = len(self.val_dataloader.dataset)
        self.val_steps = len(self.train_dataloader)

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.args.batch,
                                                    shuffle=False,
                                                    num_workers=self.args.worker,
                                                    pin_memory=True)
        self.test_total = len(self.test_dataloader.dataset)
        self.test_steps = len(self.test_dataloader)
        
        self.expandDatas = []
        if self.args.expand_test_data_dir is not None:
            expandFilenames = os.listdir(self.args.expand_test_data_dir)
            for expandFilename in expandFilenames:
                tmpExpandDataset = ImageDataset(datas=self.loadDataInfos("expand", expandFilename), datadir=os.path.join(self.args.expand_test_data_dir, expandFilename), folder="raw", img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)
                tmpExpandDataLoader = torch.utils.data.DataLoader(tmpExpandDataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=self.args.worker,
                                                        pin_memory=True)
                self.expandDatas.append({"name":expandFilename, "dataloader":tmpExpandDataLoader})

        self.model = torch.nn.DataParallel(self.model)
        # setup_distributed({"world_size":3, "rank":0})
        # self.model = DDP(self.model)
        self.model = self.model.to(self.device)
        # if len(self.device_ids) > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=10 ** self.args.weight_decay,
        )


    def loadDataInfos(self, dataType:str, expand:str=None) -> List[Dict]:
        if dataType == "train":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_train.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_train.xlsx".format(self.args.dataset))
        elif dataType == "test":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_test.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_test.xlsx".format(self.args.dataset))
        elif dataType == "val":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_val.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_val.xlsx".format(self.args.dataset))
        elif dataType == "expand":
            dataPathcsv = os.path.join(self.args.expand_test_data_dir, expand, "{}.csv".format(expand))
            dataPathxlsx = os.path.join(self.args.expand_test_data_dir, expand, "{}.xlsx".format(expand))
        else:
            raise ValueError("Unsupported data type:{}".format(dataType))
        
        if os.path.exists(dataPathcsv):
            dfData = pd.read_csv(dataPathcsv)
        elif os.path.exists(dataPathxlsx):
            dfData = pd.read_excel(dataPathxlsx)
        else:
            raise ValueError("Data does not exist:{}".format(os.path.join(self.args.datadir, self.args.dataset, "{}_test".format(self.args.dataset))))
        dataDict = dfData.to_dict(orient='records')
        return dataDict

    def loadModel(self) -> ResNet:
        if self.args.model_type == "resnet18AndMultiHeadAttention":
            model = resnet18AndMultiHeadAttention(self.args.num_classes)
        elif self.args.model_type == "resnet18":
            model = resnet18MM(self.args.num_classes)
        elif self.args.model_type == "resnet18AndMultiHeadAttention2Feature":
            model = resnet18AndMultiHeadAttention2Feature(self.args.mlp_feature, self.args.num_classes)
        else:
            raise ValueError("Unsupported model network:{}".format(self.args.model_type))
        if self.args.resume is not None:
            checkpoint = torch.load(self.args.resume, weights_only=False) #, weights_only=False)
            ckp_keys = list(checkpoint['model_state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            for ckp_key in ckp_keys:
                model_sd[ckp_key] = checkpoint['model_state_dict'][ckp_key]
                
            model.load_state_dict(model_sd,)
        return model
    
    def loadLoss(self) -> torch.nn.MSELoss:
        if self.args.loss_function == "mse":
            criterion = torch.nn.MSELoss()
        else:
            raise Exception("param {} is not supported.".format(self.args.loss_function))
        return criterion

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

    def train(self, epoch:int):
        self.model.train()
        self.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        acc_loss = 0
        for step, data in enumerate(self.train_dataloader):
            # torch.cuda.empty_cache()
            images, labels, dataInfos = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            pred = self.model(images)
            labels = labels.view(pred.shape).to(torch.float64)
            loss = self.lossFunction(pred.double(), labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            accLoss = loss.detach()
            predCpu = pred.cpu()
            labelsCpu = labels.cpu()
            evalParams = {"loss":accLoss.item(), "pred":predCpu, "label":labelsCpu, "epoch":epoch, "step":step+1, "dataInfos":dataInfos}
            # print(accLoss, predCpu, labelsCpu)
            ppv, tp, fp, tn, fn = self.outputTrainInfo(params=evalParams)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            tmpLoss = accLoss.item()
            if self.args.default_save_loss_log:
                lossLogInfo = "{},{},{:.9f},{:.9f},{},{},{}\n".format(epoch, step, tmpLoss, ppv, tp, fp, tn, fn)
                with open(self.lossLogPath, 'a+', encoding='utf-8') as f:
                    f.write(lossLogInfo)
            acc_loss += tmpLoss
        print("")
        ppv = TP/(TP+FP) if TP+FP > 0 else 0
        return {"ppv":ppv, "TP":TP, "FP":FP, "TN":TN, "FN":FN}, acc_loss/self.train_steps if self.train_steps>0 else 999999999
    
    def outputTrainInfo(self, params:dict) -> dict:
        scale = params["step"] / self.train_steps
        scale = 1 if scale > 1 else scale
        ppv, tp, fp, tn, fn = self.evaluatePPV(params["pred"], params["label"])
        outputInfo = "[ep:{} loss:{:.6f} ppv:{:.2f}] {} | {}/{}       ".format(params["epoch"], params["loss"], ppv,"❄️ "*int(30*scale),params["step"], self.train_steps)
        print("\r{}".format(outputInfo), end='')
        return ppv, tp, fp, tn, fn

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
        ppv = TP/(TP+FP) if TP+FP > 0 else 0
        return {"ppv":ppv, "TP":TP, "FP":FP, "TN":TN, "FN":FN, "pred":preds, "labels":allLabels, "labelName":labelName}
        

    def outputLog(self, name:str, epoch:int, info:dict, symbol:str="🚀", isOutputPred:bool = False):
        logStr = "{} [{}] {}      PPV:{} TP:{} FP:{} TN:{} FN:{}".format("{} ".format(symbol) * 3, epoch, name, info["ppv"], info["TP"], info["FP"], info["TN"], info["FN"])
        print(logStr)
        with open(self.log_info_path, 'a+', encoding='utf-8') as f:
            f.write("{}\n".format(logStr))
        if isOutputPred:
            tmpPreds = flatten(info["pred"])
            allLabels = flatten(info["labels"])
            labelName = flatten(info["labelName"])
            resStr = ""
            for i,x in enumerate(tmpPreds):
                if resStr == "":
                    resStr = "   name:{} label:{:.2f} pred:{:.2f}".format(labelName[i], allLabels[i], x)
                else:
                    resStr += "\n   name:{} label:{:.2f} pred:{:.2f}".format(labelName[i], allLabels[i], x)
            with open(self.log_info_path, 'a+', encoding='utf-8') as f:
                f.write("      🥇🥈🥉\n{}\n      🥇🥈🥉".format(resStr))
            print("      🥇🥈🥉\n{}\n      🥇🥈🥉".format(resStr))

    def saveModel(self, ep:int, infos:List[Dict], loss:float):
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if not os.path.exists(self.log_path):
            itemCount = len(infos)
            headInfo = "EPOCH, LOSS, MODEL, LR,, NAME, PPV, TP, FP, TN, FN,,"
            for i in range(itemCount-1):
                headInfo += "NAME, PPV, TP, FP, TN, FN,,"
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write(headInfo)
        model_cpu = {k: v.cpu() for k,v in self.model.state_dict().items()}
        state = {
            'epoch': ep,
            'model_state_dict': model_cpu,
            'loss': loss,
            'lr':self.args.lr
        }
        modeName = "{}_{}_{}_{}.pt".format(self.args.dataset, ep, self.finetuneTime, int(time.time()))
        modelPath = os.path.join(self.save_model_dir, modeName)
        torch.save(state, modelPath)
        lineInfo = "\n{},{:.9f},{},{},,".format(ep, loss, modeName, self.args.lr)
        for item in infos:
            lineInfo += "{},{},{},{},{},{},,".format(item["name"], item["res"]["ppv"], item["res"]["TP"], item["res"]["FP"], item["res"]["TN"], item["res"]["FN"])
        with open(self.log_path, 'a+', encoding='utf-8') as f:
            f.write(lineInfo)

    def run(self):
        symbols = ["💒","🚢","🛳️","⛴️"]

        for ep in range(self.args.start_epoch, self.args.epoch):
            ress = []
            # train
            trainRes, acc_loss = self.train(epoch=ep)
            ress.append({"name":"train", "res":trainRes})
            self.outputLog(name="train", epoch=ep, info=trainRes, symbol="🚀")

            # eval
            valRes = self.evaluate(self.val_dataloader, ep)
            ress.append({"name":"val", "res":valRes})
            self.outputLog(name="val", epoch=ep, info=valRes, symbol="🌈")

            testRes = self.evaluate(self.test_dataloader, ep)
            ress.append({"name":"test", "res":testRes})
            self.outputLog(name="test", epoch=ep, info=testRes, symbol="🏰")

            for index, it in enumerate(self.expandDatas):
                itRes = self.evaluate(it["dataloader"], ep)
                ress.append({"name":it["name"], "res":itRes})
                self.outputLog(name=it["name"], epoch=ep, info=itRes, symbol=symbols[index%len(symbols)], isOutputPred= True if it["name"] in self.args.output_result else False )
            
            self.saveModel(ep=ep, infos=ress, loss=acc_loss)