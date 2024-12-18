import argparse
import os
from tools.tools import inferType


class ParseArgs():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Script-related parameters for VirusImage")
        self.parser.add_argument("--in", type=str, default=None, help="Configuration file of the script")

        self.scripts = ['finetune', 'evaluate', 'image_enhance']

        # general args
        self.parser.add_argument("--datadir", type=str, default=None, help="Root directory of data that needs to be trained(or finetuning) and evaluated.")
        self.parser.add_argument("--dataset", type=str, default=None, help="A folder of data that needs to be trained(or finetuning) and evaluated is the name of the data. There should be a corresponding csv file under the folder, named after train, test, val, and evaluate. eg. H1N1 -> train.csv test.csv val.csv evaluate.csv")
        self.parser.add_argument("--gpu", type=str, default='0', help="Index of GPU to use.")
        self.parser.add_argument("--script", type=str, default=None, choices=self.scripts, help="Scripts related to tasks that need to be executed.")
        # self.parser.add_argument("--csv_split_symbol", type=str, default=',', help="")
        self.parser.add_argument("--worker", type=int, default=1, help="The number of threads executing tasks, used for data processing and some tasks that can be parallelized.")
        self.parser.add_argument("--image_size", type=int, default=224, help="Image size")
        self.parser.add_argument("--log_dir", type=str, default=None, help="Log directory.")

        # enhance script
        self.parser.add_argument("--enhance_list", type=str, default=None, help="The direction that needs to be enhanced. eg. enhance1,enhance2,...")
        self.parser.add_argument("--enhance_save_path", type=str, default=None, help="The location where the enhanced results need to be saved.")
        self.parser.add_argument("--preserve_original_data", type=bool, default=True, help="Whether to retain the original data in the enhanced results.")
        self.parser.add_argument("--enhance_rule", type=str, default=None, help="The data needs to be correct to meet what requirements.")
        self.parser.add_argument("--sp_noise_scale", type=lambda x: float(x) if 0 <= float(x) <= 1 else argparse.ArgumentTypeError(f"{x} is an invalid value. It must be between 0 and 1."), default=0.3, help="The ratio of noise to be added, [0-1].")


        # finetune script
        self.parser.add_argument("--resume", type=str, default=None, help="The path of the pre-trained model.")
        self.parser.add_argument("--only_save_best", type=bool, default=False, help="Save only the best models.")
        self.parser.add_argument("--save_model_dir", type=str, default=None, help="Model save directory")
        self.parser.add_argument("--model_type", type=str, default=None, choices=["resnet18AndMultiHeadAttention"], help="The model network needs to be loaded.")
        self.parser.add_argument("--loss_function", type=str, default=None, help="Loss function used.")
        self.parser.add_argument("--batch", type=int, default=64, help="batch")
        self.parser.add_argument("--epoch", type=int, default=50, help="Number of ep rounds to train.")
        self.parser.add_argument("--start_epoch", type=int, default=0, help="Start training ep.")
        self.parser.add_argument("--early_stop", type=int, default=20, help="Number of ep rounds stopped early.")
        self.parser.add_argument("--num_classes", type=int, default=1, help="The output dimension of the last linear layer of the model.")
        self.parser.add_argument("--expand_test_data_dir", type=str, default=None)
        self.parser.add_argument("--eval_threshold", type=float, default=None)
        self.parser.add_argument("--output_result", type=str, default=None)
        self.parser.add_argument("--default_save_loss_log", type=bool, default=False)
        self.parser.add_argument("--model_flag", type=str, default=None)
        self.parser.add_argument("--mlp_feature", type=int, default=2048)

        # exp loss
        self.parser.add_argument("--exp_loss_t", type=float, default=0.2)
        self.parser.add_argument("--exp_loss_alpha", type=float, default=6)
        self.parser.add_argument("--exp_loss_clamp_min", type=float, default=-10)
        self.parser.add_argument("--exp_loss_clamp_max", type=float, default=10)

        # optimizer
        self.parser.add_argument('--lr', default=0.0001, type=float, help='learning rate (default: 0.01)')
        self.parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

        # split
        self.parser.add_argument('--split_save_dir', type=str, default=None, help="The directory where the shred data is saved.")
        self.parser.add_argument("--split_scale", type=str, default="7,1.5,1.5", help="The proportion of the data that is shred.")
        self.parser.add_argument('--split_rule', type=str, default=None, help="Shard data rules, how not to use None.")

        # evalate
        

        self.argKeys = self.argsToAttr()
        self.parseArgs()

    def argsToAttr(self) -> list:
        args = self.parser.parse_args()
        argKeys = []
        for key, value in vars(args).items():
            setattr(self, key, value)
            argKeys.append(key)
        return argKeys


    def parseArgs(self) -> None:
        if self["in"] is None:
            pass
        else:
            self.parseConfigFile()
        for key in self.argKeys:
            if isinstance(self[key], list):
                for idx, v in enumerate(self[key]):
                    self[key][idx] = inferType(v)
            elif isinstance(self[key], str):
                self[key] = inferType(self[key])
            else:
                pass

    # If --in is set, the corresponding configuration file is parsed.
    def parseConfigFile(self) -> None:
        assert os.path.exists(self["in"]) == True, "{} is not a file.".format(self["in"])
        def getToken(line:str):
            line = line.replace("\n", "").replace("\r", "")
            for i, x in enumerate(line):
                if x != " ":
                    line = line[i:]
                    break
            i = len(line)
            while i > 0:
                if line[i-1] != " ":
                    line = line[:i]
                    break
                i -= 1
                if i == 0:
                    line = ""
            if len(line) == 0:
                return None, None
            
            index = line.find(" ")
            if index <= 0:
                return line, None
            elif index > 0:
                return line[0:index], line[index+1:]
        with open(self["in"], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            token, value = getToken(line=line)
            if token is None and value is None:
                continue
            elif value is None:
                self[token] = True
            else:
                if token == 'enhance_list' or token == 'split_scale' or token == 'output_result':
                    value = value.split(",")
                    value = [x for x in value if len(x) > 0]
                self[token] = value



    def __str__(self) -> str:
        argStr = "{\n"
        for key in self.argKeys:
            argStr += "    {} : {},\n".format(key, self[key])
        argStr += "}"
        argStr = argStr.replace(",\n}", "\n}")
        return argStr
    

    def __getitem__(self, key) -> None:
        if key in self.argKeys:
            return getattr(self,key)
        else:
            raise KeyError(f"Key '{key}' not found")
        
    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)
        if key not in self.argKeys:
            self.argKeys.append(key)
        

