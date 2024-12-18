from public.parseArgs import ParseArgs
from tools.imageEnhance import ImageEnhance
from tools.splitDataset import SplitDataset
from scripts.finetune import Finetune
from scripts.evaluate import Evaluate

def registerScript(args: ParseArgs) -> None:
    dontNoneList = ['datadir','dataset']
    if args.script == "image_enhance":
        dontNoneList = dontNoneList + ['enhance_list','enhance_save_path']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        
        imageEnhance = ImageEnhance(args=args)
        imageEnhance.run()
    elif args.script == "finetune":
        dontNoneList = dontNoneList + ['gpu','image_size', 'only_save_best', 'save_model_dir', 'log_dir', 'model_type', 'loss_function', 'lr', 'weight_decay', 'momentum', 'batch', 'epoch','num_classes', 'eval_threshold', 'model_flag']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        # print(args)
        finetune = Finetune(args=args)
        finetune.run()
    elif args.script == "split":
        dontNoneList = dontNoneList + ['split_save_dir', 'split_scale', 'split_rule']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        # print(args)
        splitDataset = SplitDataset(args=args)
        splitDataset.run()
    elif args.script == "evaluate":
        dontNoneList = dontNoneList + ['log_dir', 'resume']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        # print(args)
        evaluate = Evaluate(args=args)
        evaluate.run()



def main():
    args = ParseArgs()
    registerScript(args=args)


if __name__ == '__main__':
    main()



