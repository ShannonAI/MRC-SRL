import argparse
import pickle
from dataloader import *
from model import *
from evaluate import *


def load_checkpoint(model_path):
    config = pickle.load(
        open(os.path.join(os.path.split(model_path)[0], 'args'), 'rb'))
    checkpoint = torch.load(model_path)
    model = MyModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return config, model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--arg_query_type", type=int,choices=[0,1,2])
    parser.add_argument("--argm_query_type", type=int,choices=[0,1])
    parser.add_argument("--gold_level", type=int, choices=[0, 1])
    parser.add_argument("--max_tokens", type=int,default=1024)
    parser.add_argument("--amp",action='store_true')
    args = parser.parse_args()

    device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config,model = load_checkpoint(args.checkpoint_path)
    test_dataloader = load_data(args.data_path, config.pretrained_model_name_or_path, config.max_tokens, False,
                                   config.dataset_tag, -1, args.gold_level, args.arg_query_type, args.argm_query_type)
    evaluation(model,test_dataloader,args.amp,device,config.dataset_tag)
