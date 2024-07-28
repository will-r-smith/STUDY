import argparse
from config.load_config import load_config
from src.experiments import Experiment

if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with Roberta LLM on CounterFact')

    parser.add_argument('--test', type=str, default="intervene", choices=['intervene', 'fine_tune'])
    parser.add_argument('--model', type=str, default="roberta", choices=['pythia70m', 'roberta', 'gptj'], help="Which model to evaluate")
    
    parser.add_argument('--intervention', type=str, default="lr", choices=['dropout', 'lr', 'mm'], help="what type of intervention to perform")
    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')

    parser.add_argument('--dataset', type=str, default="counterfact", help='The dataset for analysis')
    parser.add_argument('--prop_data', type=float,default=0.002,help='The proportion of the dataset to perform evaluation with')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')

    parser.add_argument('--lname', nargs='+', default="all", help="provided which type of parameters to effect")
    parser.add_argument('--lnum', nargs='+', default="all", help='Layers to edit')

    parser.add_argument('--learning_rate', type=float, default=10, help='Learning rate for fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs in fine-tuning')

    parser.add_argument('--verbose', type=int, default=1, help='Verbosity')


    args = parser.parse_args()
    
    config = load_config()

    exp = Experiment(args, config)

    if args.test == 'intervene':
        exp.intervene()

    if args.test == 'fine_tune':
        #exp.intervene()
        exp.fine_tune()

