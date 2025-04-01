from trainer import Trainer
from params import TrainerParams
import sys,argparse
import json
import toml
import yaml
import csv

def clean_arguments(parser):
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.json'):
                config_params = json.load(f)
            elif args.config.endswith('.toml'):
                config_params = toml.load(f)
            elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config_params = yaml.safe_load(f)
            elif args.config.endswith('.csv'):
                reader = csv.reader(f)
                config_params = {rows[0]: rows[1] for rows in reader}
            else:
                raise ValueError("Unsupported config file format. Use JSON, TOML, YAML, or CSV.")
        for key, value in config_params.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
                # Update args with default values from TrainerParams if not provided
                default_params = TrainerParams()
                for field in default_params.__dataclass_fields__:
                    if not hasattr(args, field) or getattr(args, field) is None:
                        setattr(args, field, getattr(default_params, field))
    else:
        # If no config file is provided, use default parameters from TrainerParams
        default_params = TrainerParams(args.data)
        for field in default_params.__dataclass_fields__:
            # Use the value from the parser if provided, otherwise use the default
            if hasattr(args, field) and getattr(args, field) is not None:
                setattr(default_params, field, getattr(args, field))
            else:
                setattr(args, field, getattr(default_params, field))
        config_params = vars(args)
    return args,default_params

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model using the Trainer class.")
    parser.add_argument('function', type=str, choices=['train', 'predict', 'embed', 'explain', 'fewshot', 'zeroshot'], help="Function to execute.")
    parser.add_argument("--config", type=str, help="Path to a configuration file (JSON, TOML, YAML, or CSV).")
    parser.add_argument("--data", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--kind", type=str, default='ChemGPT-19M', help="Type of pretrained model to use.")
    parser.add_argument("--notation", type=str, choices=["selfies", "smiles"], default='selfies', help="Molecular notation to use.")
    parser.add_argument("--target", type=str, default='measured_log_sol', help="Target column in the dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the training data.")
    parser.add_argument("--name", type=str, default='Benchmark', help="Name for saving the model and results.")
    parser.add_argument("--lora", type=str, help="Whether to use LoRA fine-tuning.")
    parser.add_argument("--r", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout rate.")
    parser.add_argument("--gpu", action="store_true", help="Whether to use GPU for training.")
    parser.add_argument("--lr", type=float, default=0.006829907679948146, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs for training.")
    parser.add_argument("--weight_decay", type=float, default=1.30e-05, help="Weight decay for optimizer.")
    parser.add_argument("--loss", type=str, default='MAE', help="Loss function to use.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size.")
    parser.add_argument("--quantization_bits", type=int, default=4, help="Number of quantization bits.")
    parser.add_argument("--nn", type=str, default='v1', help="Neural network version.")
    parser.add_argument("--clip_gradient", action="store_true", help="Whether to clip gradients.")
    parser.add_argument("--adapter_type", type=str, default="houlsby", help="Type of adapter to use.")
    parser.add_argument("--reduction_factor", type=int, default=16, help="Reduction factor for adapters.")
    parser.add_argument("--encoder_only", action="store_true", help="Whether to use encoder-only architecture.")
    parser.add_argument("--num_virtual_tokens", type=int, default=10, help="Number of virtual tokens.")
    parser.add_argument("--encoder_hidden_size", type=int, default=512, help="Encoder hidden size.")
    parser.add_argument("--freeze_base", action="store_true", help="Whether to freeze the base model.")
    parser.add_argument("--train_whole_model", action="store_true", help="Whether to train the whole model.")
    parser.add_argument("--mera_final_state", type=str, default='last', help="Final state for MERA.")
    parser.add_argument("--min_ciat_loss", action="store_true", help="Whether to minimize CIAT loss.")

    args,params = clean_arguments(parser)

    trainer = Trainer(params = params)

    if args.function == 'train':
        trainer.train()
    elif args.function == 'predict':
        trainer.predict()
    elif args.function == 'embed':
        trainer.embed()
    elif args.function == 'explain':
        trainer.explain()
    elif args.function == 'fewshot':
        print("Few-shot training is not implemented in this version.")
        sys.exit(1)
    elif args.function == 'zeroshot':
        print("Zero-shot training is not implemented in this version.")
        sys.exit(1)
    else:
        print(f"Unknown function '{args.function}'. Please choose from 'train', 'predict', 'embed', 'explain', 'fewshot', or 'zeroshot'.")
        sys.exit(1)
