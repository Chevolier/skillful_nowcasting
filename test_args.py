import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--foo', type=int, default=42)
parser.add_argument('--bar', type=str, default='hello')

# if input_args is not None:
#     args = parser.parse_args(input_args)
# else:

args = parser.parse_args()

# Convert Namespace to dictionary
args_dict = vars(args)

# Save dictionary as JSON file
with open('args.json', 'w') as f:
    json.dump(args_dict, f)