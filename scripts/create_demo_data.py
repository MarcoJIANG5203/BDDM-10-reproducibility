import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment import create_demo

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Generate synthetic demonstration, never clinical data')
    parser.add_argument('--output',default='data/synthetic')
    args=parser.parse_args()
    config=create_demo(args.output)
    import yaml
    Path('configs').mkdir(exist_ok=True)
    Path('configs/demo.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print('Synthetic data ready:',args.output)
