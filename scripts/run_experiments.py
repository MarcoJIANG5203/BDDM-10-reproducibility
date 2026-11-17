import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.experiment import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Protocol v1 imputation and selective prediction experiments')
    parser.add_argument('--config',required=True)
    parser.add_argument('--rf',action='store_true',help='RF fixed 300 trees, S0/S2')
    parser.add_argument('--m-sensitivity',action='store_true',help='M=5/20, seed2026 LR S0')
    parser.add_argument('--alpha-sensitivity',action='store_true',help='alpha=.05 seed2026 LR M10')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    print(run(config,rf=args.rf,m_sensitivity=args.m_sensitivity,alpha_sensitivity=args.alpha_sensitivity))
