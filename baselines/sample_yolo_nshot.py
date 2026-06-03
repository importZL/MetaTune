"""Sample N support images from a YOLO dataset's train.txt for few-shot evaluation.

Writes:
  {out_root}/train_n{N}_s{seed}.txt  -- N image paths
  {out_root}/data_n{N}_s{seed}.yaml  -- YOLOv7-compatible data yaml pointing at it
"""
import os, argparse, random

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="root dir containing train.txt and val.txt")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--class_name", default="cell")
    args = p.parse_args()

    with open(os.path.join(args.root, "train.txt")) as f:
        all_train = [line.strip() for line in f if line.strip()]
    rng = random.Random(args.seed)
    sup = rng.sample(all_train, args.n)
    list_p = os.path.join(args.root, f"train_n{args.n}_s{args.seed}.txt")
    with open(list_p, "w") as f:
        f.write("\n".join(sup) + "\n")
    val_p = os.path.join(args.root, "val.txt")
    yml_p = os.path.join(args.root, f"data_n{args.n}_s{args.seed}.yaml")
    with open(yml_p, "w") as f:
        f.write(f"train: {list_p}\n")
        f.write(f"val: {val_p}\n")
        f.write(f"test: {val_p}\n")
        f.write(f"nc: {args.nc}\n")
        f.write(f"names: ['{args.class_name}']\n")
    print(f"wrote {list_p} ({args.n} imgs); yaml -> {yml_p}")

if __name__ == "__main__":
    main()
