import os

from baseline import run_baseline

def write_results(results, out_path):
    if not results:
        print("No results to save.")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_excel(out_path, index=False)
        print(f"Saved results to {out_path}")
    except Exception as exc:
        import csv
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {csv_path} (Excel export unavailable: {exc})")

def main():
    # 固定一些参数
    base_config = {
        "optimizer": "SGD",
        "epochs_head": 5,
        "epochs_ft": 20,
        "batch_size": 32
    }

    # 定义要搜索的超参数网格
    batch_size_list = [32, 64]
    fc_lr_list = [1e-4, 5e-4, 1e-3]
    backbone_lr_list = [1e-5, 1e-4]

    results = []

    for batch_size in batch_size_list:
        for fc_lr in fc_lr_list:
            for bb_lr in backbone_lr_list:
                # 组建本次实验的配置
                config = base_config.copy()
                config["batch_size"] = batch_size
                config["fc_lr"] = fc_lr
                config["backbone_lr"] = bb_lr

                # 自动生成一个有意义的名字
                run_name = f"SGD_bs{batch_size}_fc{fc_lr}_bb{bb_lr}"

                print(f"\n====== Starting experiment: {run_name} ======")
                summary = run_baseline(config, run_name=run_name)
                results.append(summary)

    out_path = os.path.join("outputs", "hypersearch_sgd_results.xlsx")
    write_results(results, out_path)

if __name__ == "__main__":
    main()