# data_loader.py

import os
import csv
import json
import glob
from dateutil.parser import parse

def load_worker_quality(worker_csv_path):
    """
    读取 worker_quality.csv，返回 {worker_id: quality(0~1)}。
    """
    worker_quality = {}
    with open(worker_csv_path, "r", newline="") as fin:
        reader = csv.reader(fin)
        for line in reader:
            try:
                wid = int(line[0])
                q = float(line[1])
            except (ValueError, IndexError):
                print(f"[WARN] 无效 worker_quality 行：{line}, 跳过")
                continue
            if q > 0.0:
                worker_quality[wid] = q / 100.0
    print(f"Loaded {len(worker_quality)} workers")
    return worker_quality


def load_project_info(project_list_csv,
                      project_dir,
                      entry_dir):
    """
    读取 project_list.csv、data/project/ 下所有 project_*.txt
    以及 data/entry/ 下对应 project 的 entry_*.txt，
    返回 (project_info, entry_info)。

    project_info: {
       project_id: {
          sub_category, category, entry_count,
          start_date, deadline, industry (int)
       }, ...
    }
    entry_info: {
       project_id: {
         entry_number: {
           worker, entry_created_at (datetime)
         }, ...
       }, ...
    }
    """
    # --- 读取 project_list.csv ---
    pid_to_count = {}
    with open(project_list_csv, "r", newline="") as fin:
        for line in fin:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0]); cnt = int(parts[1])
            except ValueError:
                continue
            pid_to_count[pid] = cnt

    project_info = {}
    entry_info = {}
    industry_map = {}

    # --- 遍历 project 文件 ---
    for pid, expected_count in pid_to_count.items():
        proj_path = os.path.join(project_dir, f"project_{pid}.txt")
        if not os.path.isfile(proj_path):
            print(f"[WARN] project 文件缺失，跳过：{proj_path}")
            continue

        with open(proj_path, "r") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] 解析 JSON 失败，跳过：{proj_path}")
                continue

        # 提取项目元信息
        try:
            sub_cat  = int(meta["sub_category"])
            cat      = int(meta["category"])
            ecount   = int(meta["entry_count"])
            start_dt = parse(meta["start_date"])
            deadline = parse(meta["deadline"])
            ind_str  = meta.get("industry", "UNKNOWN")
        except Exception as e:
            print(f"[WARN] project_{pid}.txt 字段缺失或格式错误：{e}, 跳过")
            continue

        # 行业映射
        if ind_str not in industry_map:
            industry_map[ind_str] = len(industry_map)
        ind_id = industry_map[ind_str]

        project_info[pid] = dict(
            sub_category=sub_cat,
            category=cat,
            entry_count=ecount,
            start_date=start_dt,
            deadline=deadline,
            industry=ind_id
        )

        # --- 遍历该 project 对应的所有 entry 文件 ---
        entry_info[pid] = {}
        pattern = os.path.join(entry_dir, f"entry_{pid}_*.txt")
        files = glob.glob(pattern)
        if not files:
            print(f"[WARN] 未发现任何 entry 文件：{pattern}")
            continue

        for ef in files:
            with open(ef, "r") as f_ent:
                try:
                    ent_json = json.load(f_ent)
                except json.JSONDecodeError:
                    print(f"[WARN] 解析失败，跳过：{ef}")
                    continue

            results = ent_json.get("results", [])
            if not isinstance(results, list):
                print(f"[WARN] results 非列表，跳过文件：{ef}")
                continue

            for item in results:
                # 容错读取关键字段
                try:
                    e_num = int(item["entry_number"])
                    w_id  = int(item["worker"])
                    t_str = item["entry_created_at"]
                    e_dt  = parse(t_str)
                except KeyError as ke:
                    print(f"[WARN] entry 字段缺失 {ke}，跳过条目 in {ef}")
                    continue
                except Exception as e:
                    print(f"[WARN] entry 数据格式错误 {e}，跳过条目 in {ef}")
                    continue

                entry_info[pid][e_num] = dict(
                    worker=w_id,
                    entry_created_at=e_dt
                )

    print(f"Loaded {len(project_info)} projects; entry_info keys: {list(entry_info.keys())[:5]}...")
    return project_info, entry_info
