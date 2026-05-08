#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计AIME25文件夹下所有JSON文件的token消耗
并额外统计大小模型分别回答的次数
文件夹结构: aime25/0/0.json, aime25/1/0.json, ..., aime25/197/0.json
"""

import json
from pathlib import Path

def process_json_file(file_path):
    """处理单个JSON文件，统计token数量和回答次数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_small = 0
    total_base = 0
    count_small = 0
    count_base = 0

    for record in data:
        # 累加小模型token (num_output_tokens_small)
        small_tokens = record.get('num_output_tokens_small')
        if small_tokens is not None:
            total_small += small_tokens
            if small_tokens > 0:
                count_small += 1

        # 累加大模型token (num_output_tokens_base)
        base_tokens = record.get('num_output_tokens_base')
        if base_tokens is not None:
            total_base += base_tokens
            if base_tokens > 0:
                count_base += 1

    return total_small, total_base, count_small, count_base

def main():
    # 基础路径 - 根据实际情况修改
    base_dir = Path('router_result/aime25')
    # base_dir = Path('/path/to/your/aime25')

    # 存储每个文件夹的统计结果
    results = []
    grand_total_small = 0
    grand_total_base = 0
    grand_count_small = 0
    grand_count_base = 0

    print("开始统计token数量和回答次数...")
    print("-" * 90)

    # 遍历0-197的文件夹
    for i in range(198):
        folder_name = str(i)
        folder_path = base_dir / folder_name
        json_file = folder_path / '0.json'

        if json_file.exists():
            small_tokens, base_tokens, small_count, base_count = process_json_file(json_file)

            results.append({
                'folder': folder_name,
                'small_model_tokens': small_tokens,
                'base_model_tokens': base_tokens,
                'small_model_count': small_count,
                'base_model_count': base_count,
                'total_tokens': small_tokens + base_tokens,
                'total_count': small_count + base_count
            })

            grand_total_small += small_tokens
            grand_total_base += base_tokens
            grand_count_small += small_count
            grand_count_base += base_count

            print(
                f"文件夹 {folder_name:>3}: "
                f"小模型={small_tokens:>8,} tokens, {small_count:>4} 次 | "
                f"大模型={base_tokens:>8,} tokens, {base_count:>4} 次"
            )
        else:
            print(f"文件夹 {folder_name:>3}: 未找到文件 {json_file}")

    # 打印汇总结果
    print("\n" + "=" * 90)
    print("统计汇总:")
    print("=" * 90)
    print(f"小模型总token数   : {grand_total_small:>15,}")
    print(f"大模型总token数   : {grand_total_base:>15,}")
    print(f"总计token数       : {grand_total_small + grand_total_base:>15,}")
    print(f"小模型回答次数    : {grand_count_small:>15,}")
    print(f"大模型回答次数    : {grand_count_base:>15,}")
    print(f"总回答次数        : {grand_count_small + grand_count_base:>15,}")

    # 保存详细结果到JSON文件
    output = {
        'details': results,
        'summary': {
            'total_small_model_tokens': grand_total_small,
            'total_base_model_tokens': grand_total_base,
            'grand_total_tokens': grand_total_small + grand_total_base,
            'total_small_model_count': grand_count_small,
            'total_base_model_count': grand_count_base,
            'grand_total_count': grand_count_small + grand_count_base
        }
    }

    output_file = base_dir / 'token_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {output_file}")

    # 同时保存为CSV格式，方便Excel查看
    csv_file = base_dir / 'token_statistics.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('folder,small_model_tokens,base_model_tokens,small_model_count,base_model_count,total_tokens,total_count\n')
        for r in results:
            f.write(
                f"{r['folder']},{r['small_model_tokens']},{r['base_model_tokens']},"
                f"{r['small_model_count']},{r['base_model_count']},"
                f"{r['total_tokens']},{r['total_count']}\n"
            )
        f.write(
            f"TOTAL,{grand_total_small},{grand_total_base},"
            f"{grand_count_small},{grand_count_base},"
            f"{grand_total_small + grand_total_base},{grand_count_small + grand_count_base}\n"
        )

    print(f"CSV格式已保存到: {csv_file}")

if __name__ == '__main__':
    main()