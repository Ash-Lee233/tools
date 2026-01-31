import os
import sys
import json
import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path):
    """从 config.yaml 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def scan_repo_for_symbol(repo_path: Path, symbol: str):
    """扫描给定仓库中指定符号的定义位置，返回该符号所在文件路径和行号"""
    results = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if symbol in line:
                            results.append((file_path, idx + 1, line.strip()))
    return results

def compare_symbols(pytorch_symbols, pytorch_repo, torchnpu_repo):
    """比较两个仓库中的符号，输出是否等价及判断依据"""
    comparisons = []
    for symbol in pytorch_symbols:
        pytorch_results = scan_repo_for_symbol(pytorch_repo, symbol)
        torchnpu_results = scan_repo_for_symbol(torchnpu_repo, symbol)

        comparison_result = {
            "symbol": symbol,
            "pytorch": pytorch_results,
            "torchnpu": torchnpu_results,
            "is_equivalent": False,
            "reason": ""
        }

        if pytorch_results and torchnpu_results:
            if pytorch_results == torchnpu_results:
                comparison_result["is_equivalent"] = True
                comparison_result["reason"] = "Exact match found in both repositories."
            else:
                comparison_result["reason"] = "Match found, but implementation differs."
        elif pytorch_results or torchnpu_results:
            comparison_result["reason"] = "Symbol found in only one repository."

        comparisons.append(comparison_result)

    return comparisons

def generate_excel(comparisons):
    """生成 Excel 文件，展示比较结果"""
    data = []
    for comparison in comparisons:
        for file, line, content in comparison["pytorch"]:
            data.append({
                "Symbol": comparison["symbol"],
                "Repository": "PyTorch",
                "File": file,
                "Line": line,
                "Content": content,
                "Is_Equivalent": comparison["is_equivalent"],
                "Reason": comparison["reason"]
            })
        for file, line, content in comparison["torchnpu"]:
            data.append({
                "Symbol": comparison["symbol"],
                "Repository": "TorchNPU",
                "File": file,
                "Line": line,
                "Content": content,
                "Is_Equivalent": comparison["is_equivalent"],
                "Reason": comparison["reason"]
            })

    df = pd.DataFrame(data)
    excel_path = "comparison_report.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path

def load_symbols_from_txt(txt_path):
    """从指定的 txt 文件中读取所有接口符号"""
    with open(txt_path, 'r', encoding='utf-8') as file:
        symbols = [line.strip() for line in file.readlines()]
    return symbols

def main():
    # 读取配置文件
    config = load_config('config.yaml')
    
    pytorch_repo = Path(config['pytorch_repo'])
    torchnpu_repo = Path(config['torchnpu_repo'])
    symbols_txt_path = config['symbols_txt']
    
    # 从 symbols.txt 文件加载符号
    symbols = load_symbols_from_txt(symbols_txt_path)
    
    # 比较符号
    comparisons = compare_symbols(symbols, pytorch_repo, torchnpu_repo)

    # 生成 Excel 文件
    excel_path = generate_excel(comparisons)

    print(f"Comparison complete. Report saved as {excel_path}")

if __name__ == "__main__":
    main()
