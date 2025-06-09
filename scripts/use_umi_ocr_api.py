# coding:utf-8
# @File  : use_umi_ocr_api.py
# @Author: ganchun
# @Date  :  2025/04/25
# @Description:
import os
import json
import time
import requests
import argparse


def upload_file(base_url, file_path, extraction_mode="mixed"):
    """上传文件到Umi-OCR服务获取任务ID"""
    print("=======================================")
    print("===== 1. Upload file, get task ID =====")

    url = f"{base_url}/api/doc/upload"
    print("== URL:", url)

    options_json = json.dumps(
        {
            "doc.extractionMode": extraction_mode,
        }
    )

    with open(file_path, "rb") as file:
        response = requests.post(url, files={"file": file}, data={"json": options_json})
    response.raise_for_status()
    res_data = json.loads(response.text)

    # 处理文件名包含非ASCII字符的情况
    if res_data["code"] == 101:
        file_name = os.path.basename(file_path)
        file_prefix, file_suffix = os.path.splitext(file_name)
        temp_name = "temp" + file_suffix
        print("[Warning] Detected file upload failure: code == 101")
        print(f"Attempting to use temp_name {temp_name} instead of the original file_name {file_name}")

        with open(file_path, "rb") as file:
            response = requests.post(
                url,
                files={"file": (temp_name, file)},
                data={"json": options_json},
            )
        response.raise_for_status()
        res_data = json.loads(response.text)

    assert res_data["code"] == 100, f"Task submission failed: {res_data}"

    task_id = res_data["data"]
    print("Task ID:", task_id)
    return task_id


def poll_task_status(base_url, task_id):
    """轮询任务状态直到OCR任务完成"""
    print("===================================================")
    print("===== 2. Poll task status until OCR task ends =====")

    url = f"{base_url}/api/doc/result"
    print("== URL:", url)

    headers = {"Content-Type": "application/json"}
    data_str = json.dumps(
        {
            "id": task_id,
            "is_data": True,
            "format": "text",
            "is_unread": True,
        }
    )

    while True:
        time.sleep(1)
        response = requests.post(url, data=data_str, headers=headers)
        response.raise_for_status()
        res_data = json.loads(response.text)
        assert res_data["code"] == 100, f"Failed to get task status: {res_data}"

        print(
            f"    Progress: {res_data['processed_count']}/{res_data['pages_count']}"
        )
        if res_data["data"]:
            print(f"{res_data['data']}\n========================")
        if res_data["is_done"]:
            state = res_data["state"]
            assert state == "success", f"Task execution failed: {res_data['message']}"
            print("OCR task completed.")
            break


def generate_download_link(base_url, task_id, file_types=None, ignore_blank=False):
    """生成目标文件，获取下载链接"""
    print("======================================================")
    print("===== 3. Generate target file, get download link =====")

    url = f"{base_url}/api/doc/download"
    print("== URL:", url)

    if file_types is None:
        file_types = ["txt", "txtPlain", "jsonl", "csv", "pdfLayered", "pdfOneLayer"]

    # 下载文件参数
    download_options = {
        "file_types": file_types,
        # 注意拼写问题
        "ingore_blank": ignore_blank,  # 旧版使用错误的拼写
        # "ignore_blank": ignore_blank,  # 新版使用正确的拼写
        "id": task_id
    }

    headers = {"Content-Type": "application/json"}
    data_str = json.dumps(download_options)
    response = requests.post(url, data=data_str, headers=headers)
    response.raise_for_status()
    res_data = json.loads(response.text)
    assert res_data["code"] == 100, f"Failed to get download URL: {res_data}"

    download_url = res_data["data"]
    file_name = res_data["name"]
    return download_url, file_name


def download_file(url, file_name, download_dir="./download"):
    """下载目标文件"""
    print("===================================")
    print("===== 4. Download target file =====")
    print("== URL:", url)

    # 创建下载目录
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, file_name)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 下载文件大小
    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0
    log_size = 10485760  # 每10MB打印进度

    with open(download_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                if downloaded_size >= log_size:
                    log_size = downloaded_size + 10485760
                    progress = (downloaded_size / total_size) * 100
                    print(
                        f"    Downloading file: {int(downloaded_size / 1048576)}MB | Progress: {progress:.2f}%"
                    )

    print("Target file downloaded successfully: ", download_path)
    return download_path


def clear_task(base_url, task_id):
    """清理任务"""
    print("============================")
    print("===== 5. Clean up task =====")

    url = f"{base_url}/api/doc/clear/{task_id}"
    print("== URL:", url)

    response = requests.get(url)
    response.raise_for_status()
    res_data = json.loads(response.text)
    assert res_data["code"] == 100, f"Task cleanup failed: {res_data}"
    print("Task cleaned up successfully.")


def process_ocr(file_path, base_url="http://127.0.0.1:1224", extraction_mode="mixed",
                file_types=None, download_dir="./download", ignore_blank=False):
    """完成OCR流程的主函数"""
    # 1. 上传文件获取任务ID
    task_id = upload_file(base_url, file_path, extraction_mode)

    # 2. 轮询任务状态直到完成
    poll_task_status(base_url, task_id)

    # 3. 生成目标文件，获取下载链接
    download_url, file_name = generate_download_link(base_url, task_id, file_types, ignore_blank)

    # 4. 下载目标文件
    download_path = download_file(download_url, file_name, download_dir)

    # 5. 清理任务
    clear_task(base_url, task_id)

    print("======================\nProcess completed.")
    return download_path


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="Umi-OCR API调用工具")
    parser.add_argument("--file", "-f", required=True, help="要识别的PDF文件路径")
    parser.add_argument("--url", default="http://127.0.0.1: 1224", help="Umi-OCR服务URL")
    parser.add_argument("--mode", default="mixed", choices=["mixed", "accurate", "fast"],
                        help="提取模式：mixed(混合), accurate(精准), fast(快速)")
    parser.add_argument("--output", "-o", default="./download", help="下载文件保存目录")
    parser.add_argument("--ignore-blank", action="store_true", help="是否忽略空白页")
    parser.add_argument("--formats", nargs="+",
                        default=["txt", "txtPlain", "jsonl", "csv", "pdfLayered", "pdfOneLayer"],
                        help="输出文件格式列表")

    args = parser.parse_args()

    process_ocr(
        file_path=args.file,
        base_url=args.url,
        extraction_mode=args.mode,
        file_types=args.formats,
        download_dir=args.output,
        ignore_blank=args.ignore_blank
    )


if __name__ == "__main__":

    # 自定义参数示例
    process_ocr(
        file_path=r"F:\StrivingRendersMeCozy\学习使我快乐\研究生你好\研3\甘淳的大论文\学校发的\计算教育学全书PDF.pdf",
        extraction_mode="accurate",  # 精准模式
        file_types=["txt"],  # 只需要txt和带文字层PDF
        download_dir=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\raw"  # 自定义下载目录
    )