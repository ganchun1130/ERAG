# coding:utf-8
# @File  : start_xinference.py
# @Author: ganchun
# @Date  :  2025/04/13
# @Description:
import subprocess

def start_xinference_from_windows():
    try:
        # 构建命令列表
        command = ["xinference-local", "--host", "127.0.0.1", "--port", "9997"]
        # 启动进程
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("已启动 xinference-local 服务。")
        # 获取命令的输出和错误信息
        stdout, stderr = process.communicate()

        if stdout:
            print("标准输出信息:")
            print(stdout)
        if stderr:
            print("错误输出信息:")
            print(stderr)

        # 获取命令的返回码
        return_code = process.returncode
        if return_code == 0:
            print("命令执行成功。")
        else:
            print(f"命令执行失败，返回码: {return_code}")

    except FileNotFoundError:
        print("未找到 xinference-local 命令，请确保该命令已正确安装且在系统的环境变量中。")
    except Exception as e:
        print(f"执行命令时出现未知错误: {e}")

if __name__ == "__main__":
    start_xinference_from_windows()
    # 运行成功后直接点击http://127.0.0.1:9997