from huggingface_hub import snapshot_download

def download_model(
    model_id: str ,
    save_path: str ,
    resume: bool = True,
):
    """
    通过镜像站下载Hugging Face模型
    
    参数：
    model_id - 模型标识符，格式为"用户名/模型名" [3,6](@ref)
    save_path - 本地保存路径（支持相对/绝对路径）
    resume - 是否启用断点续传[3,6](@ref)
    """
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            resume_download=resume,
            local_dir_use_symlinks=False,  # 禁用符号链接直接存储文件[3,6](@ref)
            endpoint='https://hf-mirror.com'
        )
        print(f"✅ 模型已下载至：{path}")
        return path
    except Exception as e:
        print(f"❌ 下载失败：{str(e)}")
        return None

if __name__ == "__main__":
    # 示例：下载DeepSeek-R1模型
    # download_model(
    #     model_id="BAAI/bge-reranker-v2-m3",
    #     save_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\EM\bge-reranker-v2-m3",
    # )
    # download_model(
    #     model_id="BAAI/bge-m3",
    #     save_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\EM\bge-m3",
    # )
    download_model(
        model_id="opencsg/smoltalk-chinese",
        save_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw",
    )