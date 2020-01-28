import os


def load_cuda_env(version='9.0', cudnn_path=None):
    """
    设置CUDA环境变量
    :param version: CUDA版本号
    :param cudnn_path: cudnn位置
    :return:
    """
    # 检查是否设置环境
    # IS_SET_CUDA = 0:未设置 | 1:已设置
    if 'IS_SET_CUDA' in os.environ and os.environ['IS_SET_CUDA'] == '1':
        return
    print("Initialize CUDA environment... ", end='')

    # 生成CUDA目录
    nvidia_gpu_computing_toolkit = os.environ['ProgramFiles'] + r"\NVIDIA GPU Computing Toolkit"
    current_cuda_path = nvidia_gpu_computing_toolkit + r"\CUDA\v" + version
    try:
        assert os.path.isdir(current_cuda_path + r"\bin") is True
        assert os.path.isdir(current_cuda_path + r"\extras") is True
        assert os.path.isdir(current_cuda_path + r"\include") is True
        assert os.path.isdir(current_cuda_path + r"\lib") is True
        assert os.path.isdir(current_cuda_path + r"\libnvvp") is True
    except AssertionError:
        print(f"CUDA目录“{current_cuda_path}”不合格！")
        exit(1)
    # end try

    # 设置CUDA环境变量
    os.environ['CUDA_PATH'] = current_cuda_path
    os.environ['CUDA_PATH_V' + version.replace('.', '_')] = current_cuda_path

    # 检查cudnn位置
    if cudnn_path is None:
        current_cudnn_path = current_cuda_path + r"\cuda"
    else:
        current_cudnn_path = cudnn_path
    # end if
    try:
        assert os.path.isdir(current_cudnn_path + r"\bin") is True
        assert os.path.isdir(current_cudnn_path + r"\include") is True
        assert os.path.isdir(current_cudnn_path + r"\lib") is True
    except AssertionError:
        print(f"CUDNN目录“{current_cudnn_path}”不合格！")
        exit(2)
    # end try

    # 设置PATH
    os.environ['PATH'] = r"%s\bin;%s\bin;%s\extras\CUPTI\libx64;%s\include;%s\libnvvp;%s" % (
        current_cudnn_path,
        current_cuda_path, current_cuda_path, current_cuda_path, current_cuda_path,
        os.environ['PATH']
    )
    os.environ['IS_SET_CUDA'] = '1'
    print("Finished.")


def set_log_level(level=1):
    """
    屏蔽通知
    :param level: 0:不屏蔽 | 1:屏蔽通知 | 2:屏蔽警告 | 3:屏蔽错误
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def force_use_cpu():
    """
    强制使用CPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 隐藏GPU


def gpu_first():
    """
    优先使用GPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def main(idea_app_path):
    # 加载环境
    load_cuda_env()
    force_use_cpu()
    set_log_level()
    # 启动IDEA
    os.system(rf'start "" "{idea_app_path}"')


if __name__ == '__main__':
    import sys
    # 通过批处理传进来一个参数，指定IDEA位置
    main(sys.argv[1])
