# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import shutil
import numpy as np


# 2026 年开始部分网络环境会拦截/降级 http，导致 404/无法访问；
# 这里改为优先 https，并提供多个镜像源兜底。
url_bases = [
    'https://yann.lecun.com/exdb/mnist/',
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    os.makedirs(dataset_dir, exist_ok=True)

    last_err = None
    for base in url_bases:
        url = base + file_name
        try:
            # 一些源需要 UA，否则会返回异常页面/错误码
                # 用自定义请求头去下载一个 URL 的内容，并把返回的数据流按二进制写入到本地文件。
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            # r：对这个请求发起连接后得到的响应对象（可读的字节流）
            with urllib.request.urlopen(req) as r, open(file_path, 'wb') as f:   # 同时打开两个“资源”并在结束时自动关闭：
                shutil.copyfileobj(r, f)   # 从 r 这个响应流里持续读取数据块，并写入到 f 文件中，直到读完
            print("Done")
            return
        except Exception as e:
            last_err = e
            # 清理可能的半成品文件，避免后续误判“已下载”
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            continue

    raise RuntimeError(f"Failed to download {file_name}. Last error: {last_err}")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    '''
    _change_one_hot_label(X) 把类别标签数组 X（例如 [5, 0, 4, ...]）转换成 one-hot 编码矩阵 T。

    '''
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1   # 取第 idx 个样本的真实类别 X[idx]（0～9），把这一行对应位置置为 1。
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()