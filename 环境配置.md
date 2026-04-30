## 1. 环境依赖安装

首先运行``dockerfile``建立容器，修改``run.sh``里面的路径内容

其次建立容器后执行下面的指令，实现代码的编译运行

```
cd fast_gicp
python setup.py install --user

cd ../torch-radon
python setup.py install

cd ../RINGSharp/glnet/ops
pip install -v -e .

cd ../..
python setup.py develop

export PYTHONPATH=/home/wyz/RINGSharp/:$PYTHONPATH
export PYTHONPATH=/home/wyz/RINGSharp/RINGSharp/glnet:$PYTHONPATH
export PYTHONPATH=/home/wyz/RINGSharp/RINGSharp/glnet/ops:$PYTHONPATH

git config --global --add safe.directory /home/wyz/RINGSharp

python tools/evaluate_ours_pe.py
```

**tips:**可以使用 ``git reset`` 设置代码回到原始状态
