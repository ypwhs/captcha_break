## Windows下pytorch多进程可能出现的问题

如果直接在WIndows下jupyter notebook直接运行ctc_pytorch.ipynb可能会出现如下的报错

```
> Runtime Error with DataLoader（PID XXX）: exited unexpectedly
```

即issue27 https://github.com/ypwhs/captcha_break/issues/27

## 简单的解决方法


 - 单进程处理

这个问题在github pytorch中也有人遇到，给出的一个简单的解决方法是将num_workwes=0，那么就不会使用多进程进行处理 可以运行但是速度太慢。
https://github.com/pytorch/pytorch/issues/5301

```python
train_loader = DataLoader(dataset=dataset,
batch_size=100,
shuffle=True,
num_workers=0) # change num_workers=0
```





## 解决方法

我把原来代码的ipynb脚本中的部分直接用py文件写了出来，直接python main.py就能运行，或者在ipynb中%run main.py
参考https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac
具体做了这样两件事

 - 把多进程的函数单独写在另一个py文件中，再import

 
 正如这个问题提到的https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
 
 直接在当前py中上文定义的函数用multiprocessing处理可能会出现问题
 
 而Dataloder的num_workers就是用multiprocessing实现的。
 
 - 在调用work，即该项目的train 和valid时，添加语句if \_\_name__=='\_\_main__':
 
 因为Windows的多进程并没有Linux下os.fork()这样的函数
 参考https://docs.python.org/2/library/multiprocessing.html?highlight=process#windows



