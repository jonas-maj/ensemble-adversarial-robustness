# Installation & Usage
this codebase is built on top of the Cost-Sensitive robustness work by Xiao Zhang [found here](http://github.com/xiaozhanguva/Cost-Sensitive-Robustness).

* Install Pytorch 0.4.1: 
```text
conda update -n base conda && conda install pytorch=0.4.1 torchvision -c pytorch -y
```
* Install convex_adversarial package developed by Eric Wong and Zico Kolter
[[see details]](https://github.com/locuslab/convex_adversarial/tree/master/convex_adversarial):
```text
pip install --upgrade pip && pip install convex_adversarial==0.3.5 -I --user torch==0.4.1
```
* Install other dependencies:
```text
pip install torch waitGPU setproctitle
```

* Script for training the ensemble models:
  ```text
  ./train_models.sh
  ```

* Script for evaluating the model ensembles:
  ```text
  python3 mnist_evaluate.py
  ```

