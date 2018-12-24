# HeadPose
### Dataset
300W, AFLW2000
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
BIWI
https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#


Lucas-Kanade Image Alignment
============================

1. [Definition](#definition)
2. [Optimization and Residuals](#optimization)
3. [Alignment and Visualization](#visualization)
4. [References](#references)
5. <a href="http://menpofit.readthedocs.io/en/stable/api/menpofit/lk/index.html">API Documentation <i class="fa fa-external-link fa-lg"></i></a>

---------------------------------------

<p><div style="background-color: #F2DEDE; width: 100%; border: 1px solid #A52A2A; padding: 1%;">
<p style="float: left;"><i class="fa fa-exclamation-circle" aria-hidden="true" style="font-size:4em; padding-right: 15%; padding-bottom: 10%; padding-top: 10%;"></i></p>
We highly recommend that you render all matplotlib figures <b>inline</b> the Jupyter notebook for the best <a href="../menpowidgets/index.md"><em>menpowidgets</em></a> experience.
This can be done by running</br>
<center><code>%matplotlib inline</code></center>
in a cell. Note that you only have to run it once and not in every rendering cell.
</div></p>

### 1. Definition {#definition}
The aim of image alignment is to find the location of a constant template $$\bar{\mathbf{a}}$$ in an input image $$\mathbf{t}$$.
Note that both $$\bar{\mathbf{a}}$$ and $$\mathbf{t}$$ are vectorized.
This alignment is done by estimating the optimal parameters values of a parametric motion model. The motion model consists of a Warp functiond
