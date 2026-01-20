
.. title::
   参考调色板

附录 A. 参考调色板
==========================

对于许多应用，感知均匀的调色板是最佳选择；即数据中相等步长在颜色空间中被感知为相等步长的调色板。
人脑感知亮度参数的变化比感知色调的变化等要更好。因此，在调色板中亮度单调递增的调色板将被观众更好地理解。

调色板通常根据其功能分为几类：

1. 顺序调色板：颜色明度和饱和度逐渐变化，通常使用单一色调；适用于表示具有排序信息的数据
2. 发散调色板：两种不同颜色的明度和饱和度变化，在中间处交汇于不饱和色；适用于所绘制信息具有关键中间值的情况，如地形图或数据围绕零值偏离时
3. 循环调色板：两种不同颜色的明度变化，在中间处交汇并在首尾处为不饱和色；适用于在端点处循环的值，如相位角、风向或一天中的时间
4. 离散调色板：用于无序分类变量、无序配对变量，有序分类变量也使用顺序调色板

一些杂项色图具有特定的用途，它们被创建就是为了这些用途。例如， ``gist.earth`` 、 ``ocean`` 和 ``terrain`` 
似乎都是为了绘制地形（绿色 / 棕色）和水深（蓝色）而创建的。 因此，我们期望在这些色图中看到发散，但是，他们出现
不相关颜色的多重转折，因此对于另一些用途不太友好。 ``cmrmap`` 的设计目的是转换成均匀的灰度，尽管它在开始处似乎有一些小的转折。
cubehelix 的设计目的是亮度和色调上平滑变化，但在绿色色调区域似乎有一个小驼峰。 ``turbo`` [1]_ 是 ``jet`` 的改进版本，
用于显示深度和密度的差异。

色觉异常
-----------------

数据可视化中的一个常见任务是用色彩比例尺或连续色彩映射来表示，通常以热图或分级统计图的形式呈现。
几种比例尺特别考虑了色盲人士的需求，并在学术界广泛应用，包括 ``cividis``、 ``viridis`` 和 ``parula``。
这些比例尺由一个从浅到暗的尺度叠加在一个从黄到蓝的尺度上，使其对所有形式的色觉都是单调且感知一致的。
一般而言，亮度图单调增的调色板对各种色觉异常都是不敏感的。

.. card-carousel:: 3

    .. card:: 正常色觉

        .. figure:: _static/images/normal.png
        
        这里使用 ``turbo`` 调色板，该调色板在除全色盲以外的人群中都是可以区分的

    .. card:: 红色视觉异常

        .. figure:: _static/images/protanomaly.png

        不完全红-绿色觉异常，绿色调看起来更红

    .. card:: 红色盲

        .. figure:: _static/images/protanopia.png

        无法区分红色和绿色

    .. card:: 绿色视觉异常

        .. figure:: _static/images/deuteranomaly.png

        不完全红-绿色觉异常，红色调看起来更绿

    .. card:: 绿色盲

        .. figure:: _static/images/deuteranopia.png

        无法区分红色和绿色

    .. card:: 三色视觉异常

        .. figure:: _static/images/tritanomaly.png

        分辨蓝色-绿色、红色-黄色的能力减退

    .. card:: 三色盲

        .. figure:: _static/images/tritanopia.png

        无法区分蓝色-绿色、紫色-红色、黄色-粉色

    .. card:: 全色盲

        .. figure:: _static/images/achromatopsia.png

        灰度视觉


参考
----------

下图展示了当前可用的内建调色板及其名称：

.. raw:: html

    <embed>
        <img class='image-light' src='_static/images/palettes-light.png'/>
        <img class='image-dark' src='_static/images/palettes-dark.png'/>
    </embed>


.. [1] https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/