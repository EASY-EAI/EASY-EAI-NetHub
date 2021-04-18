# **EASY EAI NetHub**

**EASY EAI NetHub** is a repository aiming to help build and train model.



**EASY EAI NetHub** 是一个神经网络模型的搭建及训练的仓库，旨在收纳适用于边缘端进行前向预测的模型。主要优化 模块复用、数据集迁移、回溯训练记录 、调试参数 等方面的体验。主要包含 图像分类、物体检测、图像分割、关键点定位 等模型类别。不定期更新新模型的训练配置及效果对比，通过提供模型训练脚本和经典数据集上的模型效果对比参考，方便用户参考搭建合适模型。



目前还在努力完善中，完善过程中可能会出现不少bug或缺陷，欢迎提意见、参与到完善中~



-----------------

技术交流QQ群：null

相关rk产品链接：null

---------------------



### 使用指南描述

- 此仓库为了方便模块复用，采用动态导入的形式链接各函数，[示例文件](https://github.com/EASY-EAI/EASY-EAI-NetHub/tree/master/luanch_script/config/classification_mnist.yaml)
- 动态导入的解析规则说明，[这里](https://github.com/EASY-EAI/EASY-EAI-NetHub/tree/master/luanch_script/config/解析工具说明)
- 模型适配边缘端处理。目前主要适配瑞星微硬件上的 npu 处理器。
- 主要包含 图像分类、物体检测、图像分割、关键点定位 的神经网络模型。
- 提供模型训练结果参考。
- 提供常用模块，如 [框型标注](https://github.com/EASY-EAI/EASY-EAI-NetHub/tree/master/module_lib/image_process_tool/boundingbox_tools.py)、[关键点标注](https://github.com/EASY-EAI/EASY-EAI-NetHub/tree/master/module_lib/image_process_tool/keypoint_tools.py)、[分割图标注](https://github.com/EASY-EAI/EASY-EAI-NetHub/tree/master/module_lib/image_process_tool/pixel_label_tools.py)的数据增强等功能。



### 未来计划中功能

- [ ] 训练结果可选导出所使用文件，方便后期对以往的训练结果进行追溯复现，同时亦可用作分发。

- [ ] 增加物体跟踪模型。


