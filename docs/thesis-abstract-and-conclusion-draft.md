# 摘要、关键词与结论短版草案

## 中文摘要

针对车辆目标检测、车牌定位与车牌字符识别在复杂场景下存在流程分散、系统联调困难以及识别精度不稳定等问题，本文设计并实现了一套基于深度学习的车辆类型与车牌检测识别系统。系统面向 PC 端应用场景，采用“YOLO 检测 + PaddleOCR 识别”的总体技术路线，完成了车辆检测、车牌检测、车牌字符识别、结果可视化以及 JSON/CSV 结果导出等功能。

在系统实现上，本文将车辆检测与车牌检测拆分为两个独立检测模块，并通过统一流水线完成目标匹配、车牌裁剪、OCR 识别与结果组织。车牌检测部分基于 CCPD 数据集完成了 quick 和 mvp 两组训练方案的对比实验，最终选用扩展数据集训练得到的 mvp 模型作为系统主检测方案。实验结果表明，该模型在测试集上的最优 mAP@0.5:0.95 达到 0.78672，相较 quick 基线的 0.77963 有一定提升。

在 OCR 部分，本文首先对基于规则的稳定化策略进行了小样本验证，但结果表明该策略未能带来整体识别精度提升。基于此，本文进一步构建了 CCPD 车牌识别训练集，并对 PaddleOCR 的 PP-OCRv5_mobile_rec 模型进行了车牌任务微调。完整测试集评估结果表明，车牌专用 PaddleOCR 将整牌完全正确率从 2.52% 提升到 14.80%，字符级准确率从 31.32% 提升到 38.28%。同时，系统已完成图片与视频场景下的远端联调验证，能够稳定输出标注图像、标注视频以及结构化结果文件。

研究结果表明，本文所实现的系统已经具备较完整的端到端处理能力，能够满足毕业设计中“能运行、能演示、能进行实验验证”的要求。本文工作更偏向应用型系统实现与实验验证，在算法创新性和实验深度上仍有进一步提升空间，但在当前课题范围内已实现了较为完整的工程闭环。

## 关键词

车辆类型检测；车牌检测；车牌识别；YOLO；PaddleOCR

## English Abstract

To address the problems of fragmented processing flow, difficult system integration, and unstable recognition accuracy in complex scenes, this thesis designs and implements a deep-learning-based vehicle type and license plate detection and recognition system. The system is oriented to PC-side application scenarios and adopts the overall technical route of YOLO-based detection and PaddleOCR-based recognition. It supports vehicle detection, license plate detection, plate character recognition, result visualization, and JSON/CSV result export.

At the system implementation level, vehicle detection and license plate detection are separated into two independent detector modules, while a unified pipeline is used to complete target matching, plate cropping, OCR recognition, and result organization. For the license plate detection task, two training schemes, namely quick and mvp, were compared on CCPD-based data, and the mvp model trained on the expanded dataset was selected as the final detection model. Experimental results show that the best mAP@0.5:0.95 of this model reached 0.78672, which is slightly higher than the quick baseline result of 0.77963.

For the OCR task, a rule-based stabilization strategy was first evaluated on a small sample set, but the results showed that it did not improve overall recognition accuracy. Based on this finding, a CCPD-based license plate recognition dataset was further constructed, and PaddleOCR PP-OCRv5_mobile_rec was fine-tuned for the license plate recognition task. The evaluation results on the full test set show that the specialized PaddleOCR model improved the full-plate exact accuracy from 2.52% to 14.80%, and the character-level accuracy from 31.32% to 38.28%. In addition, remote end-to-end validation on both image and video inputs verified that the final system can stably generate annotated outputs and structured result files.

The results indicate that the implemented system has formed a relatively complete end-to-end processing pipeline and satisfies the graduation project requirements of being runnable, demonstrable, and experimentally verifiable. This work is mainly application-oriented and engineering-driven. Although there is still room for improvement in algorithmic novelty and experimental depth, a complete engineering loop has been achieved within the scope of this project.

## English Keywords

Vehicle type detection; License plate detection; License plate recognition; YOLO; PaddleOCR

## 结论短版

本文完成了一套面向 PC 端的基于深度学习的车辆类型与车牌检测识别系统，实现了车辆检测、车牌检测、OCR 识别、可视化展示以及 JSON/CSV 结果导出等功能。实验结果表明，扩展数据集训练得到的车牌检测模型在 mAP@0.5:0.95 指标上优于 quick 基线；在 OCR 部分，早期基于规则的稳定化策略未体现出整体精度提升，而后续基于 CCPD 裁剪车牌数据微调得到的专用 PaddleOCR 在完整测试集上显著提高了整牌完全正确率和字符级准确率。最终系统已完成图片与视频场景下的联调验证，能够满足毕业设计对系统实现、实验分析和演示展示的基本要求。
