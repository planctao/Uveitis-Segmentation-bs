# 英文工作稿导出与实际渲染记录

英文源`docs/severity_english_manuscript_20261013.md`已独立导出为Word、离线HTML和LibreOffice 7.3.7.2原生转换的10页A4 PDF。导出保留152个源文字块、3幅实验图、4张表和2个公式；PDF核验包含新增表题在内的156个正文/表格文字块，缺失为0。5个主图像对象均在实际页面绘制指令中找到。英文稿的4表41个数值单元与归档CSV、16个引用编号另由英文审计核验。本轮没有重新训练、选择seed或改变主方法与临床锁定。

检查了全部10页70dpi缩略图，另实际查看第1/3/4/5/6/7/8/9/10页130dpi图。此范围覆盖标题摘要、两个公式、全部表图及参考文献；第2页只检查缩略图。未见明显文字/公式裁切、重叠或图例遮挡。图与图注同页，4张短表各自保持整体。公式2位于介绍段落的下一页；参考文献1–12位于第9页，13–16位于第10页，末页留白较多但没有丢失内容。

三幅英文展示图直接读取归档CSV重新排版，原图和结果保留，未重新拟合。初次图像检查发现旋转图图例遮柱、左右面板行位置不齐，修正后重新导出并核对最新PDF。展示来源行、源码和输出SHA256位于`english_manuscript/presentation_assets/audit.json`。

初次PDF自动核验报告3个表格文字块缺失。原PDF中三者均完整，问题来自`pdftotext -layout`在换行单元格间交错提取。新增独立`-raw`阅读顺序提取后，要求每个完整文字块至少在一种提取中出现；3个回退单元格及两种原始文本均归档。未通过删除单元格或缩短核验文字规避错误。图像计数采用页面实际`Do`绘制指令，避免把共享图像资源误认为每页都画了全部图。

Word请求Times New Roman，本机通过独立fontconfig将正文字体替换为Liberation Serif；实际PDF还嵌入Liberation Serif Bold和DejaVu Sans Bold，清单见`pdffonts.txt`。两个公式仍是栅格图片，不是可编辑Word公式。没有验证Microsoft Word、浏览器HTML渲染或正式投稿模板，其他机器的分页可能不同。

审阅文件位于`outputs/severity_research_20261013/english_manuscript/`：Word与HTML在根目录，PDF及栅格检查记录在`native_render/`。`export_manifest.json`、`native_render/render_manifest.json`和`native_render/visual_review.json`分别限定导出、自动渲染核验和实际视觉检查范围，均绑定当前输入/输出哈希。原中文26页稿件与导出脚本保持原版本；新增显著性结果属于独立英文工作稿和补充报告。

可在项目根目录重新执行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/plot_severity_english_figures.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/export_severity_english_manuscript.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/render_severity_english_manuscript.py
# 实际检查新页面后，重新记录与当前哈希匹配的visual_review.json。
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/audit_severity_english_manuscript.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/summarize_severity_continuation.py
```

本次完成的是可审阅英文稿的导出及排版核对。医生评分rubric、真实患者/眼别/视野/时相关联、标签临床语义和独立外部验证仍缺失，`manuscript_submission_ready=false`，研究目标继续active。面积相关与0–3分位层级不能替代临床严重程度结果。
