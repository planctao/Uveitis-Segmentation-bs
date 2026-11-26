# 论文工作稿的原生DOCX渲染核对

纳入SwAV/VICReg全部三种子结果、几何压力测试及C4聚合对照的最新Word经LibreOffice 7.3.7.2转换为26页A4 PDF。保留全部11幅实验图、8张表及5个公式图片，PDF共16个主图像对象；378个正文/表格文字块均在提取文本中找到，缺失文字块为0。原提取文件保留，文字比较仅删除项目既定页脚并规范Unicode及空白。该核对没有修改训练或原冻结预测。

全部26页生成65dpi栅格并检查缩略图；另检查第5/6/7/8/11/12/16/18/21/22/23/24/26页130dpi图，覆盖五个公式、SwAV/VICReg结果表、病例叠加、表征诊断、新增几何压力测试及C4聚合表/图。未发现最新展示图的明显裁切、公式重叠或中文缺字。密集病例图及表征诊断的嵌入标签较小或间距较密，逐例数值应结合原高分辨率图和CSV阅读；第23页在下一页图11之前保留较多空白，没有内容丢失。6.11节表格跨21/22页，续页重复表头，所有条件完整。此范围不等于每页高分辨率排版检查或外部编辑审查。

本轮检查发现旧图10左侧`double_dose`标签首字符轻微截断。已从10个哈希核验的冻结方向state重绘同一均值矩阵，逐元素最大差2.22×10⁻¹⁶，并在展示图增加保存边距。新图与来源审计位于`thesis_export/presentation_assets/`，原评价图片及数值产物保留；重新导出后复核了新图及相邻页。

本轮修正新增小表跨页只留一行的问题：最多6行的短表在Word中保持整体，新C4完整四方法/三数值种子表位于第23页；长表仍可跨页并重复表头。参考文献14的多余换行及图11的图注样式也已修正，重新导出后复核。

Word仍请求SimSun/Times New Roman；本机转换通过独立fontconfig将其替换为Noto Serif CJK SC/Liberation Serif，实际PDF字体清单另存。没有验证Microsoft Word渲染或浏览器HTML渲染；不同阅读端字体可改变分页。该PDF是工作稿，医生评分、患者独立和论文终稿质量尚未完成。

PDF路径为`outputs/severity_research_20261013/thesis_export/native_render/severity_thesis_work_draft_20261013.pdf`。自动转换与正文/图像哈希见同目录`render_manifest.json`；实际人工式视觉检查的图片、范围与限定见`visual_review.json`。导出manifest只记录Word/HTML导出阶段，原生PDF验证由独立manifest记录，避免混淆验证范围。

安装LibreOffice及字体时首次软件源索引过期导致404，更新索引后完成安装，初始错误日志保留。可重运行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/plot_severity_thesis_geometry.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/export_severity_thesis.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/render_severity_thesis.py
```

重新导出会改变输入哈希，必须重新渲染，并对新页面重新记录实际视觉检查。`render_severity_thesis.py`只自动验证结构和文字、生成页面；它不会自动宣称已完成视觉审查。
