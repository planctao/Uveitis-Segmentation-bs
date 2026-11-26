# 血管渗漏公开数据：不限疾病、器官与成像模态

日期：2026-11-01（Asia/Shanghai）。根据补充要求，检索范围扩展到非葡萄膜炎、非眼底的血管渗漏、造影剂外渗、血脑屏障通透性与显微成像。这里“相关”与“可直接评价现有分割模型”分别说明，公开原图不等于提供人工渗漏掩码。

## 最值得考虑的资源

| 数据 | 疾病/对象与成像 | 已核实的公开内容 | 许可与访问 | 对当前方法的适用性 |
|---|---|---|---|---|
| **Duke DME / Rabbani 2015** | 糖尿病黄斑水肿；人眼 FA 早晚时相及视频 | 官方声明原图、人工及自动渗漏标记；原研究 24 名患者、24 眼 | 科研/教学开放，禁止商业化及再分发；官方页可访问，Box 本机超时 | 最接近现有 FA 渗漏分割；需核对黄斑 ROI 和实际标记格式 |
| **ApoM–S1P / CNV 与 vascular leak，Dryad 2025** | 激光诱导小鼠 nAMD/CNV；FFA 及 FITC-albumin 外渗共聚焦成像 | 已读官方完整 47 文件目录和说明；包括原始 FFA 渗漏图像包、外渗共聚焦图像包和量化数据 | **CC0**；目录和元数据可读。API 下载返回 401，官方网页下载链接返回 403，未取得 ZIP 内容 | **本轮新增最接近原模态的候选**；未确认人工渗漏掩码。适合渗漏成像/量化方法研究，监督分割可能需新增标注 |
| **FUS-induced BBB opening DCE-MRI，Zenodo** | 聚焦超声诱导小鼠血脑屏障开放；9.4T DCE-MRI | 已读数据说明 PDF，提供真实及模拟原始数据、重建 NIfTI、重建脚本；9 次真实测量，按 3 组列出，每组即刻/~4 h/~23 h | **CC BY 4.0**；ZIP 约 **6.88 GB**，说明 PDF 已下载并阅读 | 与造影剂输运和屏障通透性直接相关；需适配三维/动态 MRI。未确认人工渗漏区域真值 |
| **Repetitive mild head injury / QUTE-CE MRI，Mendeley** | 大鼠轻度重复头部损伤；血脑屏障通透性 | DataCite 注册说明原始/处理影像转为 NIfTI，每组目录含 9 只动物（R1–R10，除 R6），包含基线和造影后扫描 | 注册元数据标为开放、**CC BY 4.0**；本轮官网页面只返回前端壳，未取得当前文件列表 | 可研究脑血管通透性变化；不是已确认具有像素级渗漏掩码的分割基准 |
| **Multiregional BBB phenotyping，Zenodo** | 23 只年轻/老年小鼠；3 脑区毛细血管 TEM 电镜 | 官方说明原始图像和分割分析文件；README 指定 `.tif` 原图、`.xml.gz` 与 TrakEM2 文件；另有小鼠索引表 | **CC BY 4.0**；原图包约 **7.30 GB**；README 已下载阅读 | 有结构分割文件，但目标是屏障超微结构，**不是直接的荧光/造影剂渗漏掩码**。适合作辅助结构研究 |
| **p-Brain example sub-01，Zenodo** | 单名匿名受试者脑 DCE-MRI 示例 | 脑掩码处理后的 DCE、反转恢复序列、脑区分区和动脉输入函数 | **CC BY 4.0**；示例包约 **98.6 MB**，元数据与文件清单已确认 | 可低成本学习 BBB 通透性分析数据流程；单病例不能充当外部测试队列 |

动物实验组数、扫描次数、图像切片数不能替代独立动物数。DCE 数据说明列出的 3 组扫描，不在缺乏唯一动物索引核对时进一步推断为 3 只独立动物。

## 原始入口与重点文件

### 1. Duke DME

- 官方数据页：https://people.duke.edu/~sf59/Rabbani_IOVS_2014_dataset.htm
- 作者研究：https://doi.org/10.1167/iovs.14-15457
- 原定量分析使用以黄斑为中心、半径 1500 μm 的区域，不能直接与本地全图双病灶结果合并。

### 2. ApoM-bound S1P acts via endothelial S1PR1 to suppress choroidal neovascularization and vascular leak

- DOI：https://doi.org/10.5061/dryad.z8w9ghxpp
- 官方页：https://datadryad.org/dataset/doi:10.5061/dryad.z8w9ghxpp
- `Fig4f.zip`：**21,749,403 字节**，README 明确为 FFA 后的原始 fluorescein leakage 图像；有 vehicle、ApoM-Fc 及合并 anti-VEGF 的实验组。
- `FIG4g_FFA_SET19_CNV.pzfx`：FFA 渗漏量化数据。
- `FIG4i_image.zip`：**7,544,108 字节**，用于 FITC-albumin 外渗量化的原始共聚焦图像。
- `FIG4j_VASCULAR_LEAK_TD.pzfx`：VEGF 诱导急性高通透性实验的 FITC-albumin leakage 量化。
- `Fig4j_Quant.zip`：**119,021,048 字节**，相关外渗量化文件；内容形式未实际解包核实。

全数据共 47 个文件、约 4.16 GB。这里优先考虑 FFA 和 albumin 外渗相关小包，无需先获取其他 CNV 和 western blot 的全部原始图。

当前证据边界：已核对官方数据页、三页 API 文件目录、CC0 许可及 README 的在线内容；未完成 ZIP 下载、样本计数、图像质量检查或掩码核对。公开网页有普通文件下载链接，但本服务器下载返回 403；API 下载接口返回 401。因此不能宣称“本机已下载”或“已确认含人工区域标注”。

### 3. DCE-MRI data of focused ultrasound-induced blood-brain barrier opening in mice (real & simulated)

- 版本入口：https://zenodo.org/records/23351644
- 早期记录入口：https://doi.org/10.5281/zenodo.10992069 （本轮 API 返回当前版本元数据）
- `ISIBrno_FUS_DCEMRI_v3.zip`：**6,884,261,909 字节**。
- `ISIBrno_FUS_DCEMRI_datadescription_v3.pdf`：已下载阅读。
- 真实数据包含 `Real_data_raw/`、`Real_data_reco/`；模拟数据包含 `Simulated_data_raw/`、`Simulated_data_reco/`。
- 重建后每组含 `cal-00.nii`、`dyn-00.nii` 和 `header.json`；NIfTI 为 `(x,y,z,t,1)`、complex64、identity affine，需按说明处理幅度及空间信息。
- 模拟组包括 2D 无噪声、2D 有噪声和 3D 有噪声；不能在未检查实际仿真输出前假定已提供真实通透性参数图作为 ground truth。
- 数据附重建脚本；PerfLab 分析软件说明为按需提供，不能将其一并描述为完全开源的软件。若自建分析，可从已重建的 NIfTI 开始。

### 4. Quantitative imaging of blood brain barrier permeability following repetitive mild head injury

- DOI：https://doi.org/10.17632/hzf4rmj8yz.1
- 数据页：https://data.mendeley.com/datasets/hzf4rmj8yz/1
- 元数据说明 QUTE-CE 为 180×180×180、167 μm 等体素；另有 RARE 解剖扫描，并使用大鼠脑图谱分析 173 个脑区。
- 元数据描述使用 dextran 外渗验证屏障变化，但**不能据此认定公开包内包含 dextran 原图或人工渗漏掩码**。

### 5. Multiregional BBB phenotyping

- 数据页：https://zenodo.org/records/18669865
- 论文：https://doi.org/10.1093/braincomms/fcaf332
- 23 只 C57Bl/6J 小鼠：12 只年轻（2–3 月）、11 只老年（18–20 月）；脑区为胼胝体、海马和前额叶。
- `.xml.gz` 由 FIJI/TrakEM2 打开；并非已经整理好的二值渗漏 PNG 掩码。README 还说明两个分析文件在分析后损坏，使用前需检查。
- 与 3BTRON 数据来自相同实验室且年龄/小鼠构成重合，不能未经样本核对将两者当作独立外部队列。

### 6. p-Brain 单病例示例

- 数据页：https://zenodo.org/records/20826857
- 代码：https://github.com/edtireli/p-brain
- 论文数据声明只开放一个示例；完整临床队列仍需申请。不要把软件开源等同于完整队列开源。

## 其他保留和排除的线索

- **Zhao 疟疾视网膜病变**：仍相关。2022 年比较论文 https://doi.org/10.1167/tvst.11.3.9 描述 30 幅历史公开 FA（20 大灶、10 点状渗漏）；当前原始下载与许可未核实。它不因非葡萄膜炎而被排除。
- **3BTRON**：https://zenodo.org/records/14845497 ，CC BY 4.0、`data.zip` 约 572 MB。359 幅用于年龄组建模的电镜图像，以及中年小鼠附加数据；是屏障形态分类，不是渗漏掩码基准。
- **LEAKAGEanalysis**：https://github.com/peanutpress/LEAKAGEanalysis 。实际递归文件树只含 MATLAB apps、脚本及 README；未见公开图像数据，不计作新数据集。
- **Hypoglycemia Promotes Breakdown of the Blood-Retinal Barrier and Retinal Vascular Leakage in Diabetic Mice**：DOI `10.17632/96v2wbzwpx`、`10.17632/6xg5b8yrxn`。注册描述为 RNA-seq、HIF 靶基因及 western blot，未证实可用于图像分割；标题含 leakage 不代表发布了对应影像。
- **eNOS/NO source data**：DOI `10.5061/dryad.x69p8czhv`。官方元数据为 Excel 数值表，约 81 kB，不是眼底渗漏图像集。
- **Mosquito saliva / sialokinin-dependent vascular leakage**：DOI `10.5518/1163`。官方目录为 README、正文和补充图的 Excel 数值表，不是显微渗漏图像集。
- **Yes kinase / VE-cadherin**：https://zenodo.org/records/7229061 ，提供视网膜原图和血管/细胞分布分析文件；本次描述的发布部分不是直接渗漏标签，不能据文章主题推定其为渗漏分割基准。
- **Zebrafish MTC diffusion study**：https://zenodo.org/records/14016517 ，提供 Origin `.opju` 实验和分析文件；没有核实独立原始显微图像及掩码，不优先推荐为分割数据。
- **TCIA QIN collections**：本机连接超时，本次未完成官方目录核对，不把其加入已验证清单。

## 对你的方法，建议如何选

1. **先补 FA 渗漏分割证据**：Duke（人眼、专家标记）优先；ApoM 小鼠 FFA 可作为新候选，但先获取并核实图像和标注，必要时进行专家标注。
2. **想验证输运/扩散假设**：FUS DCE-MRI 的时序及模拟数据更有针对性，但需定义新的三维/动态通透性任务，不能把现有 FA 模型直接测试后的结果当作同任务外部 Dice。
3. **想用现成分割标注扩展实验**：TEM 数据有分析文件，但它验证的是屏障结构分割能力；应单独表述为相关任务迁移，不宣称直接验证了渗漏检测。

## 记录与核对范围

本轮新增 Europe PMC 四组题名/摘要查询（vascular leakage/permeability、BBB、intravital extravasation、retinal leakage）、DataCite 四组数据查询、GitHub 相关仓库和官方落地页。查询混有药物 BBB 穿透、细胞跨内皮迁移、组学与源数据表，均需阅读实际描述后筛选。

原始响应、解析记录、DCE 数据说明和电镜 README 存在 `outputs/public_leakage_dataset_search_20261101/`。没有修改本地 `dataset/`，没有启动训练或产生外部验证指标，没有取得新的完整原始影像队列。
