# 公开 FA 血管渗漏数据集核对与适用性

核对日期：2026-11-01（Asia/Shanghai）。任务依据：本项目 `AGENTS.md`、`RESULTS.md`、IsoLeak 方法文档及此前外部数据调研。当前任务为 FA 图像两通道病灶分割（lesion_1 / lesion_2）；候选资源按输入模态、标注内容和访问条件判断，不能仅凭名称包含“血管”认定适用。

补充范围：根据“不限葡萄膜炎，只要与血管渗漏相关”的要求，新增跨疾病、跨器官、显微与 DCE-MRI 数据核对见 [广义血管渗漏公开数据](public_vascular_leakage_datasets_broad_20261101.md)。其中新增 ApoM 小鼠 FFA/albumin 外渗、FUS 血脑屏障开放 DCE-MRI、头部损伤 QUTE-CE 和 BBB 电镜资源。本文原结论限于直接适配本地 FA 渗漏分割的资源。

## 结论

有相关公开资源，但本次**没有新增核实到一个能够直接下载、带完整像素级渗漏真值、且与本地两类病灶定义一致的数据集**。最值得优先获取的仍是 Duke DME。Zhao 疟疾视网膜病变是有文献支持的历史公开候选，但当前完整下载入口和许可未核实。FFA-IR 是受控访问的框标注与报告数据；RECOVERY-FA19、FLoRI21 可支持辅助研究，不能直接充当渗漏分割测试集。

本轮仅检索和保存公开网页/元数据/论文，没有下载完整患者影像数据，没有注册、签署协议或联系作者。

## 候选资源

| 资源 | 规模、模态与标注 | 开放状态核对 | 对本方法的用途 |
|---|---|---|---|
| **Duke DME / Rabbani 2015** | 原研究 24 名患者、24 眼；早晚期 FA、视频及人工/自动渗漏标记；研究比较了两名专家标注 | 作者官方数据页本轮 HTTP 200，并明确链接 FULL DATA SET。科研/教学使用，要求引用，禁止商业化和再分发。Box 连接本轮超时，尚未取得文件清单，也未确认标记文件是独立掩码还是带标记图像 | **首选跨疾病渗漏验证候选**。原研究定量区域为黄斑中心半径 1500 μm ROI，需保留该范围和专家标注口径；不能当作葡萄膜炎专属双类真值 |
| **Zhao malarial retinopathy / 2015、2016–2017** | 疟疾视网膜病变 FA。2022 年后续论文明确称其为公开集，并描述 30 幅异常 FA：20 幅大灶、10 幅点状渗漏，存在专家参考标注 | 本轮读取原 2015 研究及 2022 比较论文；没有核实到当前可下载完整原图/标注的官方入口及使用许可。此前调研记录中的旧相关域名已变更为无关内容，不提供该旧地址 | 可检验大灶/点状渗漏的跨疾病泛化；**只能列为待恢复获取的候选**，不能写成已取得的开源数据 |
| **FFA-IR v1.1.0 / PhysioNet** | 47,247 幅 FFA、766 份报告；46 类病灶、12,166 个区域；官方说明病灶位置为最小外接矩形，提供中英文报告 | 官方页本轮 HTTP 200；Credentialed Access，需身份认证、CITI 指定培训、签署项目 DUA | 可用于 FA 领域表征预训练、多病灶定位或弱监督研究；**不是现成的像素级渗漏真值**，不能将框直接当作掩码计算 Dice |
| **RECOVERY-FA19 / DeepVesselSeg4FA** | 作者仓库明确涉及 8 幅 UWFA 的留一验证；真值为视网膜血管图，仓库提供单幅示例并指向 IEEE DataPort 全集 | 作者 GitHub README 本轮 HTTP 200；完整数据访问、文件和许可未在本轮取得核对 | 可辅助血管先验、血管分支或结构张量方向研究；**血管掩码不是渗漏掩码**，不能验证渗漏分割增益 |
| **FLoRI21** | UWFA 纵向视网膜图像配准数据 | DataCite 元数据本轮确认数据 DOI、IEEE DataPort open-access 落地页及 CC BY 4.0；完整文件未下载 | 可研究时相/随访配准；本次未确认像素级渗漏真值，不用于直接评价渗漏 Dice |

### 官方或原始文献入口

- Duke 数据页：https://people.duke.edu/~sf59/Rabbani_IOVS_2014_dataset.htm
- Duke 官方页链接的 Box：https://duke.box.com/s/u3ia6fjsfbrzkhsswex4cqosezuo0az1
- Duke 研究：https://doi.org/10.1167/iovs.14-15457
- Zhao 原研究：https://doi.org/10.1038/srep10425
- Zhao 后续方法：https://doi.org/10.1109/TMI.2016.2593725
- 明确描述两个历史公开渗漏集的 2022 比较论文：https://doi.org/10.1167/tvst.11.3.9
- FFA-IR：https://physionet.org/content/ffa-ir-medical-report/1.1.0/
- FFA-IR DOI：https://doi.org/10.13026/f7w3-gm74
- RECOVERY-FA19 作者代码及数据说明：https://github.com/ShamaLabUR/DeepVesselSeg4FA
- 作者 README 指向的数据 DOI：https://doi.org/10.21227/m9yw-xs04
- FLoRI21：https://doi.org/10.21227/ydp8-zf19

## 本轮新增核对、但不能列为已开放渗漏数据集的线索

1. **CMWNet，TVST 2026**：`Collaborative Multiscale and Wavelet-Based Fusion Network for Leakage Area Semantic Segmentation of Ultrawide Field Fluorescein Angiography`，DOI https://doi.org/10.1167/tvst.15.3.22 。研究有内部 183 幅（111/29/43）、外部 43 幅 UWFA 渗漏分割数据。本轮读取 XML 中数据、实验及相关段落，未定位原始影像及真值的公开下载入口。属于研究方向非常接近、可申请合作的数据线索，不是已确认开源集。
2. **CSC 漏点分割，IJO 2026**：`An intelligent segmentation method for leakage points in central serous chorioretinopathy based on fluorescein angiography images`，DOI https://doi.org/10.18240/ijo.2026.03.01 。182 幅 FFA，包含点级和像素级标注；102/40/40 划分。本轮 PMC HTML 读取成功，未定位完整数据下载链接。其任务偏局灶漏点定位，也不同于全图弥漫性炎症渗漏。
3. **nAMD 渗漏与新生血管分割**：DOI https://doi.org/10.1038/s41433-022-02156-6 。原文明确因隐私限制不公开，合理请求可联系通讯作者，不计入直接公开下载数据。
4. **huapu4/FFA / Zenodo 11212584**：仓库提供分类/分割方法和部署协议，README 明确要求放入自己的数据。Zenodo 的代码压缩包不等于公开患者数据集。代码：https://github.com/huapu4/FFA 。
5. **JingWei / Zenodo 23503796**：公开应用及模型权重，预测输入是 OCT 或彩色眼底照片；演示样本说明为 AI 生成。不能把模型/演示样例当作真实 FA 渗漏分割队列。代码：https://github.com/laleoarrow/jingwei-uveitis 。
6. **Zenodo 6331029 CSC-sCNV**：元数据提供三个 ORIGINAL DATA ZIP 文件及 CC BY 4.0 声明，但本轮未检查压缩包内容、病例结构或任何渗漏标注；不能将“存在 ZIP”推定为已确认可用的渗漏分割数据集。

## 对当前方法的具体建议

- **要补外部测试，先尝试 Duke**：首先核对影像、独立真值格式、ROI、患者索引和早晚时相。报告应称“跨疾病 FA 渗漏分割验证”，与本地葡萄膜炎结果分开呈现。
- **先核对病灶语义，再映射输出**：本地是两通道多标签，外部通常只提供一类渗漏。不能把同一张外部真值复制到两个通道。应根据标注定义预先确定比较某一通道还是并集；若无合理映射，可另设单通道任务比较方法架构，明确这是重新训练后的外部任务结果。
- **冻结外部测试规则**：使用开发集确定的模型和阈值，不在外部测试真值上选择阈值、后处理或最佳模型。若使用部分外部病例适配，应单列适配训练集和独立测试集。
- **患者是独立单位**：同一患者的早晚时相、多帧、切块和增强图不能跨训练/测试；Duke 24 人不能因切块变成数千名独立样本。
- **血管先验研究可考虑 RECOVERY-FA19**，但其结果支持的是辅助任务，不能代替渗漏性能证据。现有 VA-RDH 的结构张量本身不依赖外部血管标注，新增血管监督需单独做消融。

## 检索范围及可追溯记录

本轮实际访问 Europe PMC 两组题名/摘要查询（命中 50、15 条）、DataCite 的 FA dataset 查询（210 条，分页读取 100/100/10；含病例补充材料、版本重复和非目标模态，不代表 210 个可用图像集）、另一组 fluorescein/leakage 数据查询、Zenodo 组合查询、GitHub 相关仓库搜索及上述官方资源页。

Google 请求超时；通用网页搜索曾返回明显不相关内容，因此结论主要依据作者页面、PhysioNet、论文全文及科研仓库元数据。GitHub API 搜索触发限流后改查公开网页。检索不是系统综述或穷尽性证明；没有定位到公开入口不等于数据永远无法申请。

本轮网页、XML、JSON 及错误记录：`outputs/public_leakage_dataset_search_20261101/`。其中 `*.response` 为服务器原始响应，`*.json` 为抽取记录，`*.error` 为连接失败记录。Duke 和其他历史来源的更早证据还可见 `outputs/severity_research_20261013/external_data_search/`。
