# 葡萄膜炎无人工定级标签研究：文献核对

检索时间：2026-10-13 UTC。使用 Europe PMC/PubMed 元数据、开放全文 XML、arXiv、PMLR 和 Crossref。原始响应与首轮四篇实际阅读的全文位于 `outputs/severity_research_20261013/literature/`；后续领域学习文献、作者固定commit和公式核对材料另位于`domain_ssl/literature/`。外部数据可行性检索、官网响应及本轮实际阅读范围保存在`external_data_search/`。没有检索到不等于不存在相关论文。

| 来源 | 已核对的任务与证据 | 对本课题的意义 |
|---|---|---|
| Tugal-Tutkun 等，2010，[ASUWOG](https://doi.org/10.1007/s10792-008-9263-x) | 原始摘要明确 FA 总分上限40，涵盖视盘高荧光、黄斑水肿、血管染色/渗漏、毛细血管渗漏、无灌注及其他征象；ICGA上限20。它不是仅按渗漏面积分级 | 应研究多征象炎症负担；不能把面积四分位点叫ASUWOG轻/中/重 |
| Jabs 等，2005，[SUN](https://doi.org/10.1016/j.ajo.2005.03.057) | 共识包括前房细胞、flare及玻璃体混浊等临床分级 | SUN临床等级与单幅FA病灶面积不是同一个参考标准 |
| Amiot 等，2025，[Automatic transformer-based grading](https://doi.org/10.1016/j.compbiomed.2025.110327) | 摘要核对；4个后极部炎症征象的监督式 transformer，543患者、1042眼、40987图像；各征象报告F1及1−OCI | 是直接相关的监督定级工作；需与标签需求和整眼终点作清晰区分 |
| Amiot 等，2026，[UveAI](https://doi.org/10.1038/s41598-026-46069-w) | 开放全文核对；3220图像、644眼、369患者；按患者保留双眼同组；6个征象模型，后极部加4个周边象限；93眼独立测试，由额外3名专家评分；总分相关R=0.96，均值AUC=0.952。模型及视野选择器都使用人工标签 | 当前高相关结果来自临床评分、患者划分和多视野整合，不能与本研究的图像掩码面积相关性横向比较 |
| Jin 等，2026，[Pediatric uveitis quantitative leakage](https://doi.org/10.1016/j.xops.2026.101102) | 开放全文核对；内部2459图像155患者五折，文中称按患者划分；外部396图像45患者；28例独立随访队列，自动分割后由两名grader校正，再做临床关联；视网膜渗漏变化−54.8%阈值属于随访炎症静息判别，并另做cluster bootstrap/LOOCV | 面积确有临床关联，但不同征象/区域不一致；其阈值不能移植到单幅图像横断面定级。本地数量及每折范围与文中一致，只是来源线索，尚不能确认同源或本地患者独立；本地跨折重复仍须审计 |
| Dhirachaikulpanich 等，2023，[Retinal vasculitis severity assessment](https://doi.org/10.1097/iae.0000000000003838) | 已从Crossref摘要扩展至PMC10442128全文Methods/Results/Discussion阅读；50患者99眼，每眼选两幅105/150/200度WFFA，4名grader评分并间隔1月重复；以GEE处理双眼关联。渗漏活动度与闭塞损伤分别分级，中央/周边和象限参与定义；四grader ICC为0.655/0.750，两专家渗漏ICC为0.795 | RV特定活动度不同于整体葡萄膜炎负担，真实时间扩展、解剖位置和可判读性不可由单图面积代替。尚未取得原队列/逐grader评分或医生确认；具体规则及分母见[临床终点补充](severity_clinical_rubric_evidence_20261013.md) |
| [Automated Detection of Vascular Leakage，2022](https://doi.org/10.1167/tvst.11.7.19) | 开放全文核对；训练200图像61患者，另50图像21患者测试；算法DSC0.563，双医生分割DSC0.483；标注者会参考前后FA时相，并排除视盘等结构 | 单幅亮度无法唯一确定渗漏；人工掩码也存在观察者差异，应保留真实时相与一致性评价 |
| [DGCL，2023](https://doi.org/10.3390/diagnostics13203251) | 开放全文核对；糖网眼底图像，多结构特征、图聚类、伪标签/centroid在线memory；主表实际是正常/异常二分类，不是五级有序严重程度；另有1%标签半监督对照 | 可以参考图聚类和稳定性，但不能用无监督二分类论文证明无监督有序定级已解决；测试类别映射也需要披露 |
| Siméoni 等，2025，[DINOv3](https://arxiv.org/abs/2508.10104) | arXiv题目、作者和版本元数据核对，仓库原始预训练权重SHA及strict load核对；本轮只用冻结ConvNeXt-Tiny | 是无人工定级标签的特征基线，属于外部大规模预训练，不是本地从零完全无监督训练 |
| Xie 等，ICML2016，[DEC](https://proceedings.mlr.press/v48/xieb16.html) | PMLR正式页面核对；深度嵌入与聚类联合优化 | 聚类编号存在置换对称性；类别发现与有序定级是不同问题 |
| Coifman 等，2005，[Diffusion maps](https://doi.org/10.1073/pnas.0500334102) | Crossref元数据核对；局部核与密度校正谱几何 | 数据流形坐标可能对应视野/时相，须验证其临床方向，并明确Nyström保留图像外推 |
| Caron 等，2020，[SwAV](https://arxiv.org/abs/2006.09882) | arXiv元数据核对；跨增强视图预测聚类分配 | 为后续领域自监督的视图一致性提供基础；不证明亮度增强可以模拟真实FA时相 |
| Bardes、Ponce、LeCun，ICLR2022，[VICReg](https://arxiv.org/abs/2105.04906) | 实际阅读23页arXiv v3全文，作者仓库固定commit `4e12602fd495af83efd1631fbe82523e6db092e0` 与下载源码匹配；单设备无偏variance/covariance及25/25/1公式通过3项测试 | 明确方差与去相关约束可作为SwAV集中之后的开发控制；本地batch16、AdamW、有限投影维数属于机制适配，不能称完整作者预算复现，也不能由有效秩推断医学语义 |

| Cohen与Welling，ICML2016，[Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html) | PMLR48:2990–2999正式元数据及全文第3/6节实际阅读；第3节明确对群平均可以构造不变量 | 四旋转均值属于已有通用思想，不训练其G-CNN；零旋转变化不是临床发现或充分创新 |
| Laptev等，CVPR2016，[TI-Pooling](https://openaccess.thecvf.com/content_cvpr_2016/html/Laptev_TI-Pooling_Transformation-Invariant_Pooling_CVPR_2016_paper.html) | 官方摘要与元数据核对，289–297；讨论变换集合上的不变池化；未声称阅读全文 | 本轮固定等权均值不同于其具体网络/池化策略，不声称架构复现 |
| Zhao等，TMI2017（2016在线），[Intensity and Compactness Enabled Saliency](https://doi.org/10.1109/TMI.2016.2593725) | 成功取得Edge Hill大学12页作者稿，全文文本已阅读并查看第4页公式；三层SLIC、强度/紧致度线索均值后相乘，再以阈值、图割、IPACHR及视盘去除得到渗漏。参数和阈值经原标注数据调试 | 已完成四种显著性图像排序适配，原图rho −0.0277/0.2211/0.0737/0.1031；明确不称完整分割器或作者代码复现，全部差异和负结果见[追加报告](severity_saliency_findings_20261013.md) |
| Zhao等，2015，[Automated Detection of Leakage](https://doi.org/10.1038/srep10425) | 本轮进一步检查介绍/数据及完整Method段；该前作Method包含grader测量视盘直径和125微米规则，未完成前作全图视觉审查 | 不能将有人工尺度输入的完整流程直接称本地完全标签自由；本轮不伪造视盘/像素尺度 |
| Li等，SASHIMI2020，[Unsupervised Adversarial Learning](https://doi.org/10.1007/978-3-030-59520-3_15) | Crossref题目/DOI和OpenAlex可用性元数据核对；未取得全文，后续2022比较文献描述其CycleGAN路径 | 不凭摘要实现或宣称完成该方法对照；其全文和正常/异常域定义仍需核对 |

## 研究差异与下一步假设

已发表的直接相关强工作依靠人工征象定级或像素分割，再验证临床终点。本研究拟减少定级标签需求，但最终医生标签仍用于独立评价。采用像素监督训练过的encoder、用掩码面积选择方向、或用医生分级挑超参数，都需要如实披露，不能归入严格标签自由拟合。

本轮的关键问题是**无序表征如何确定生物学排序方向，以及如何避免把采集条件当病情**。当前冻结DINO/聚类是必要基线，尚不是CCF-B级创新。可能研究的机制包括：用物理启发的合成局部荧光改变校准方向；用均亮度匹配的全局变化和gamma作为混杂对照；表征中分离局部病灶变化与整体采集变化；有真实多视野/时相后再做整眼汇总和纵向一致性。合成改变只有已知生成剂量，不能先验保证等价于真实病情。

DEC/AE、SwAV与VICReg均已完成三种子五折训练、冻结重载和评价。独立临床排序评价、真实患者隔离及外部/随访证据仍缺失。首轮负结果不能通过事后反向掩盖。血管炎文献初始Crossref响应保存在`outputs/severity_research_20261013/literature/retinal_vasculitis_severity_assessment_2023.json`，新增PMC全文、访问记录及实际阅读范围保存在`external_data_search/`。

四旋转聚合的文献原文、下载哈希和阅读范围另存`outputs/severity_research_20261013/literature/rotation_pooling/source_audit.json`。特征均值联合方向没有提高原图代理rho，简单raw分数均值仅有小开发差异；不据此替换临床锁定或宣称新架构。

## 外部数据来源与临床终点补充

已核对10项候选来源，当前取得的外部原始图像数为0。Rabbani等DME研究（DOI:10.1167/iovs.14-15457）的PMC方法、黄斑ROI及医生标注相关段落与Duke官方数据页已阅读；官方说明24眼/24患者，含早晚FA、原始视频及医生标注，允许研究/教学使用并规定引用和禁止再分发。Box下载连接仍超时，尚未获得文件清单；即使取得，也仅适合单独开展跨疾病渗漏代理转移，不能视为葡萄膜炎严重程度外部验证。

FFA-IR的PhysioNet v1.1.0官方页（DOI:10.13026/f7w3-gm74）已核对数据描述和访问政策；46类病灶框及报告的文件需credentialed user、CITI培训及DUA，尚未取得，整体葡萄膜炎评分也未确认。其他候选、检索分页及失败响应的具体限制见[外部数据可行性报告](severity_external_validation_feasibility_20261013.md)。全文下载、局部阅读和阅读全文的范围分别记录在`external_data_search/reading_scope.json`，不按开放全文状态推定已阅读全文。

本地2189幅原图的标准文件头均已检查，没有观察到候选临床EXIF字段；19字段交接模板只有表头。患者、眼别、时相和视野关联仍待来源提供。血管炎全文支持将活动与损伤分开，并保留ungradable和双眼关联；它不构成本地rubric批准，详见[临床终点证据](severity_clinical_rubric_evidence_20261013.md)。这两份独立补充报告尚未加入已核验的26页论文工作稿正文。

新增TMI来源的成功PDF与403/202等失败响应同时保留于`classical_leakage_baseline/literature/`，实际阅读范围及21次请求哈希见`reading_audit.json`。成功下载使用大学公开的Pure portalfiles链接，不将其他URL的错误页当作全文。四个显著性排序适配完成全量计算及评价后，另形成[英文投稿工作稿](severity_english_manuscript_20261013.md)：4表41个数值单元已与归档CSV核对，3图和16个引用编号已核验。英文Word/离线HTML已导出，10页PDF的156个文字块与5个主图像核验通过，并完成全页缩略图和9个重点页检查，见[英文渲染记录](severity_english_render_review_20261013.md)。正式投稿模板与临床论文终稿审查仍未完成。

## 投稿范围核对状态

目标保持为达到CCF-B要求的毕业论文/投稿研究。MICCAI可作为医学影像研究范围候选，但CCF官网请求仍返回WAF校验页，未将HTTP200当作分类证明。已取得2026年3月目录第三方PDF镜像，其中MICCAI为交叉/综合/新兴B类条目，但镜像真实性及CCF原件直连未核实。MICCAI Society官网已确认2027会议在Auckland、日期2027年9月26日–10月1日，投稿截止尚未公布；2026截止已过。有效官方原文、镜像来源与失败请求见[目标会议核对](severity_venue_verification_20261013.md)。具体学校认可目录、2027征稿规则及研究质量仍待确认，现阶段不声称达到录用标准。
