"""Build an editable Chinese presentation from existing aggregate experiment results."""
from pathlib import Path
import json
import math
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.chart.data import CategoryChartData
from pptx.oxml.xmlchemy import OxmlElement

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path('/root/autodl-tmp/uveitis_threeclass_20261124')
OUT = PROJECT / 'classification/outputs/four_methods_20261126'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'Noto Sans CJK SC'
C = dict(bg='F5F7FA', white='FFFFFF', navy='172C46', text='263C50', muted='63758A',
         teal='168B91', light='E4F2F1', blue='376AB0', pale='EAF0FA', coral='D97858',
         peach='FBEEE7', line='D8E1EA', gray='EDF1F5')
prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)
source_notes = []


def rgb(s): return RGBColor.from_string(C.get(s, s))


def rect(s, x, y, w, h, fill='white', line=None, radius=False):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                            Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    if line: sh.line.color.rgb = rgb(line); sh.line.width = Pt(1)
    else: sh.line.fill.background()
    if radius:
        try: sh.adjustments[0] = .10
        except Exception: pass
    return sh


def text(s, x, y, w, h, value, size=18, color='text', bold=False, align=None, valign=MSO_ANCHOR.TOP):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.025)
    tf.margin_top = tf.margin_bottom = Inches(.015)
    tf.vertical_anchor = valign
    for i, ln in enumerate(str(value).split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.font.name = FONT; p.font.size = Pt(size)
        p.font.bold = bold; p.font.color.rgb = rgb(color)
        p.space_after = Pt(4)
        if align is not None: p.alignment = align
        for run in p.runs:
            rpr = run._r.get_or_add_rPr()
            ea = OxmlElement('a:ea'); ea.set('typeface', FONT); rpr.append(ea)
    return sh


def box(s, x, y, w, h, title, sub='', fill='white', accent='teal', fs=17, subfs=12):
    rect(s,x,y,w,h,fill,'line',True)
    rect(s,x,y+.12,.045,h-.24,accent)
    if sub:
        compact = h < .82
        text(s,x+.15,y+(.06 if compact else .12),w-.28,.34,title,fs,accent,True)
        text(s,x+.15,y+(.40 if compact else .56),w-.28,h-(.43 if compact else .61),sub,subfs)
    else:
        text(s,x+.10,y+.03,w-.2,h-.06,title,fs,accent,True,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)


def arrow(s,x1,y1,x2,y2,color='muted',width=1.5,dashed=False):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
    ln=sh._element.spPr.get_or_add_ln()
    end=OxmlElement('a:tailEnd'); end.set('type','triangle');end.set('w','sm');end.set('len','sm');ln.append(end)
    if dashed:
        dash=OxmlElement('a:prstDash');dash.set('val','dash');ln.append(dash)
    return sh


def label(s,x,y,w,value,fill='light',color='teal'):
    rect(s,x,y,w,.32,fill,radius=True)
    text(s,x+.03,y+.015,w-.06,.29,value,10,color,True,PP_ALIGN.CENTER)


def slide(title, subtitle='', tag='METHODS', source=''):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid();s.background.fill.fore_color.rgb=rgb('bg')
    rect(s,0,0,13.333,.085,'teal')
    text(s,.5,.30,11.5,.6,title,29,'navy',True)
    if subtitle:text(s,.53,1.00,12.15,.48,subtitle,13,'muted')
    text(s,.52,7.12,10.9,.21,'葡萄膜炎 FA 图像分类  /  '+tag+'  /  2026.11.26',9,'muted')
    text(s,12.05,7.08,.75,.25,f'{len(prs.slides):02d}',10,'muted',align=PP_ALIGN.RIGHT)
    if source:
        s.notes_slide.notes_text_frame.text=source
    source_notes.append({'slide':len(prs.slides),'title':title,'sources':source})
    return s


def table(s, x,y,width, headers, rows, widths=None, rowh=.49, fontsize=14):
    widths = widths or [width/len(headers)]*len(headers)
    for r,row in enumerate([headers]+rows):
        cy=y+r*rowh; cx=x
        for j,val in enumerate(row):
            fill='navy' if r==0 else ('white' if r%2 else 'gray')
            rect(s,cx,cy,widths[j]-.015,rowh-.018,fill)
            text(s,cx+.09,cy+.07,widths[j]-.19,rowh-.12,val,fontsize,'white' if r==0 else 'text',r==0)
            cx+=widths[j]


def metric(s,x,y,w,value,caption,color='teal'):
    rect(s,x,y,w,1.13,'white',radius=True)
    text(s,x+.15,y+.10,w-.3,.52,value,29,color,True)
    text(s,x+.15,y+.69,w-.3,.28,caption,11,'muted')


def read(rel):return json.loads((SOURCE/rel).read_text())
base=read('results/vitb_320/baseline/comparison.json')
m1=read('results/milestone1/primary/metrics.json')
m2=read('results/milestone2/primary/metrics.json')
sel1=read('results/milestone1/selection_before_test.json')
sel2=read('results/milestone2/selection_before_test.json')
pct=lambda a:f'{100*a:.2f}%'
f1=lambda a:f'{a:.4f}'
models=[('基线 MLP',base['flat']),('感染优先两阶段',base['cascade_i']),('正常优先两阶段',base['cascade_n']),('M1 区域双层级',m1),('M2 病种汇总',m2)]

# 01 — Cover
s=slide('葡萄膜炎三分类：四类方法与实验结果',
        '基线  ·  两阶段  ·  M1 区域双层级  ·  M2 病种概率汇总', 'OVERVIEW',
        '来源：README.md；results/vitb_320/baseline/comparison.json；milestone1/2 的 selection_before_test.json 和 primary/metrics.json。全部结果为既有汇总，无新增模型拟合或外部测试。')
text(s,.65,1.80,9.9,1.25,'从全局特征直接分类，\n走向区域证据与病种层级建模',32,'navy',True)
text(s,.69,3.20,10.9,.48,'统一采用冻结 DINOv3 ViT-B/16、320 输入及 3 个随机种子的概率平均。',16,'muted')
cards=[('01','基线','全局 CLS → MLP'),('02','两阶段','两种顺序，硬路由'),('03','M1','区域注意力＋软层级'),('04','M2','增加病种汇总分支')]
for i,(n,t,sub) in enumerate(cards):
    x=.67+i*3.12;rect(s,x,4.25,2.92,1.63,'white',radius=True)
    label(s,x+.16,4.43,.48,n)
    text(s,x+.16,4.95,2.5,.39,t,23,'navy',True)
    text(s,x+.16,5.47,2.6,.25,sub,12,'muted')
text(s,.71,6.40,11.8,.35,'重点：M1 可编辑结构图、两阶段与软层级的区别、M1 / M2 的外部收益及局限。',14,'teal')

# 02 — Protocol
s=slide('统一任务与评估口径', '类别顺序固定：感染 / 非感染 / 正常。所有方法使用相同纳入规则与数据划分。','PROTOCOL',
        '来源：README.md 数据与评估；experiment.py normalize/sample_weights/train_one；crossvalidate.py。不同患者的多眼、多时点图像整组隔离；CV 内另设早停患者。')
table(s,.62,1.77,8.15,['集合','感染','非感染','正常','图像 / 患者'],[
    ['训练','702','787','228','1717 / 332'],['验证','141','174','59','374 / 83'],
    ['Independent','364','327','116','807 / 198'],['Ningbo','134','284','90','508 / 151']],
    [1.9,1.12,1.12,1.0,3.01],rowh=.65,fontsize=15)
box(s,9.05,1.77,3.65,1.35,'开发选型','验证 F1 选择 / 早停；\n患者三折 CV 检查稳定性。',fill='light',subfs=14)
box(s,9.05,3.35,3.65,1.35,'外部评估','先冻结模型，再计算测试指标；\n两来源使用同一冻结配方。',fill='pale',accent='blue',subfs=14)
box(s,.62,5.10,12.08,1.10,'报告口径','准确率与 Macro-F1 均为图像级；置信区间按患者整组重采样。CV 不是患者聚合分类准确率。',subfs=14)
text(s,.69,6.48,11.8,.3,'预处理：四周裁 5% → 等比例补边至 448 → 缩放 320；特征统计仅拟合训练患者。',13,'muted')

# 03 — Baseline
s=slide('基线：DINOv3 全局特征＋直接 MLP', '用最少的分类结构建立可比较的三分类参考。','BASELINE',
        '来源：vitb_320/baseline/comparison.json 的 flat.selection/config；extract_vit.py:32-35；experiment.py:96-100,150-168,300-302,366-368。配置 global/flat/h256/dropout0.3，3种子平均概率。')
nodes=[(.65,1.85,'FA 图像','320 × 320',1.66),(2.75,1.85,'冻结 DINOv3-B','ViT-B/16',2.25),(5.48,1.85,'CLS 特征','768 维；训练集标准化',2.35),(8.30,1.85,'MLP','768 → 256 → 3',1.93),(10.68,1.85,'输出概率','感染 / 非感染 / 正常',2.0)]
for x,y,t,sub,w in nodes:box(s,x,y,w,1.37,t,sub,fs=16,subfs=12)
for a,b in zip(nodes,nodes[1:]):arrow(s,a[0]+a[4],2.53,b[0]-.05,2.53)
box(s,.66,3.68,5.78,1.35,'训练与推理','分类头：Linear → GELU → Dropout(0.3) → Linear；\n三个独立种子预测概率取均值，最终 argmax。',subfs=15)
box(s,6.73,3.68,5.94,1.35,'这个基线回答什么','冻结预训练表示的全局信息，能否直接分清三类？\n它为区域证据、路由结构和病种监督提供统一参照。',fill='pale',accent='blue',subfs=15)
metric(s,.67,5.47,3.76,pct(base['flat']['results']['test_independent']['image']['accuracy']),'Independent 准确率')
metric(s,4.72,5.47,3.76,pct(base['flat']['results']['test_ningbo']['image']['accuracy']),'Ningbo 准确率')
metric(s,8.77,5.47,3.88,'0.7441','原患者三折 CV Macro-F1','blue')

# 04 — Hard cascades
s=slide('两阶段：把三分类拆成两个二分类任务', '每种流程使用两个独立 MLP；第二阶段只在真实适用类别子集上训练。','TWO-STAGE',
        '来源：experiment.py routed/tune_thresholds/Model/loss_fn；baseline comparison.json。感染优先选定阈值[0.55,0.30]；正常优先[0.35,0.50]；阈值由开发验证集选择，CV只用内层早停集。外部评价采用硬路由。')
text(s,.65,1.63,5.6,.38,'A  感染优先',20,'teal',True)
box(s,.65,2.17,2.30,1.12,'阶段 1','感染 vs 其余',subfs=14)
box(s,3.55,1.81,2.25,.65,'感染',fill='light',fs=17)
box(s,3.55,2.82,2.25,1.06,'阶段 2','非感染 vs 正常',subfs=14)
arrow(s,2.95,2.56,3.5,2.14);arrow(s,2.95,2.81,3.5,3.28)
text(s,.75,4.20,5.5,.7,'端到端准确率\nIndependent 79.80%  /  Ningbo 82.68%',17,'navy',True)
text(s,6.9,1.63,5.6,.38,'B  正常优先',20,'blue',True)
box(s,6.90,2.17,2.30,1.12,'阶段 1','异常 vs 正常',accent='blue',subfs=14)
box(s,9.80,1.81,2.25,.65,'正常',fill='pale',accent='blue',fs=17)
box(s,9.80,2.82,2.25,1.06,'阶段 2','感染 vs 非感染',accent='blue',subfs=14)
arrow(s,9.20,2.56,9.75,2.14);arrow(s,9.20,2.81,9.75,3.28)
text(s,7.0,4.20,5.5,.7,'端到端准确率\nIndependent 80.55%  /  Ningbo 82.68%',17,'navy',True)
rect(s,.65,5.34,12.03,.96,'peach',radius=True)
text(s,.88,5.54,11.55,.6,'阶段 1 一旦错分，阶段 2 无法补救。第二阶段条件准确率 ≠ 全系统准确率；最终指标必须包含路由错误。',17,'text',True)
text(s,.73,6.51,11.9,.29,'所选隐藏层均为 128；感染优先阈值 0.55 / 0.30，正常优先 0.35 / 0.50。',12,'muted')

# 05 — M1 full editable architecture
s=slide('M1 模型结构：区域证据＋双层级软融合', '单尺度 320 输入；DINOv3-B 冻结。下图各模块、文字和连接线均可编辑。','M1 ARCHITECTURE',
        '结构严格依据 extract_vit.py:21-35 与 experiment.py:110-117,150-179,254-282,366-379。四路特征768维；区域9×768→9×128；2头区域加性打分softmax池化为256；其余投影256/64/64；拼接640→共享MLP128。直接/感染优先/正常优先三路概率经softmax门控融合。细病种头只作为辅助监督，不参与M1推理融合。')
box(s,.50,2.96,1.32,1.17,'FA 图像','320 × 320',fs=16,subfs=12)
box(s,2.12,2.75,1.62,1.62,'DINOv3-B','ViT-B/16\n冻结骨干',fill='pale',accent='blue',fs=16,subfs=13)
arrow(s,1.82,3.55,2.08,3.55)
ys=[1.73,2.88,4.03,5.18]
features=[('末层 CLS','768 → 256'),('3×3 区域 token','9×768 → 9×128'),('第 8 个 block 均值','768 → 64'),('末层 patch 标准差','768 → 64')]
for i,(t,sub) in enumerate(features):
    box(s,4.10,ys[i],2.20,.89,t,sub,fs=13,subfs=11)
    arrow(s,3.76,3.55,4.06,ys[i]+.45,'blue',1.1)
box(s,6.56,2.88,1.62,.89,'2 头区域池化','每头 128 → 共 256',fill='light',fs=13,subfs=10)
arrow(s,6.30,3.32,6.52,3.32)
box(s,6.56,4.45,1.62,1.20,'拼接＋MLP','256+256+64+64\n640 → 128',fill='light',fs=13,subfs=11)
arrow(s,7.37,3.77,7.37,4.40)
for i in [0,2,3]:arrow(s,6.30,ys[i]+.45,6.52,4.99,'muted',1.1)
branchys=[1.82,3.09,4.36]
for i,(t,sub) in enumerate([('直接三分类','p₀：softmax 3类'),('感染优先软层级','pᵢ：条件概率分解'),('正常优先软层级','pₙ：条件概率分解')]):
    box(s,8.57,branchys[i],2.02,.92,t,sub,fill='white',accent='blue' if i==0 else 'teal',fs=14,subfs=11)
    arrow(s,8.18,5.02,8.53,branchys[i]+.46,'muted',1.1)
box(s,10.98,2.95,1.80,1.17,'门控加权融合','w = softmax(Wz)\np = Σ wⱼ pⱼ',fill='navy',accent='white',fs=14,subfs=11)
# Correct the secondary text colour on the dark fusion card.
for shape in list(s.shapes)[-2:]:
    if shape.has_text_frame:
        for p in shape.text_frame.paragraphs:
            for r in p.runs:r.font.color.rgb=rgb('white')
for y in branchys:arrow(s,10.59,y+.46,10.94,3.54,'teal',1.2)
box(s,8.57,5.72,2.02,.68,'细病种辅助头',fill='peach',accent='coral',fs=13)
arrow(s,8.18,5.30,8.54,6.04,'coral',1.1,True)
text(s,10.97,4.55,1.85,.95,'输出三类概率\n3 种子概率平均\n→ 最终类别',12,'navy',True)
arrow(s,11.88,4.14,11.88,4.46,'teal',1.2)
text(s,.56,6.62,12.2,.30,'特征先按训练集统计标准化。虚线为仅训练时的辅助监督；M1 的病种分支不参与最终概率融合。',11,'muted')

# 06 — M1 mechanics
s=slide('M1 如何计算：局部选择与软层级融合', '区域注意力负责提取局部证据，两个条件分解提供不同的分类组织方式。','M1 DETAIL',
        '来源：experiment.py:265-282,370-379,455-468；M1配置aux0.4、consistency0.3。q的四个sigmoid分量a,b,c,d对应感染、非感染|非感染性集合、异常、感染|异常；BCE使用各真实适用子集掩码。')
box(s,.62,1.72,5.85,2.17,'区域注意力：2 组可学习权重','tᵢ = GELU(Wᵣ rᵢ)，每个区域 128 维\nαᵢₖ = softmaxᵢ[W₂ tanh(W₁ tᵢ)]ₖ\nuₖ = Σᵢ αᵢₖ tᵢ，k = 1, 2；拼接为 256 维',fill='light',fs=18,subfs=16)
box(s,6.78,1.72,5.90,2.17,'融合保持为三类概率','p₀ = softmax(直接分类 logits)\npᵢ = [a, (1−a)b, (1−a)(1−b)]\npₙ = [cd, c(1−d), 1−c]',fill='pale',accent='blue',fs=18,subfs=17)
box(s,.62,4.20,12.06,1.23,'可学习融合','w = softmax(Wₘz)；p = w₀p₀ + wᵢpᵢ + wₙpₙ。三路同时计算并参与预测，不使用硬路由丢弃分支。',fs=18,subfs=16)
text(s,.72,5.76,11.95,.77,'训练目标：三分类 NLL ＋ 掩码二分类辅助损失 ＋ 0.3 × 双层级 JS 一致性\n                 ＋ 0.4 × 细病种交叉熵。分类样本按患者与类别加权。',16,'text')
text(s,.72,6.61,12.0,.26,'a=P(感染)，b=P(非感染｜其余)，c=P(异常)，d=P(感染｜异常)；类别顺序均为感染 / 非感染 / 正常。',11,'muted')

# 07 — M2 architecture
s=slide('M2：将细病种辅助头变成第四个预测分支', 'M1 的视觉编码器和双层级分支保留；新增“病种 → 三类”的概率汇总与融合。','M2 ARCHITECTURE',
        '来源：milestone2/selection_before_test.json；subtype_experiments.py；experiment.py:174-182,370-379,455-468。subtype_fusion=true；病种类别映射由拟合患者构建；aux0.6, consistency0.3, subtype_consistency0.1；非后续M4/M5紧凑taxonomy-only结构。')
box(s,.65,2.92,2.25,1.44,'M1 共享表示','区域＋全局多路特征\n共享隐藏向量 z∈R¹²⁸',fill='pale',accent='blue',fs=17,subfs=14)
for y,t in [(1.75,'直接三分类 p₀'),(2.82,'感染优先 pᵢ'),(3.89,'正常优先 pₙ')]:
    box(s,3.40,y,2.10,.69,t,fs=15)
    arrow(s,2.90,3.64,3.36,y+.345,'muted',1.1)
box(s,3.40,5.03,2.10,.96,'细病种 softmax','K 类病种概率',fill='peach',accent='coral',fs=15,subfs=12)
arrow(s,2.90,3.94,3.36,5.48,'coral')
box(s,6.00,4.97,2.31,1.08,'按病种映射汇总','p病种(c) = Σd→c p(d)',fill='peach',accent='coral',fs=16,subfs=13)
arrow(s,5.50,5.52,5.96,5.52,'coral')
box(s,8.89,2.83,3.70,1.85,'四路门控融合','p = Σⱼ wⱼpⱼ；j = 直接 / 感染优先 /\n正常优先 / 病种汇总\n输出仍为感染、非感染、正常三类',fill='light',fs=20,subfs=15)
for y in [2.095,3.165,4.235]:arrow(s,5.50,y,8.85,3.48,'muted',1.1)
arrow(s,8.31,5.50,8.86,4.15,'coral')
text(s,.74,6.34,11.9,.52,'病种辅助损失权重 0.4 → 0.6；保留双层级一致性 0.3，新增融合输出与病种分支的一致性 0.1。',14,'text')

# 08 — Development results
s=slide('开发集结果：M1 / M2 提升了验证与患者 CV 表现', 'CV 使用原训练患者的外层三折，内层患者用于早停及阈值选择；每折 3 个种子。','DEVELOPMENT',
        '来源：baseline comparison.json；milestone1 selection_before_test.json winner.cv/group；milestone2 selection_before_test.json selected/cv。CV标准差来自三折。')
cvvals=[(.7440732996916449,.0319),(.7427074592432131,.0259),(.7551182255054377,.0128),
        (sel1['winner']['cv']['mean_macro_f1'],sel1['winner']['cv']['sd_macro_f1']),
        (sel2['cv']['mean_macro_f1'],sel2['cv']['sd_macro_f1'])]
rows=[]
for (name,obj),(cv,sd) in zip(models,cvvals):
    im=obj['results']['val']['image'];rows.append([name,pct(im['accuracy']),f1(im['macro_f1']),f'{cv:.4f} ± {sd:.4f}'])
table(s,.65,1.82,12.02,['方法','验证准确率','验证 Macro-F1','患者 CV Macro-F1'],rows,[3.75,2.45,2.55,3.27],rowh=.64,fontsize=17)
metric(s,.67,5.85,3.76,'+0.0351','M1 相对基线的 CV F1 变化')
metric(s,4.72,5.85,3.76,'+0.0100','M2 相对 M1 的 CV F1 变化')
box(s,8.77,5.85,3.88,1.12,'选型原则','开发集选择后冻结，外部结果不参与选型。',fs=15,subfs=12)

# 09 — external chart
s=slide('外部准确率：M1 两来源提高，M2 的收益不一致', '同一冻结模型分别评估 Independent（807 张）与 Ningbo（508 张）。','EXTERNAL ACCURACY',
        '来源：baseline comparison.json；milestone1/primary/metrics.json；milestone2/primary/metrics.json。图表为PowerPoint原生可编辑图表；y轴从0开始。')
data=CategoryChartData();data.categories=[n for n,_ in models]
for split,title in [('test_independent','Independent'),('test_ningbo','Ningbo')]:
    data.add_series(title,[o['results'][split]['image']['accuracy']*100 for _,o in models])
chart=s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.69), Inches(1.73), Inches(11.96), Inches(4.28),data).chart
chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.BOTTOM;chart.legend.include_in_layout=False
chart.legend.font.name=FONT;chart.legend.font.size=Pt(12)
chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=100;chart.value_axis.major_unit=20
chart.value_axis.tick_labels.font.name=FONT;chart.value_axis.tick_labels.font.size=Pt(11)
chart.value_axis.tick_labels.number_format='0"%"'
chart.category_axis.tick_labels.font.name=FONT;chart.category_axis.tick_labels.font.size=Pt(11)
plot=chart.plots[0];plot.has_data_labels=True;plot.data_labels.position=XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.number_format='0.00"%"';plot.data_labels.font.name=FONT;plot.data_labels.font.size=Pt(11)
plot.gap_width=75
for ser,color in zip(chart.series,['teal','blue']):
    ser.format.fill.solid();ser.format.fill.fore_color.rgb=rgb(color);ser.format.line.fill.background()
text(s,.76,6.21,11.95,.58,'相对基线：M1 +1.86 / +1.38 个百分点；M2 +3.72 / +1.18 个百分点（Independent / Ningbo）。',16,'navy',True)

# 10 — Metrics stage + F1
s=slide('为什么不能只看阶段准确率或总体准确率', '把路由错误和类别差异一起报告，才能判断改进发生在哪里。','ERROR ANALYSIS',
        '来源：baseline comparison.json 各results.image.stage*_accuracy；milestone1/2 primary metrics recall；milestone1 paired_comparison.json；milestone2 paired_external_comparison.json；README.md。M2宁波感染召回67.16%。')
text(s,.65,1.63,11.9,.34,'两阶段的条件指标很高，但端到端仍受错误路由影响',18,'navy',True)
rows=[]
for key,name in [('cascade_i','感染优先'),('cascade_n','正常优先')]:
    for split,st in [('test_independent','Independent'),('test_ningbo','Ningbo')]:
        v=base[key]['results'][split]['image'];rows.append([name+' / '+st,pct(v['stage1_accuracy']),pct(v['stage2_oracle_accuracy']),pct(v['accuracy'])])
table(s,.65,2.19,12.02,['流程 / 集合','阶段 1','阶段 2 条件','端到端'],rows,[5.0,2.1,2.4,2.52],rowh=.51,fontsize=14)
box(s,.65,5.08,5.85,1.42,'M2 在 Ningbo 的类别召回','感染 67.16%  /  非感染 90.14%  /  正常 88.89%\n总体准确率 83.86%，类别表现仍不均衡。',fill='peach',accent='coral',fs=17,subfs=14)
box(s,6.82,5.08,5.85,1.42,'相对基线的患者配对区间','M1 两来源准确率差异区间均跨 0；\nM2 仅 Independent 的差异区间高于 0。',fill='pale',accent='blue',fs=17,subfs=14)

# 11 — M2 diagnostic
s=slide('M2 的分支诊断：病种汇总承担主要决策权重', '诊断来自 374 张开发验证图；它解释模型行为，不能单独证明某个组件的因果贡献。','M2 DIAGNOSTIC',
        '来源：milestone2/selection_before_test.json.subtype_contribution。mean_mixture_weights=[0.0106559163,0.0002415119,0.0003604556,0.9887433052]；with_subtype_branch_removed F1=.81204275；full F1=.83668780；subtype_branch_only与full分类相同。移除诊断使用联合训练权重，不等同于重新训练消融。')
weights=sel2['subtype_contribution']['mean_mixture_weights']
labels=['直接三分类','感染优先','正常优先','病种汇总']
for i,(name,w) in enumerate(zip(labels,weights)):
    y=1.94+i*.88
    text(s,.72,y,1.75,.35,name,17,'navy',True)
    rect(s,2.80,y+.04,7.75,.32,'gray',radius=True)
    rect(s,2.80,y+.04,max(.025,7.75*w),.32,'coral' if i==3 else 'teal',radius=True)
    text(s,10.90,y-.01,1.60,.38,f'{100*w:.2f}%',19,'coral' if i==3 else 'muted',True)
box(s,.67,5.65,5.87,.97,'完整模型 F1 = 0.8367','移除病种汇总分支后：F1 = 0.8120。',fill='peach',accent='coral',fs=17,subfs=14)
box(s,6.85,5.65,5.80,.97,'下一步结构假设','可检验去除冗余路由头的紧凑病种结构。',fill='light',fs=17,subfs=14)

# 12 — Conclusions and sources
s=slide('四类方法的结论与可追溯来源', '本汇报聚焦基线、两阶段、M1、M2；不混入后续 M4–M8 的结果。','TAKEAWAYS',
        '全部数据仅重排既有汇总，不涉及训练或外部评估。完整来源及sha256见同目录source_data.json；可编辑结构图在第5页，PDF/PNG为导出的审阅版本。生成脚本：classification/tools/build_four_methods_ppt.py。')
for i,(t,sub,color) in enumerate([
    ('基线：80.79% / 82.68%','冻结全局特征＋MLP提供统一参考。','blue'),
    ('两阶段：未超过直接 MLP','硬路由会传播第一阶段错误；必须看端到端效果。','blue'),
    ('M1：82.65% / 84.06%','区域注意力＋双层级软融合，两个来源点估计均提高。','teal'),
    ('M2：84.51% / 83.86%','病种汇总在独立集进一步提高；宁波感染召回仍偏低。','coral')]):
    y=1.72+i*.93;box(s,.65,y,12.02,.78,t,sub,accent=color,fs=17,subfs=12)
text(s,.73,5.69,11.9,.29,'数值顺序：Independent / Ningbo 图像级准确率。结构图依照代码实现绘制，模块名称不是原创性声明。',11,'muted')
text(s,.73,6.15,11.9,.60,'来源：results/vitb_320/baseline/comparison.json；milestone1 / milestone2 的冻结选择与评估记录。\n代码：extract_vit.py、experiment.py、subtype_experiments.py。每页备注含对应来源与解释。',12,'muted')

# Make sure every object stays on the slide; content overflow is also checked in rendered PDF.
for i,s in enumerate(prs.slides,1):
    for sh in s.shapes:
        assert sh.left>=0 and sh.top>=0, (i,sh.name)
        assert sh.left+sh.width <= prs.slide_width+Inches(.01), (i,sh.name,'right')
        assert sh.top+sh.height <= prs.slide_height+Inches(.01), (i,sh.name,'bottom')
prs.core_properties.title='葡萄膜炎分类：基线、两阶段、M1与M2'
prs.core_properties.subject='方法结构与既有实验结果汇报'
prs.core_properties.author='Classification research'
prs.core_properties.keywords='DINOv3,葡萄膜炎,三分类,M1,M2'
path=OUT/'分类方法汇报_基线_两阶段_M1_M2.pptx'
prs.save(path)
import hashlib
rels=['results/vitb_320/baseline/comparison.json','results/milestone1/primary/metrics.json',
      'results/milestone2/primary/metrics.json','results/milestone1/selection_before_test.json',
      'results/milestone2/selection_before_test.json','extract_vit.py','experiment.py','subtype_experiments.py']
bundle={'source_root':str(SOURCE),'source_sha256':{r:hashlib.sha256((SOURCE/r).read_bytes()).hexdigest() for r in rels},
        'slide_notes':source_notes,'baseline':base,'m1':m1,'m2':m2}
# Export only aggregate metrics and method configurations, not checkpoint/individual identifiers.
clean={'source_root':str(SOURCE),'source_sha256':bundle['source_sha256'],'slide_notes':source_notes,
       'metrics':{n:{sp:ob['results'][sp]['image'] for sp in ['val','test_independent','test_ningbo']} for n,ob in models}}
(OUT/'source_data.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n')
print(path)
print('slides',len(prs.slides))
