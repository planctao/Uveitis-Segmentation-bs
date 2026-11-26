"""Plain black-and-white method slides with editable orthogonal diagrams."""
from pathlib import Path
import json
import hashlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

PROJECT = Path(__file__).resolve().parents[2]
SRC = Path('/root/autodl-tmp/uveitis_threeclass_20261124')
OUT = PROJECT / 'classification/outputs/four_methods_plain_20261126'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'Noto Sans CJK SC'
BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)
prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)
routes = []
notes = []


def txt(s, x, y, w, h, content, size=20, bold=False, center=False):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.015)
    tf.margin_top = tf.margin_bottom = Inches(.01)
    for i, line in enumerate(str(content).split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = BLACK
        p.font.bold = bold
        p.space_after = Pt(3)
        p.line_spacing = 1.12
        if center:
            p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            ea = OxmlElement('a:ea')
            ea.set('typeface', FONT)
            r._r.get_or_add_rPr().append(ea)
    return sh


def slide(title, intro, source):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    txt(s, .50, .35, 12.25, .58, title, 28, True)
    txt(s, .52, 1.08, 12.20, .69, intro, 19)
    s.notes_slide.notes_text_frame.text = source
    notes.append({'slide': len(prs.slides), 'title': title, 'source': source})
    return s


def block(s, x, y, w, h, label, size=16, bold=False):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = WHITE
    sh.line.color.rgb = BLACK
    sh.line.width = Pt(1)
    # Remove the theme style so PowerPoint/LibreOffice cannot add shadows.
    style = sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
    if style is not None:
        sh._element.remove(style)
    sh._element.spPr.append(OxmlElement('a:effectLst'))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(.05)
    tf.margin_top = tf.margin_bottom = Inches(.035)
    for i, ln in enumerate(label.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = BLACK
        p.font.bold = bold
        p.space_after = Pt(2)
        p.line_spacing = 1.08
        for r in p.runs:
            ea = OxmlElement('a:ea'); ea.set('typeface', FONT)
            r._r.get_or_add_rPr().append(ea)
    return sh


def path(s, points, arrow=True, dashed=False):
    for i, ((x1,y1),(x2,y2)) in enumerate(zip(points,points[1:])):
        assert abs(x1-x2)<1e-6 or abs(y1-y2)<1e-6, points
        if abs(x1-x2)<1e-6 and abs(y1-y2)<1e-6:
            continue
        sh = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        sh.line.color.rgb = BLACK
        sh.line.width = Pt(1)
        style = sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
        if style is not None:
            sh._element.remove(style)
        ln = sh._element.spPr.get_or_add_ln()
        if dashed:
            el=OxmlElement('a:prstDash');el.set('val','dash');ln.append(el)
        if arrow and i==len(points)-2:
            el=OxmlElement('a:tailEnd');el.set('type','triangle');el.set('w','sm');el.set('len','sm');ln.append(el)
        routes.append((len(prs.slides),x1,y1,x2,y2))


def table(s, x,y,widths,headers,rows,rowh=.70,fontsize=18):
    shape=s.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(sum(widths)),Inches((len(rows)+1)*rowh))
    tab=shape.table
    tab.first_row=False
    tab.horz_banding=False
    for j,w in enumerate(widths):
        tab.columns[j].width=Inches(w)
    for i,row in enumerate([headers]+rows):
        tab.rows[i].height=Inches(rowh)
        for j,val in enumerate(row):
            c=tab.cell(i,j)
            c.fill.solid();c.fill.fore_color.rgb=WHITE
            c.margin_left=c.margin_right=Inches(.08)
            c.margin_top=c.margin_bottom=Inches(.03)
            c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.text=str(val)
            for p in c.text_frame.paragraphs:
                p.font.name=FONT;p.font.size=Pt(fontsize);p.font.bold=(i==0);p.font.color.rgb=BLACK
                p.alignment=PP_ALIGN.LEFT;p.space_after=Pt(0)
                for r in p.runs:
                    r.font.name=FONT;r.font.size=Pt(fontsize);r.font.color.rgb=BLACK;r.font.bold=(i==0)
                    ea=OxmlElement('a:ea');ea.set('typeface',FONT);r._r.get_or_add_rPr().append(ea)
            for side in ['L','R','T','B']:
                ln=OxmlElement('a:ln'+side);ln.set('w','9525')
                fill=OxmlElement('a:solidFill');col=OxmlElement('a:srgbClr');col.set('val','000000');fill.append(col);ln.append(fill)
                c._tc.get_or_add_tcPr().append(ln)
    return shape


def read(rel): return json.loads((SRC/rel).read_text())
base=read('results/vitb_320/baseline/comparison.json')
m1=read('results/milestone1/primary/metrics.json')
m2=read('results/milestone2/primary/metrics.json')
sel1=read('results/milestone1/selection_before_test.json')
sel2=read('results/milestone2/selection_before_test.json')
pct=lambda v:f'{100*v:.2f}%'

# 1. Baseline, with no title page or ornament.
s=slide('基线：冻结 DINOv3 特征＋直接 MLP 分类',
        '是什么：把整张图像编码成一个全局特征向量，直接预测感染、非感染、正常三类。',
        'extract_vit.py:21-35；experiment.py:96-100,150-168,366-368；results/vitb_320/baseline/comparison.json.flat。')
blocks=[(.55,2.43,1.55,'输入图像\n320×320'),(2.55,2.43,2.10,'冻结 DINOv3-B\nViT-B/16'),(5.10,2.43,2.05,'全局 CLS\n768 维'),(7.60,2.43,2.12,'MLP\n768→256→3'),(10.17,2.43,2.60,'三类概率\n感染 / 非感染 / 正常')]
for x,y,w,lab in blocks:block(s,x,y,w,1.12,lab,16)
for a,b in zip(blocks,blocks[1:]):path(s,[(a[0]+a[2],2.99),(b[0],2.99)])
txt(s,.62,4.14,12.0,.55,'怎么训练：骨干不更新，只训练 MLP；GELU 激活，Dropout=0.3。',21)
txt(s,.62,4.92,12.0,.85,'怎么预测：训练 3 个随机种子，平均三个模型的预测概率，再取概率最大的类别。\n作用：建立“仅使用全局表示”的参考，用来比较后续结构是否有收益。',21)
txt(s,.62,6.23,12.0,.47,'外部准确率：Independent 80.79%；Ningbo 82.68%。',21)

# 2. Cascade I.
s=slide('两阶段方法一：先判断感染，再区分非感染和正常',
        '是什么：用两个独立的二分类 MLP，按照第一阶段的结果进行硬路由。',
        'experiment.py Model/cascade_i、routed、tune_thresholds、loss_fn；baseline comparison.json.cascade_i。阈值0.55/0.30来自开发验证选择。')
block(s,.60,3.07,2.48,1.05,'全局特征\nDINOv3-B CLS',17)
block(s,3.69,3.07,2.60,1.05,'阶段 1\n感染 vs 其余',17)
block(s,7.24,2.52,2.16,.83,'感染',18)
block(s,7.24,4.10,2.16,1.05,'阶段 2\n非感染 vs 正常',17)
block(s,10.10,3.84,2.55,.73,'非感染',18)
block(s,10.10,4.88,2.55,.73,'正常',18)
path(s,[(3.08,3.595),(3.69,3.595)])
path(s,[(6.29,3.595),(6.78,3.595)],False)
path(s,[(6.78,2.935),(6.78,4.625)],False)
path(s,[(6.78,2.935),(7.24,2.935)])
path(s,[(6.78,4.625),(7.24,4.625)])
path(s,[(9.40,4.625),(9.76,4.625)],False)
path(s,[(9.76,4.205),(9.76,5.245)],False)
path(s,[(9.76,4.205),(10.10,4.205)])
path(s,[(9.76,5.245),(10.10,5.245)])
txt(s,.62,1.76,11.9,.52,'训练方式：阶段 1 使用全部训练图；阶段 2 只使用非感染、正常的真实子集。',20)
txt(s,.62,5.97,12.0,.58,'局限：第一阶段误判后，第二阶段无法恢复被错误路由的类别。',21)
txt(s,.62,6.67,12.0,.45,'端到端准确率：Independent 79.80%；Ningbo 82.68%。',20)

# 3. Cascade N.
s=slide('两阶段方法二：先判断异常，再区分感染和非感染',
        '是什么：先把正常图像分出去，再在异常图像中判断病因，仍采用硬路由。',
        'experiment.py Model/cascade_n、routed、tune_thresholds、loss_fn；baseline comparison.json.cascade_n。阈值0.35/0.50来自开发验证选择。')
block(s,.60,3.07,2.48,1.05,'全局特征\nDINOv3-B CLS',17)
block(s,3.69,3.07,2.60,1.05,'阶段 1\n异常 vs 正常',17)
block(s,7.24,2.52,2.16,.83,'正常',18)
block(s,7.24,4.10,2.16,1.05,'阶段 2\n感染 vs 非感染',17)
block(s,10.10,3.84,2.55,.73,'感染',18)
block(s,10.10,4.88,2.55,.73,'非感染',18)
path(s,[(3.08,3.595),(3.69,3.595)])
path(s,[(6.29,3.595),(6.78,3.595)],False)
path(s,[(6.78,2.935),(6.78,4.625)],False)
path(s,[(6.78,2.935),(7.24,2.935)])
path(s,[(6.78,4.625),(7.24,4.625)])
path(s,[(9.40,4.625),(9.76,4.625)],False)
path(s,[(9.76,4.205),(9.76,5.245)],False)
path(s,[(9.76,4.205),(10.10,4.205)])
path(s,[(9.76,5.245),(10.10,5.245)])
txt(s,.62,1.76,11.9,.52,'训练方式：阶段 1 使用全部训练图；阶段 2 只使用感染、非感染的真实子集。',20)
txt(s,.62,5.97,12.0,.58,'注意：第二阶段条件准确率只在真实异常子集上计算，不等于端到端准确率。',20)
txt(s,.62,6.67,12.0,.45,'端到端准确率：Independent 80.55%；Ningbo 82.68%。',20)

# 4. Explicit M1 motivation, framed as design hypotheses rather than causal proof.
s=slide('M1 为什么要这样改？',
        '设计出发点：既保留局部区域线索，又避免硬路由过早丢弃其他类别的可能性。',
        '方法动机依据首轮候选结构及README.md“架构探索与稳健性”，并与extract_vit.py和experiment.py实际实现核对。以下是结构设计假设，不是已证明的独立机制贡献；首轮多个组件联合改变。')
txt(s,.62,1.96,12.0,.45,'1. 全局 CLS 是整体摘要，不能显式选择局部区域。',22,True)
txt(s,.92,2.53,11.65,.82,'分类线索可能只出现在部分区域，因此增加 3×3 区域特征和两头注意力池化，\n让分类头学习不同区域的权重，而不只依赖一个全局向量。',20)
txt(s,.62,3.59,12.0,.45,'2. 两阶段硬路由会传播第一阶段的错误。',22,True)
txt(s,.92,4.16,11.65,.82,'改为同时计算直接三分类、感染优先、正常优先三个概率分支，\n再学习融合权重，保留三类的可能性。',20)
txt(s,.62,5.22,12.0,.45,'3. 单一末层全局特征的信息形式较少。',22,True)
txt(s,.92,5.79,11.65,.82,'补入中间层均值、末层 patch 的空间标准差，并使用细病种辅助监督，\n尝试让共享表示同时利用不同层次和空间变化的信息。',20)
txt(s,.62,6.89,12.0,.30,'以上是优化动机；现有联合实验不能把收益单独归因于某一个模块。',14)

# 5. M1 diagram. Rectangles only; all paths are orthogonal.
s=slide('M1：区域特征＋双层级软融合',
        '是什么：四路图像特征先形成共享表示，再由三个三分类概率分支融合输出。',
        'extract_vit.py:21-35；experiment.py:110-117,150-179,254-282,366-379。CLS投影256；区域9×768投影9×128后2头池化256；中层/标准差投影64/64；拼接640→共享隐藏128。细病种头仅辅助训练。')
block(s,.40,3.28,1.27,.84,'FA 图像\n320×320',14)
block(s,2.00,3.05,1.50,1.28,'DINOv3-B\nViT-B/16\n冻结',14)
path(s,[(1.67,3.70),(2.00,3.70)])
path(s,[(3.50,3.70),(3.75,3.70)],False)
ys=[1.90,3.10,4.30,5.50]
centers=[y+.42 for y in ys]
path(s,[(3.75,centers[0]),(3.75,centers[-1])],False)
features=['末层 CLS\n768 → 256','区域注意力（2头）\n9×768 → 256','第8个 block 均值\n768 → 64','末层 patch 标准差\n768 → 64']
for y,c,lab in zip(ys,centers,features):
    block(s,4.03,y,2.08,.84,lab,14)
    path(s,[(3.75,c),(4.03,c)])
    path(s,[(6.11,c),(6.40,c)],False)
path(s,[(6.40,centers[0]),(6.40,centers[-1])],False)
block(s,6.76,3.05,1.53,1.28,'拼接 640 维\n共享 MLP\n640 → 128',14)
path(s,[(6.40,3.69),(6.76,3.69)])
path(s,[(8.29,3.69),(8.56,3.69)],False)
bys=[2.10,3.30,4.50]
bcs=[y+.39 for y in bys]
path(s,[(8.56,bcs[0]),(8.56,bcs[-1])],False)
for y,c,lab in zip(bys,bcs,['直接三分类\np₀','感染优先软层级\npᵢ','正常优先软层级\npₙ']):
    block(s,8.86,y,1.98,.78,lab,14)
    path(s,[(8.56,c),(8.86,c)])
    path(s,[(10.84,c),(11.14,c)],False)
path(s,[(11.14,bcs[0]),(11.14,bcs[-1])],False)
block(s,11.49,3.18,1.42,1.04,'门控融合\nw(z)\np = Σ wⱼpⱼ',14)
path(s,[(11.14,3.70),(11.49,3.70)])
block(s,11.49,5.02,1.42,.82,'三类概率\n3种子平均',14)
path(s,[(12.20,4.22),(12.20,5.02)])
block(s,6.76,5.90,1.85,.76,'细病种辅助头\n仅训练时监督',13)
path(s,[(7.525,4.33),(7.525,5.90)],True,True)
txt(s,.52,6.93,12.2,.25,'实线：预测路径。虚线：辅助训练路径。四路特征的标准化统计只使用拟合患者。',13)

# 6. M1 explained without decorative cards or charts.
s=slide('M1 的“软层级”如何工作？',
        '是什么：把两个分类顺序写成可微的三类概率，再与直接三分类概率加权求和。',
        'experiment.py:265-282,370-379,455-468；M1冻结配置aux0.4、consistency0.3。四个二分类sigmoid输出a,b,c,d对应条件概率；三个分支均按感染/非感染/正常排序。')
txt(s,.62,1.99,12.0,.46,'直接分支：p₀ = softmax（直接分类 logits）',22)
txt(s,.62,2.78,12.0,.46,'感染优先：pᵢ = [a, (1−a)b, (1−a)(1−b)]',22)
txt(s,.92,3.34,11.6,.38,'a = P(感染)，b = P(非感染｜非感染＋正常)。',18)
txt(s,.62,4.02,12.0,.46,'正常优先：pₙ = [cd, c(1−d), 1−c]',22)
txt(s,.92,4.58,11.6,.38,'c = P(异常)，d = P(感染｜感染＋非感染)。',18)
txt(s,.62,5.26,12.0,.48,'最终输出：w = softmax(Wz)，p = w₀p₀ + wᵢpᵢ + wₙpₙ。',22)
txt(s,.62,6.03,12.0,.88,'训练目标：三分类损失＋掩码二分类辅助损失＋0.3×双层级一致性\n                 ＋0.4×细病种辅助损失。三个概率分支同时参与预测。',19)

# 7. M2. Existing M1 features retained; subtype becomes predictive branch.
s=slide('M2：增加病种概率汇总分支',
        '是什么：把 M1 中只辅助训练的细病种头，变成参与最终预测的第四个分支。',
        'milestone2/selection_before_test.json；subtype_experiments.py；experiment.py:174-182,370-379,455-468。M2保留M1的三个预测分支，并将病种softmax通过映射求和作为第四分支；不是后续紧凑M4结构。')
block(s,.60,3.15,2.20,1.10,'M1 共享表示\n128 维',18)
path(s,[(2.80,3.70),(3.13,3.70)],False)
ys=[1.95,3.05,4.15,5.25];cs=[y+.39 for y in ys]
path(s,[(3.13,cs[0]),(3.13,cs[-1])],False)
for y,c,lab in zip(ys,cs,['直接三分类 p₀','感染优先 pᵢ','正常优先 pₙ','细病种 softmax']):
    block(s,3.55,y,2.65,.78,lab,17)
    path(s,[(3.13,c),(3.55,c)])
block(s,6.73,5.25,2.20,.78,'按病种映射求和\n得到三类概率',15)
path(s,[(6.20,cs[-1]),(6.73,cs[-1])])
for c in cs[:3]:path(s,[(6.20,c),(9.47,c)],False)
path(s,[(8.93,cs[-1]),(9.47,cs[-1])],False)
path(s,[(9.47,cs[0]),(9.47,cs[-1])],False)
block(s,10.00,3.15,2.68,1.10,'四路门控加权\n输出三类概率',18)
path(s,[(9.47,3.70),(10.00,3.70)])
txt(s,.62,6.35,12.0,.83,'为什么增加：细病种标签提供更细的监督；病种概率汇总后可直接表达三类判断。\n实现变化：病种辅助权重 0.4→0.6，并增加病种分支与融合输出的一致性约束。',18)

# 8. Last page: one complete table, no later closing slide.
s=slide('方法与结果总表',
        '共同设置：冻结 DINOv3-B/16、320 输入、3 种子概率平均；两阶段采用硬路由。\n验证 374 张；Independent 807 张；Ningbo 508 张。CV 按患者分组。',
        '验证及外部指标：baseline comparison.json、milestone1/2 primary/metrics.json；CV：milestone1 selection_before_test.json的相应finalists与milestone2 selection_before_test.json.cv。所有准确率/F1为图像级；CV按患者分组。')
models=[('基线 MLP',base['flat'],.7440732996916449),
        ('两阶段：感染优先',base['cascade_i'],.7427074592432131),
        ('两阶段：正常优先',base['cascade_n'],.7551182255054377),
        ('M1：区域软层级',m1,sel1['winner']['cv']['mean_macro_f1']),
        ('M2：病种汇总',m2,sel2['cv']['mean_macro_f1'])]
rows=[]
for name,obj,cv in models:
    v=obj['results']['val']['image'];a=obj['results']['test_independent']['image'];b=obj['results']['test_ningbo']['image']
    rows.append([name,pct(v['accuracy']),f"{v['macro_f1']:.4f}",f'{cv:.4f}',f"{pct(a['accuracy'])} / {a['macro_f1']:.4f}",f"{pct(b['accuracy'])} / {b['macro_f1']:.4f}"])
table(s,.48,1.93,[2.55,1.34,1.43,1.44,2.78,2.78],
      ['方法','验证\n准确率','验证\nMacro-F1','患者三折\nMacro-F1','Independent\n准确率 / Macro-F1','Ningbo\n准确率 / Macro-F1'],rows,rowh=.72,fontsize=17)
txt(s,.55,6.47,12.2,.63,'结论：M1 在两个外部集的点估计均高于基线；M2 在 Independent 进一步提高，\n但 Ningbo 未超过 M1。M1 的配对差异区间均跨 0；M2 仅 Independent 高于 0。',17)

# Structural validation: monochrome, basic rectangles, orthogonal connectors, bounds.
for i,s in enumerate(prs.slides,1):
    for sh in s.shapes:
        assert sh.left>=0 and sh.top>=0,(i,sh.name)
        assert sh.left+sh.width<=prs.slide_width+Inches(.01),(i,sh.name,'right')
        assert sh.top+sh.height<=prs.slide_height+Inches(.01),(i,sh.name,'bottom')
        if sh.shape_type == 1:
            assert sh.auto_shape_type == MSO_SHAPE.RECTANGLE,(i,sh.name)
for i,x1,y1,x2,y2 in routes:
    assert x1==x2 or y1==y2,(i,x1,y1,x2,y2)
prs.core_properties.title='分类方法说明：基线、两阶段、M1、M2（简洁版）'
prs.core_properties.subject='方法解释、M1设计动机、正交结构图及结果总表'
prs.core_properties.author='Classification research'
file=OUT/'分类方法说明_简洁版.pptx'
prs.save(file)
sources=['results/vitb_320/baseline/comparison.json','results/milestone1/primary/metrics.json','results/milestone2/primary/metrics.json',
         'results/milestone1/selection_before_test.json','results/milestone2/selection_before_test.json','extract_vit.py','experiment.py','subtype_experiments.py']
(OUT/'source_notes.json').write_text(json.dumps({'slides':notes,'source_root':str(SRC),'source_sha256':{r:hashlib.sha256((SRC/r).read_bytes()).hexdigest() for r in sources},
                                               'validation':{'slide_count':len(prs.slides),'orthogonal_line_segments':len(routes),'last_slide':'方法与结果总表'}},ensure_ascii=False,indent=2)+'\n')
print(file)
print('slides:',len(prs.slides),'orthogonal line segments:',len(routes))
