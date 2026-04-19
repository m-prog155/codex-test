# 论文 Word 装配顺序清单

## 1. 使用原则

1. 以学校正式 Word 模板为准。
2. 当前仓库中的 Markdown 文件只提供正文内容，不替代学校模板中的题名页、声明页、页眉页脚和目录样式。
3. 先完成“装配”，再做“格式终审”，不要边贴边反复调细节。

## 2. 推荐装配顺序

建议按下面顺序将内容装入 Word 模板：

1. 封面
   来源文件：[thesis-cover-draft.md](/D:/Projects/Car/docs/thesis-cover-draft.md:1)
2. 题名页
   当前仓库未单独成稿，需按学校模板填写
3. 原创性声明/授权页
   当前仓库未提供，按学校模板填写
4. 中文摘要
   来源文件：[thesis-abstract-and-conclusion-draft.md](/D:/Projects/Car/docs/thesis-abstract-and-conclusion-draft.md:1)
5. 中文关键词
   来源文件：[thesis-abstract-and-conclusion-draft.md](/D:/Projects/Car/docs/thesis-abstract-and-conclusion-draft.md:13)
6. 英文摘要
   来源文件：[thesis-abstract-and-conclusion-draft.md](/D:/Projects/Car/docs/thesis-abstract-and-conclusion-draft.md:17)
7. 英文关键词
   来源文件：[thesis-abstract-and-conclusion-draft.md](/D:/Projects/Car/docs/thesis-abstract-and-conclusion-draft.md:27)
8. 目录
   不手打，必须用 Word 自动生成
9. 正文第 1 章至第 5 章
   来源文件：[thesis-chapters-1-5-unified-draft.md](/D:/Projects/Car/docs/thesis-chapters-1-5-unified-draft.md:1)
10. 参考文献
   来源文件：[thesis-chapters-1-5-unified-draft.md](/D:/Projects/Car/docs/thesis-chapters-1-5-unified-draft.md:281)
11. 附录
   当前如无内容，可不加
12. 致谢
   当前仓库未单独成稿，如学校要求再补

## 3. 装配时的切分规则

### 3.1 封面

- 只取封面文案，不要把“说明”“推荐填写说明”等草稿说明贴进正文。
- 最终按学校模板处理学校名称、题目、院系、专业、年级、学号、姓名、指导教师、日期。

### 3.2 摘要部分

- 中文摘要和英文摘要通常各自单独起页。
- “摘要”“Abstract”一般作为独立标题，不保留 Markdown 的 `##` 符号。
- 关键词紧跟摘要正文，不另起大节编号。

### 3.3 正文部分

- 直接从“第1章 绪论”开始粘贴正文。
- 不要把草稿性标题、文件名、备注说明粘进论文。
- 正文里已包含第 1 章到第 5 章连续内容，不需要再从分章文件重复拷贝。

### 3.4 参考文献

- 优先使用总稿末尾那一版，因为这一版的文献编号已经按正文首次出现顺序重排。
- 不要直接使用 [thesis-references-draft.md](/D:/Projects/Car/docs/thesis-references-draft.md:1) 替换总稿末尾参考文献，它是候选清单，不是最终编号版。

## 4. 每装完一部分立刻检查什么

### 4.1 封面与题名页

- 是否完全使用学校模板样式
- 学校名、题目、院系、专业、年级、学号、姓名、指导教师、日期是否齐全
- “封面”和“题名页”是否被错误合并为一页

### 4.2 摘要与关键词

- 中文摘要、英文摘要是否分别独立成页
- 关键词数量是否符合学校要求
- 中文关键词是否用中文分号分隔
- 英文关键词大小写是否统一

### 4.3 目录

- 目录必须自动生成
- 章、节、三级标题是否都套用了正确样式
- 页码跳转是否正确

### 4.4 正文

- 一级标题是否统一为“第X章”
- 二级、三级标题编号是否连续
- 正文首行缩进、行距、段前段后是否统一
- 中英文标点、括号、数字、百分号写法是否统一

### 4.5 表格与图片

- 表题是否放在表上方
- 图题是否放在图下方
- 表号、图号是否按章连续编号
- 表格是否采用学校要求的三线表样式

### 4.6 参考文献

- 是否采用学校要求的字号、行距和悬挂缩进
- 文献编号是否与正文引用一致
- 中英文文献格式是否统一
- DOI、电子文献访问日期是否按学校要求保留

## 5. 本项目对应的最终拷贝来源

如果只看“往 Word 里贴什么”，当前以这 4 份文件为主：

1. [thesis-cover-draft.md](/D:/Projects/Car/docs/thesis-cover-draft.md:1)
2. [thesis-abstract-and-conclusion-draft.md](/D:/Projects/Car/docs/thesis-abstract-and-conclusion-draft.md:1)
3. [thesis-chapters-1-5-unified-draft.md](/D:/Projects/Car/docs/thesis-chapters-1-5-unified-draft.md:1)
4. [thesis-format-review-20260416.md](/D:/Projects/Car/docs/thesis-format-review-20260416.md:1)

其中第 4 份不是正文来源，而是装配后自查用。

## 6. 最后的终审顺序

Word 装配完成后，建议只按这个顺序做最后检查：

1. 先查页码、目录、标题样式
2. 再查摘要、正文、结论中的关键数字是否一致
3. 再查表格、图片、参考文献格式
4. 最后只做错别字和标点检查

不要在终审阶段再大改正文结构。
