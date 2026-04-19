from __future__ import annotations

import shutil
from pathlib import Path

import pythoncom
import win32com.client as win32


ROOT = Path(r"D:\Projects\Car")
DOCS = ROOT / "docs"
DOC_PATH = DOCS / "基于深度学习的车辆类型与车牌检测识别系统-论文初稿.docx"
BACKUP_PATH = DOCS / "基于深度学习的车辆类型与车牌检测识别系统-论文初稿.backup-20260418-before-cover-table.docx"

THESIS_TITLE = "基于深度学习的车辆类型与车牌检测识别系统"

WD_ALIGN_PARAGRAPH_LEFT = 0
WD_ALIGN_PARAGRAPH_CENTER = 1
WD_ALIGN_PARAGRAPH_RIGHT = 2
WD_ALIGN_ROW_CENTER = 1
WD_BORDER_TOP = -1
WD_BORDER_LEFT = -2
WD_BORDER_BOTTOM = -3
WD_BORDER_RIGHT = -4
WD_BORDER_HORIZONTAL = -5
WD_BORDER_VERTICAL = -6
WD_LINE_STYLE_NONE = 0
WD_LINE_STYLE_SINGLE = 1
WD_ROW_HEIGHT_EXACTLY = 2
WD_PAGE_BREAK = 7
PT_PER_CM = 28.3464567


def normalize(text: str) -> str:
    return (
        text.replace("\r", "")
        .replace("\x07", "")
        .replace("\n", "")
        .replace("\x0b", "")
        .strip()
    )


def find_paragraph(doc, exact: str):
    for idx in range(1, doc.Paragraphs.Count + 1):
        paragraph = doc.Paragraphs(idx)
        if normalize(paragraph.Range.Text) == exact:
            return paragraph
    raise RuntimeError(f"paragraph not found: {exact}")


def format_paragraph(selection, doc, *, font_name: str, size: float, align: int, bold: bool = False):
    selection.Font.NameFarEast = font_name
    selection.Font.Name = font_name
    selection.Font.Size = size
    selection.Font.Bold = -1 if bold else 0
    selection.ParagraphFormat.Alignment = align
    selection.ParagraphFormat.SpaceBefore = 0
    selection.ParagraphFormat.SpaceAfter = 0


def type_paragraph(selection, doc, text: str, *, font_name: str, size: float, align: int, bold: bool = False):
    format_paragraph(selection, doc, font_name=font_name, size=size, align=align, bold=bold)
    selection.TypeText(text)
    selection.TypeParagraph()


def clear_cover(doc) -> None:
    abstract_para = find_paragraph(doc, THESIS_TITLE)
    cover_range = doc.Range(0, abstract_para.Range.Start)
    cover_range.Delete()


def build_cover(doc, word) -> None:
    sel = word.Selection
    sel.SetRange(0, 0)

    sel.TypeParagraph()
    type_paragraph(sel, doc, "阳 光 学 院", font_name="华文行楷", size=42, align=WD_ALIGN_PARAGRAPH_CENTER)
    sel.TypeParagraph()
    type_paragraph(sel, doc, "本科毕业论文(设计)", font_name="黑体", size=31.5, align=WD_ALIGN_PARAGRAPH_CENTER)
    sel.TypeParagraph()
    sel.TypeParagraph()
    sel.TypeParagraph()

    table = doc.Tables.Add(sel.Range, 8, 3)
    table.AllowAutoFit = False
    table.Rows.Alignment = WD_ALIGN_ROW_CENTER
    table.TopPadding = 0
    table.BottomPadding = 0
    table.LeftPadding = 0
    table.RightPadding = 0

    table.Columns(1).Width = 4.1 * PT_PER_CM
    table.Columns(2).Width = 9.2 * PT_PER_CM
    table.Columns(3).Width = 3.2 * PT_PER_CM

    for border_id in (
        WD_BORDER_LEFT,
        WD_BORDER_RIGHT,
        WD_BORDER_TOP,
        WD_BORDER_BOTTOM,
        WD_BORDER_HORIZONTAL,
        WD_BORDER_VERTICAL,
    ):
        table.Borders(border_id).LineStyle = WD_LINE_STYLE_NONE

    rows = [
        ("题    目：", "基于深度学习的车辆类型", ""),
        ("", "与车牌检测识别系统", ""),
        ("院(系)别：", "人工智能学院", ""),
        ("专    业：", "数据科学与大数据技术", ""),
        ("年    级：", "2024级", ""),
        ("学    号：", "2024349243", ""),
        ("姓    名：", "陈东东", ""),
        ("指导教师：", "", "（签名（职称））"),
    ]

    for row_idx, (label, value, note) in enumerate(rows, start=1):
        row = table.Rows(row_idx)
        row.HeightRule = WD_ROW_HEIGHT_EXACTLY
        row.Height = (1.15 if row_idx <= 2 else 1.0) * PT_PER_CM

        label_cell = table.Cell(row_idx, 1)
        value_cell = table.Cell(row_idx, 2)
        note_cell = table.Cell(row_idx, 3)

        label_range = label_cell.Range
        label_range.End -= 1
        label_range.Text = label
        label_range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_RIGHT
        label_range.Font.NameFarEast = "宋体"
        label_range.Font.Name = "宋体"
        label_range.Font.Size = 18

        value_range = value_cell.Range
        value_range.End -= 1
        value_range.Text = value
        value_range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        value_range.Font.NameFarEast = "楷体"
        value_range.Font.Name = "楷体"
        value_range.Font.Size = 18

        note_range = note_cell.Range
        note_range.End -= 1
        note_range.Text = note
        note_range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_LEFT
        note_range.Font.NameFarEast = "宋体"
        note_range.Font.Name = "宋体"
        note_range.Font.Size = 18

        for border_id in (
            WD_BORDER_LEFT,
            WD_BORDER_RIGHT,
            WD_BORDER_TOP,
        ):
            value_cell.Borders(border_id).LineStyle = WD_LINE_STYLE_NONE
        value_cell.Borders(WD_BORDER_BOTTOM).LineStyle = WD_LINE_STYLE_SINGLE

    sel.SetRange(table.Range.End, table.Range.End)
    sel.TypeParagraph()
    sel.TypeParagraph()
    type_paragraph(sel, doc, "2026 年        05 月", font_name="楷体", size=18, align=WD_ALIGN_PARAGRAPH_CENTER)
    sel.InsertBreak(WD_PAGE_BREAK)


def main() -> None:
    if DOC_PATH.exists():
        shutil.copy2(DOC_PATH, BACKUP_PATH)

    pythoncom.CoInitialize()
    word = win32.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(DOC_PATH))
        clear_cover(doc)
        build_cover(doc, word)
        doc.Fields.Update()
        if doc.TablesOfContents.Count >= 1:
            doc.TablesOfContents(1).Update()
            doc.TablesOfContents(1).UpdatePageNumbers()
        doc.Save()
    finally:
        try:
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass
        try:
            word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
