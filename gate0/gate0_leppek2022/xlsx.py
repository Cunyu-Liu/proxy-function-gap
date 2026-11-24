import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS, "r": REL_NS, "p": PKG_REL_NS}


def column_name(reference):
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {reference}")
    return match.group(1)


def column_index(reference):
    result = 0
    for char in column_name(reference):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _text(element):
    return "" if element is None else "".join(element.itertext())


def _shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [_text(item) for item in root.findall(f"{{{MAIN_NS}}}si")]


def _cell_value(cell, shared):
    kind = cell.attrib.get("t", "n")
    if kind == "inlineStr":
        return _text(cell.find("m:is", NS))
    value = cell.find("m:v", NS)
    if value is None or value.text is None:
        return ""
    raw = value.text
    if kind == "s":
        return shared[int(raw)]
    if kind == "b":
        return "true" if raw == "1" else "false"
    return raw


def workbook_sheets(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
        }
        result = []
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = "xl/" + target
            result.append((sheet.attrib["name"], target))
        return result


def iter_sheet_rows(path, sheet_name):
    with zipfile.ZipFile(path) as archive:
        targets = dict(workbook_sheets(path))
        shared = _shared_strings(archive)
        with archive.open(targets[sheet_name]) as handle:
            for event, element in ET.iterparse(handle, events=("end",)):
                if element.tag != f"{{{MAIN_NS}}}row":
                    continue
                values = {}
                formulas = {}
                for cell in element.findall("m:c", NS):
                    reference = cell.attrib["r"]
                    index = column_index(reference)
                    values[index] = _cell_value(cell, shared)
                    formula = cell.find("m:f", NS)
                    if formula is not None:
                        formulas[index] = _text(formula)
                width = max(values, default=-1) + 1
                row = [values.get(index, "") for index in range(width)]
                yield int(element.attrib["r"]), row, formulas
                element.clear()


def read_table(path, sheet_name):
    rows = iter_sheet_rows(path, sheet_name)
    try:
        header_number, header, _ = next(rows)
    except StopIteration:
        return [], []
    if header_number != 1:
        raise ValueError(f"Expected header on row 1 in {path}:{sheet_name}")
    names = []
    seen = set()
    for index, value in enumerate(header):
        name = value.strip() if value.strip() else f"__column_{index + 1}"
        if name in seen:
            raise ValueError(f"Duplicate column {name!r} in {path}:{sheet_name}")
        seen.add(name)
        names.append(name)
    result = []
    for row_number, values, formulas in rows:
        padded = values + [""] * max(0, len(names) - len(values))
        if not any(value != "" for value in padded[:len(names)]):
            continue
        record = {name: padded[index] if index < len(padded) else "" for index, name in enumerate(names)}
        record["__row_number__"] = row_number
        record["__formulas__"] = {names[index]: value for index, value in formulas.items() if index < len(names)}
        result.append(record)
    return names, result


def sheet_dimensions(path):
    result = {}
    for name, target in workbook_sheets(path):
        count = 0
        width = 0
        for _, row, _ in iter_sheet_rows(path, name):
            count += 1
            width = max(width, len(row))
        result[name] = {"rows_including_header": count, "maximum_columns": width, "xml_path": target}
    return result
