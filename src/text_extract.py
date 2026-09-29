"""EE_DOC_DATA/UD_DOC_DATA/NB_DOC_DATA(효능효과/용법용량/사용상주의사항) XML 문자열을
사람이 읽을 평문으로 변환."""
import re
import xml.etree.ElementTree as ET


def doc_xml_to_text(xml_str: str | None) -> str:
    if not xml_str:
        return ""

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        # 잘못된 XML은 태그만 걷어내고 반환
        return re.sub(r"<[^>]+>", " ", xml_str).strip()

    lines = []
    for article in root.iter("ARTICLE"):
        title = (article.get("title") or "").strip()
        if title:
            lines.append(title)
        for para in article.findall("PARAGRAPH"):
            text = "".join(para.itertext()).strip()
            if text:
                lines.append(text)

    return "\n".join(lines).strip()
