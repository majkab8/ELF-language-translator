import csv
import xml.etree.ElementTree as ET
import os


def parse_xml_to_csv(xml_path, output_csv_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ALLOWED_POS = {"n", "vb", "adj", "adv"}
    ALLOWED_LANGS = {"q"} 
    dataset = []

    for w in root.findall(".//word"):
        lang = w.attrib.get("l")
        speech = w.attrib.get("speech", "").lower()

        if lang not in ALLOWED_LANGS: continue
        if speech not in ALLOWED_POS: continue

        form = w.attrib.get("v")
        if not form: continue

        glosses = []
        gloss_attr = w.attrib.get("gloss")
        if gloss_attr: glosses.append(gloss_attr.strip())

        for child in w:
            if child.tag.lower() in ("gloss", "meaning") and child.text:
                glosses.append(child.text.strip())

        if not glosses: continue

        for g in glosses:
            parts = [part.strip() for section in g.split(',') for part in section.split(' or ')]
            for p in parts:
                if p:
                    dataset.append((form, p))

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elvish", "english"])
        writer.writerows(dataset)

    print(f"Saved {len(dataset)} to {output_csv_path}")


if __name__ == "__main__":
    from config import XML_FILE, CSV_FILE
    parse_xml_to_csv(XML_FILE, CSV_FILE)

