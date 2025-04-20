import csv
import re


def load_dataset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            label = int(row[0])  # Convert label to integer
            text = row[1].strip()  # Remove extra spaces

            data.append((process_text(text), label))
    return data


def process_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
