import os
import csv


def readarray(file_path):
    data, times = [], []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        has_time = 'time' in (reader.fieldnames or [])
        for row in reader:
            if not row.get('frame') or row.get('happy') is None:
                break
            try:
                data.append([float(row['happy']), float(row['sad']),
                             float(row['confused']), float(row['angry'])])
                if has_time:
                    times.append(float(row['time']))
            except (ValueError, TypeError):
                break
    return data, times if times else None


def readcomment(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == '':
            return ''.join(lines[i + 1:])
    return ''


def savedata(file_path, text):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == '':
            cutoff = i + 1
            break
    with open(file_path, 'w') as file:
        file.writelines(lines[:cutoff])
        file.write(text)


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]
