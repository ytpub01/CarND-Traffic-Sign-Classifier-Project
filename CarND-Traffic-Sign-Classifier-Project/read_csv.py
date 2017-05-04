import csv

with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    y_label = {}
    for row in reader:
        label = int(row[0])
        description = row[1]
        y_label.update({label: description})
print(y_label)

