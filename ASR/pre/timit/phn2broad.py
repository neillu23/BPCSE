import os 
import csv


phn_path = "/home/neillu/Downloads/TIMIT_Neil/TIMIT/"
broad_path = "/home/neillu/Downloads/TIMIT_Clean_Broad/"
phn_to_broad = {}

with open('broad_phone_5.csv') as bp:
    rows = csv.reader(bp)
    for row in rows:
        if row[0][0] == "#":
            continue
        for phn in row[1:]:
            print phn, row[0]
            phn_to_broad[phn] = row[0]
# phn_to_broad["h#"] = "h#"

for root, dirs, files in os.walk(phn_path):
    # print root, dirs, files
    for phn_file in files:
        if ".PHN" in phn_file:
            broad_phns = []
            with open(os.path.join(root,phn_file),'r') as fp:
                all_lines = fp.readlines()
                for line in all_lines:
                    eles = line[:-1].split(" ")
                    # try:
                    eles[2] = phn_to_broad[eles[2]]
                    # except:
                    #     eles[2] = "Others"
                    broad_phns.append(eles)
            broad_root = root.replace(phn_path, broad_path)
            with open(os.path.join(broad_root,phn_file),'w') as fp:
            # with open(os.path.join("test/",phn_file+".new"),'w') as fp:
                for bp in broad_phns:
                    fp.write(bp[0]+" "+bp[1]+" "+bp[2]+"\n")




