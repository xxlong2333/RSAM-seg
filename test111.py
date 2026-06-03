import os

img_dir = "./data/PVP/val/images"
lab_dir = "./data/PVP/val/labels"

imgs = sorted(os.listdir(img_dir))
labs = sorted(os.listdir(lab_dir))

print("num images:", len(imgs))
print("num labels:", len(labs))

for i in range(20):
    print(i, imgs[i], "<->", labs[i])

assert len(imgs) == len(labs)

for i, (img, lab) in enumerate(zip(imgs, labs)):
    img_stem = os.path.splitext(img)[0]
    lab_stem = os.path.splitext(lab)[0]

    lab_stem_norm = lab_stem.replace("_mask", "").replace("_label", "").replace("_gt", "")

    if img_stem != lab_stem_norm:
        print("mismatch at", i, img, lab)
        break
else:
    print("all matched")