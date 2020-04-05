import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# Reading data: I manually transformed found data to two separate sheets:
# first consists from information about oligonucleotides
info = pd.read_csv("information.zip", sep="\t", index_col="ID", compression="zip")
# second consists from data: 18 mouses, 22514 values each
raw_data = pd.read_csv("data.txt", sep="\t", index_col="ID_REF")

# Checking that there is no manual mistake in filling GENE and GENE_SYMBOL fields:
check = info.duplicated(subset=["GENE"], keep=False).equals(info.duplicated(subset=["GENE_SYMBOL"], keep=False))
# = True - everything is okay

raw_data = raw_data.join(info["GENE_SYMBOL"])
raw_data = raw_data.join(info["GENE"])
unknown_genes = raw_data[raw_data["GENE_SYMBOL"].isna() == 1]
raw_data = raw_data.dropna(subset=["GENE_SYMBOL"])
data = raw_data.groupby("GENE_SYMBOL").mean()

# now we have to data bases:
# data - data with gene symbol as index and averaged if there is
# more than one oligonucleotide matches one gene
# unknown_genes - data with oligonucleotide ID as index for those,
# that we have no gene symbol for


# Answer to first part:
# If we consider every na gene as a unique one:
higher_estimate = len(data) + len(unknown_genes)  # = 15133
# If we consider every na gene as a mistake or one big unknown gene
lower_estimate = len(data)                        # = 13321
# So as study suggests: there has been 44k oligonucleotides on Aglient microarrays kit
# that was transformed to 22514 values for each mouse.
# Then we found that not every oligonucleotide belongs to a unique gene, so we transformed data to
# 15133 values for each mouse.
# However, many genes didn't have name in the list, so if we consider those irrelevant the answer is
# There are 13321 genes in the study.

# About data:
# There is a probability that researches made some mistakes in naming genes.
# In the info data sheet there is a GENE_NAME section that we may consider as well.
# I suggest comparing duplicates on GENE_SYMBOL's with different GENE_NAME's on values to
# determine if they it's similar gene or a mistake.



gene = data["GENE"]
control = data[data.columns[range(6)]]
low_dose = data[data.columns[range(6, 12)]]
high_dose = data[data.columns[range(12, 18)]]

control_unknown = unknown_genes[unknown_genes.columns[range(6)]]
low_dose_unknown = unknown_genes[unknown_genes.columns[range(6, 12)]]
high_dose_unknown = unknown_genes[unknown_genes.columns[range(12, 18)]]



# Answer to second part:
control_mean = control.mean(axis=1)
low_dose_mean = low_dose.mean(axis=1)
high_dose_mean = high_dose.mean(axis=1)
control_unknown_mean = control_unknown.mean(axis=1)
low_dose_unknown_mean = low_dose_unknown.mean(axis=1)
high_dose_unknown_mean = high_dose_unknown.mean(axis=1)


# Answer to third part:
model1 = DecisionTreeClassifier(max_depth=2)
model2 = DecisionTreeClassifier(max_depth=2)

x = round(control_mean)
y1 = round(low_dose_mean)
y2 = round(high_dose_mean)
X = x.to_frame(name="B").join(gene.to_frame(name="A")).to_numpy()

model1.fit(X, y1)
model2.fit(X, y2)

pred1 = model1.predict(X)   # = 8 for all
pred2 = model2.predict(X)   # = 8 for all

features1 = model1.feature_importances_   # = [9.99880454e-01 1.19546472e-04]
features2 = model2.feature_importances_   # = [9.99922207e-01 7.77932421e-05]

genes = x[x == 8].index     # those genes are the best for splitting

# I know this is a really bad solution, however I really wan't
# to improve my skill's and I am ready to work hard.

# P. S. I would've done it in jupyter but I have some kernel problem there
# and I had not much time to solve it, sorry.