import nltk

names = nltk.corpus.names

# The names are contained in two .txt files, male.txt and female.txt

male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Here we will find the names that are ambiguous for gender

[w for w in male_names if w in female_names]

# After showing all the ambiguous names we will produce a graph
# that shows the gender patterns for the last letter of the name

cfd = nltk.ConditionalFreqDist(
      (fileid, name[-l])
      for fileid in names.fileids()
      for name in names.words(fileid)
)

cfd.plot()
