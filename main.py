import codecs
import multiprocessing
import re
import collections
import pickle

ignored_chars = {'$', '(', ',', '.', ':', ';', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\', '`', '\'',
                 '+', '-', '*', '/', '<', '>', '^', '%', '=', '?', '!', '[', ']', '{', '}', '_', '\n', '"', '&', '~'}


def base_forms(line):
    tokens = unicode(line, "utf-8").lower().split(", ")
    return tokens


def normalize_text(text):
    text = unicode(text, "utf-8").lower()
    for pattern in ignored_chars:
        text = re.sub(re.escape(pattern), '', text)
    return text.split()


def to_base(args):
    (word_list, base_form) = args
    counter = collections.Counter()
    for word in word_list:
        try:
            counter[base_form[word]] += 1
        except KeyError:
            counter[word] += 1
    return counter


def tf_idf(word, notice, notices, global_counter):
    tf = 1.0 * notice.get(word, 0) / sum(notice.values())
    idf = 1.0 * len(notices) / global_counter[word]
    return tf * idf

pool = multiprocessing.Pool()

# read odm
base_form = {}
with open("lab7/odm_utf8.txt") as file:
    for line in file.readlines():
        tokens = base_forms(line)
        for el in tokens:
            base_form[el] = tokens[0]

# read pap
with open("lab7/pap.txt") as file:
    text = file.read()
notice_text = re.split(r'#.*', text)
notice_words = map(normalize_text, notice_text)
notice_words = map(lambda words: map(lambda word: base_form.get(word, ''), words), notice_words)
notice_words = map(lambda words: filter(lambda word: word != '', words), notice_words)
notice_counters = map(lambda words: collections.Counter(words), notice_words)

global_counter = collections.Counter()
for notice in notice_counters:
    for elem in list(notice):
        global_counter[elem] += 1

tl_idf_matrix = map(lambda notice: map(lambda word: (word, tf_idf(word, notice, notice_counters, global_counter)), list(notice)), notice_counters)


pickle.dump(tl_idf_matrix, open("tl_idf", "wb"))

with codecs.open("result", "w", encoding='utf-8') as file:
    for i in range(len(tl_idf_matrix)):
        file.write('###')
        file.write(str(i))
        file.write('\n')
        for (word, weight) in tl_idf_matrix[i]:
            file.write('   ')
            file.write(word)
            file.write(' ')
            file.write(str(weight))
            file.write('\n')


