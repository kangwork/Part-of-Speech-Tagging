import sys

from math import log
from copy import copy

M = {}
Initial = {}
T = {}
TagToCountT = {}
TagToCount = {}
E = {}
# map index of E to tag iff obvious and not end of the sentence
threshold = 3  # a value we use for the unknown M val
AfterThese = {}
CountThese = {}

def tag(training_names_list, test_name_file, output_name_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    print("Reading Training files...")
    _all_training_files_reader(training_names_list)
    print("Reading Test files...")
    E, Esize, SentenceStartsAt, Obvious = _test_file_reader(test_name_file)
    print("Getting answers...")
    results = Viterbi(E, Esize, SentenceStartsAt, Obvious)
    print("Writing Output file...")
    _write_answers(E, Esize, results, output_name_file)


# ----------------------- Initializing all the tags ----------------------------

HASHtoTAG = {0: 'AJ0', 1: 'AJC', 2: 'AJS', 3: 'AT0', 4: 'AV0', 5: 'AVP',
             6: 'AVQ', 7: 'CJC', 8: 'CJS', 9: 'CJT', 10: 'CRD', 11: 'DPS',
             12: 'DT0', 13: 'DTQ', 14: 'EX0', 15: 'ITJ', 16: 'NN0', 17: 'NN1',
             18: 'NN2', 19: 'NP0', 20: 'ORD', 21: 'PNI', 22: 'PNP', 23: 'PNQ',
             24: 'PNX', 25: 'POS', 26: 'PRF', 27: 'PRP', 28: 'PUL', 29: 'PUN',
             30: 'PUQ', 31: 'PUR', 32: 'TO0', 33: 'UNC', 34: 'VBB', 35: 'VBD',
             36: 'VBG', 37: 'VBI', 38: 'VBN', 39: 'VBZ', 40: 'VDB', 41: 'VDD',
             42: 'VDG', 43: 'VDI', 44: 'VDN', 45: 'VDZ', 46: 'VHB', 47: 'VHD',
             48: 'VHG', 49: 'VHI', 50: 'VHN', 51: 'VHZ', 52: 'VM0', 53: 'VVB',
             54: 'VVD', 55: 'VVG', 56: 'VVI', 57: 'VVN', 58: 'VVZ', 59: 'XX0',
             60: 'ZZ0', 61: 'AJ0-AV0', 62: 'AJ0-VVN', 63: 'AJ0-VVD',
             64: 'AJ0-NN1', 65: 'AJ0-VVG', 66: 'AVP-PRP', 67: 'AVQ-CJS',
             68: 'CJS-PRP', 69: 'CJT-DT0', 70: 'CRD-PNI', 71: 'NN1-NP0',
             72: 'NN1-VVB', 73: 'NN1-VVG', 74: 'NN2-VVZ', 75: 'VVD-VVN'}

TAGtoHASH = {'AJ0': 0, 'AJC': 1, 'AJS': 2, 'AT0': 3, 'AV0': 4, 'AVP': 5,
             'AVQ': 6, 'CJC': 7, 'CJS': 8, 'CJT': 9, 'CRD': 10, 'DPS': 11,
             'DT0': 12, 'DTQ': 13, 'EX0': 14, 'ITJ': 15, 'NN0': 16, 'NN1': 17,
             'NN2': 18, 'NP0': 19, 'ORD': 20, 'PNI': 21, 'PNP': 22, 'PNQ': 23,
             'PNX': 24, 'POS': 25, 'PRF': 26, 'PRP': 27, 'PUL': 28, 'PUN': 29,
             'PUQ': 30, 'PUR': 31, 'TO0': 32, 'UNC': 33, 'VBB': 34, 'VBD': 35,
             'VBG': 36, 'VBI': 37, 'VBN': 38, 'VBZ': 39, 'VDB': 40, 'VDD': 41,
             'VDG': 42, 'VDI': 43, 'VDN': 44, 'VDZ': 45, 'VHB': 46, 'VHD': 47,
             'VHG': 48, 'VHI': 49, 'VHN': 50, 'VHZ': 51, 'VM0': 52, 'VVB': 53,
             'VVD': 54, 'VVG': 55, 'VVI': 56, 'VVN': 57, 'VVZ': 58, 'XX0': 59,
             'ZZ0': 60, 'AJ0-AV0': 61, 'AJ0-VVN': 62, 'AJ0-VVD': 63,
             'AJ0-NN1': 64, 'AJ0-VVG': 65, 'AVP-PRP': 66, 'AVQ-CJS': 67,
             'CJS-PRP': 68, 'CJT-DT0': 69, 'CRD-PNI': 70, 'NN1-NP0': 71,
             'NN1-VVB': 72, 'NN1-VVG': 73, 'NN2-VVZ': 74, 'VVD-VVN': 75}

# -------------------------- Feeding Training Data -----------------------------
def _one_training_file_reader(one_training_file_name):
    """Not yet divided.
    All Initial, T, M have count, not prob.
    So after putting all training file names into this function,
    divide every element(count) by the total number of words-tags pair n."""
    with open(one_training_file_name, 'r') as one_training_file:
        tagged_words_list_in_one_training = one_training_file.readlines()
    # if tagged_words_list_in_one_training[-1] == "\n":
    #     del tagged_words_list_in_one_training[-1]
    global Initial
    global T
    global M
    global TagToCount, TagToCountT
    global AfterThese
    global CountThese

    prev = None
    double_quote_closed = True
    single_quote_closed = True
    these = ()
    # sharp_quote_closed = True  # ‘ = opening quote ’ = closing quote
    for pair_str in tagged_words_list_in_one_training:
        pair_list = pair_str.rstrip().split()
        curr_word = pair_list[0].lower()
        try:
            curr_tag = TAGtoHASH[pair_list[-1]]
        except:
            curr_tag = TAGtoHASH[pair_list[-1].split('-')[1]+'-'+pair_list[-1].split('-')[0]]  # 고쳐 do we have to do this tho?

        # TagToCount
        if curr_tag not in TagToCount:
            TagToCount[curr_tag] = 1
        else:
            TagToCount[curr_tag] += 1
            TagToCountT[curr_tag] += 1
        # Initial Count
        if prev is None:
            Initial[curr_tag] += 1

            # names? 고쳐

        # Transition
        # future denom is count of the yesterday(prev) = TagToCount[prev]
        # Transition Count
        transFromTo = (prev, curr_tag)
        if transFromTo in T:
            T[transFromTo] += 1
        else:
            T[transFromTo] = 1

        # Emission future denominator is count(tag)
        # Emission Count
        if curr_word in M:
            if curr_tag in M[curr_word]:
                M[curr_word][curr_tag] += 1
            else:
                M[curr_word][curr_tag] = 1
        else:
            M[curr_word] = {curr_tag: 1}

        if len(these) != 0:
            these = these[-threshold:]
            if these in AfterThese:
                if curr_tag in AfterThese[these]:
                    AfterThese[these][curr_tag] += 1
                else:
                    AfterThese[these][curr_tag] = 1
                CountThese[these] += 1
            else:
                AfterThese[these] = {curr_tag: 1}
                CountThese[these] = 1
        these += (curr_word,)

        # Incrementing
        if curr_word == '.' or curr_word == '?' or curr_word == '!' or curr_word == ';':# or '’': # or '’' or '"':
            prev = None
        elif curr_word == "'":
            if single_quote_closed:
                single_quote_closed = False
            else:  # it has been opened. Now it is a closing one.
                single_quote_closed = True
                prev = None
        elif curr_word == '"':
            if double_quote_closed:
                double_quote_closed = False
            else:  # it has been opened. Now it is a closing one.
                double_quote_closed = True
                prev = None
        else:
            prev = curr_tag


def _all_training_files_reader(training_name_list:list):
    """Read all training files by calling one_training_file_reader,
    and make the counts stored in Initial, T, M into probabilities."""
    for each_training_file_name in training_name_list:
        _one_training_file_reader(each_training_file_name)

    global Initial
    global T
    global M
    global TagToCount, TagToCountT
    global AfterThese
    global CountThese
    ini = sum(Initial.values())
    for possible_first_tag in Initial:
        if Initial[possible_first_tag] == 0:
            Initial[possible_first_tag] = -11
            continue
        Initial[possible_first_tag] = log(Initial[possible_first_tag] / ini)

    for transFromTo in T:
        if transFromTo[0] is None:
            continue
        T[transFromTo] = log(T[transFromTo] / TagToCountT[transFromTo[0]])

    for word in M:
        for ta in M[word]:
            M[word][ta] = log(M[word][ta] / TagToCount[ta])

    for these in AfterThese:
        for ta in AfterThese[these]:
            AfterThese[these][ta] = log(AfterThese[these][ta]/CountThese[these])


# ---------------------------- Reading Test File -------------------------------

def _if_obvious(original_word: str, t: int, Obvious: dict, SentenceStartsAt):
    """See if word at index t is obvious.
    Add that index and maps to a tag, if so.

    Exclude the cases for ending:
    '.' '!' '?' ';':
    '"' '’':
    because this should be in not only Obvious but before the SentenceStart."""
    word = original_word.lower()
    if word == "‘" or word == "’":  # ‘ = opening quote ’ = closing quote
        Obvious[t] = 'PUQ'
    # elif word =! "'" and word[0] == "'":
    #     Obvious[t] = 'POS'
    elif word == ',' or word == ':':
        Obvious[t] = 'PUN'
    elif original_word.istitle() and len(SentenceStartsAt)>0 and t != SentenceStartsAt[-1]:
        Obvious[t] = 'NP0'
    elif word == 'of':
        Obvious[t] = 'PRF'
    elif word == 'the':
        Obvious[t] = 'AT0'
    elif word == {'he', 'she', 'i', 'you', 'me', 'it', 'him', 'her', 'them', 'they', 'hers', 'theirs', 'ours', 'us', 'mine'}:
        Obvious[t] = 'PNP'   # training 5 last 고쳐'
    elif word in {"my", "your", "their", "her", "our", "its"}:
        Obvious[t] = 'DPS'
    else:
        return

def _test_file_reader(test_file_name:str):
    """
    Obvious = {} map index of E to tag iff obvious and not end of the sentence
    :param test_file_name: string of the test file name
    :return: list of the words in order E
    """
    E = {}
    SentenceStartsAt = []
    Obvious = {}
    with open(test_file_name, 'r') as test_file_obj:
        untagged_lines = test_file_obj.readlines()
    Esize = len(untagged_lines)
    t = 0
    while t < Esize-1:
        word = untagged_lines[t].split()[0]
        E[t] = word
        if word == '.' or word == '!' or word == '?' or word == ';':
            Obvious[t] = 'PUN'
            SentenceStartsAt.append(t+1)
        elif word == '"' or word == '’':
            Obvious[t] = 'PUQ'
            SentenceStartsAt.append(t+1)
        else:
            _if_obvious(word, t, Obvious, SentenceStartsAt)
        t += 1
    # By the end of the loop, it is t=Esize-1, which is the very last index.

    word = untagged_lines[t].split()[0]
    E[t] = word
    if word == '.' or word == '!' or word == '?' or word == ';':
        Obvious[t] = 'PUN'
    elif word == '"' or word == '’':
        Obvious[t] = 'PUQ'
    else:
        _if_obvious(word, t, Obvious, SentenceStartsAt)
    return E, Esize, SentenceStartsAt, Obvious


# --------------------------- Viterbi Functions --------------------------------

def _find_obvious(prev_tag, word) -> list:
    """

    :param prev_tag:
    :param word: observed word
    :return: [] if there is no obvious tag
             [list of possible tags]
              if there is an obvious tag, and the corresponding tag.
    """
    if prev_tag == 'AT0':
        return ['NN0', 'NN1', 'NN2', 'NP0']

def _get_M(E, total_t, i, t) -> float:
    """Return log prob of estimated emission prob for this word.
    Consider this when there's no tag for this word."""
    worst = -11
    word = E[total_t]
    # check if there's a variation of this word.
    if word in M and i in M[word]:
        return M[word][i]     # IF there's a word similar to this, we can return that prob.
    # ELSE: word not in M and i not in M[word] " I have never seen such a word"
    return _check_variation(E, total_t, i, word)


def _check_variation(E, total_t, i, word) -> float:
    """ tnstjorder might matter fixfix
    Return True, log prob of that word's variation emission prob
    """
    v1 = word.lower()
    if v1 in M:
        if i in M[v1]:
            return M[v1][i]
    else:
        temp = copy(total_t)
        seen_these = ()
        added = 0
        while added < threshold and temp > 0:
            seen_these = (E[temp-1],) + seen_these
            temp -= 1
            added += 1
        if seen_these in AfterThese:
            if i in AfterThese[seen_these]:
                return AfterThese[seen_these][i]
    return -11



def _find_max_index(t: int, prob: dict, i: int, E: dict, total_t:int) \
        -> tuple:
    max_index = None  # or can be None instead
    max_probval = -float('inf')  # cux probabilities cannot be negative, can make it -1
    for x in prob[0]:  # = range(len(prob[t-1])):
        transFromTo = (x, i)
        word = E[total_t]  # i is tag
        onlyep_used = True  # NOT only epsilon (fake value) used => only use sth
        m = _get_M(E, total_t, i, t)
        if m > -11 and transFromTo in T:
            curr_val = prob[0][x] + T[transFromTo] + m
            onlyep_used = False
        elif onlyep_used:
            if transFromTo in T and (word not in M or i not in M[word]):
                curr_val = prob[0][x] + T[transFromTo] + m
            else: # rhcu fixfix you dont want to use epsilon if there's at least one
                curr_val = prob[0][x] -11 + m
        else:
            continue
        if curr_val > max_probval:
            max_index = x
            max_probval = curr_val
    return max_index, max_probval


def Viterbi(E, Esize, SentenceStartsAt, Obvious) -> list:
    """
    Make a guess for a whole paragraph(test file)
    E[t] sequence of obs prob for each t
    S is possible values of hidden states
    :param start: starting index
    :param end: ending index (exclusive, as we have filled in already possibly
                                except for the very last index)
    :param prob: the prob dict we have to fill in for each timestep and state
    :param prev: the prev dict we have to fill in for each timestep and state
    :return: None. Instead, it fills in the value inside prob and prev
    """
    global Initial, T, M
    NumberOfSentences = len(SentenceStartsAt)
    # Splitting the test paragraph to sentences
    sentences = []
    # 몇번째 문장 = 1
    sentence = [E[0]]
    for t in range(1, Esize):
        if t in SentenceStartsAt:
            sentences.append(sentence)
            sentence = [E[t]]  # reset
        else:
            sentence += [E[t]]
    sentences.append(sentence)
    total_t = 0
    results = []
    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len == 1:  # this means this sentence ends by 1 char.
            results += [Obvious[total_t]]
            total_t += 1
            continue
        prob = {0: {}}
        prev = {0: {}}
           
        # BASE CASE
        # Determine the values for time step 0 (BASE CASE) # 고쳐... M[][]로
        if total_t in Obvious:
            otag = Obvious[total_t]
            h = TAGtoHASH[otag]
            prob[0][h] = 0
            prev[0][h] = [otag]
        else:
            for i in HASHtoTAG:
                word = E[total_t]
                if word in M and i in M[word]:
                    prob[0][i] = Initial[i] + M[word][i] # initial_prob * emission_prob
                else:  # elif word not in M or i not in M[word]:
                    prob[0][i] = Initial[i] - _get_M(E, total_t, i, 0)
                prev[0][i] = [HASHtoTAG[i]]
        # Normalize and put it back
        # sum_by_row = -sum(prob[0].values())  # timestep = t
        # for i in prob[0]:
        #     prob[0][i] = prob[0][i] / sum_by_row
        total_t += 1


        # Revursive case
        # for time steps 1 to len(E)-1,
        # find each curr state's most likely prior state x (RECURSIVE CASE!)
        t = 1
        while t < sentence_len:
            prob[1] = {}
            prev[1] = {}
            if total_t in Obvious:
                otag = Obvious[total_t]
                h = TAGtoHASH[otag]
                prob[1][h] = 0
                max_former = max(prob[0], key=prob[0].get)
                prev[1][h] = prev[0][max_former] + [otag]
                total_t += 1
                t += 1
                prob[0] = prob[1]
                prev[0] = prev[1]
                continue
            for i in HASHtoTAG:
                max_former, max_probval = _find_max_index(t, prob, i, E, total_t)
                prob[1][i] = max_probval  # dㅝㄴ래 0이상인 것만 넣었는데 없앰.. fixfix
                prev[1][i] = prev[0][max_former] + [HASHtoTAG[i]]
            prob[0] = prob[1]
            prev[0] = prev[1]
            total_t += 1
            t += 1
            # Normalize and put it back
            # sum_by_row = -sum(prob[t].values())  # timestep = t
            # for i in prob[t]:
            #     prob[t][i] = prob[t][i] / sum_by_row
        # Handling the known i for each end. for the unknown VL, we will still do this
        # by just considering it as PUN=1, but will be better handled outside later!
        # prob[end][TAGtoHASH[SentenceEndsAt[end]]] = 0
        # prev[end][TAGtoHASH[SentenceEndsAt[end]]] = max(prob[end-1], key=prob[end-1].get)
        last = _find_last(prob, 1, total_t-1, Obvious)
        final_path = prev[1][last]
        results += final_path
    return results


# ----------------------------- Fill in the output -----------------------------

def _find_last(prob, end, total_t, Obvious) -> int:
    """Find the largest probability having HASH for the last observation."""
    if total_t in Obvious:
        return TAGtoHASH[Obvious[total_t]]
    # return prob[end].index(max(prob[end]))  # max(prob[end], key=prob[end].get)
    return max(prob[end], key=prob[end].get)


def _write_answers(E, Esize, results, output_filename):
    combined = []
    for i in range(Esize):
        combined.append(f"{E[i]} : {results[i]}\n")
    with open(output_filename, 'w') as output_file_obj:
        output_file_obj.writelines(combined)

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    try:
        parameters = sys.argv
        training_list = parameters[
                        parameters.index("-d") + 1:parameters.index("-t")]
        test_file = parameters[parameters.index("-t") + 1]
        output_file = parameters[parameters.index("-o") + 1]
    except:
        training_list = ["data/training1.txt", "data/training2.txt", "data/training3.txt", "data/training4.txt", "data/training5.txt", "data/new.txt"]
        test_file = "/validation/given_test1.txt"
        output_file = "/validation/output1.txt"
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)
    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
