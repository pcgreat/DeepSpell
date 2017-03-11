# encoding: utf-8
'''
Created on Nov 26, 2015

@author: tal

Based in part on:
Learn math - https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py

See https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm
'''

from __future__ import print_function, division, unicode_literals

import errno
import itertools
import json
import logging
import os
import re
from collections import Counter
from hashlib import sha256

import numpy as np
import requests
import tensorflow as tf
from numpy import zeros as np_zeros  # pylint:disable=no-name-in-module
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, \
    seed as random_seed, rand

# Set a logger for the module
from seq2seq_model import BabySeq2Seq, test

LOGGER = logging.getLogger(__name__)  # Every log will use the module name
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)

random_seed(123)  # Reproducibility


class Configuration(object):
    """Dump stuff here"""


CONFIG = Configuration()
# pylint:disable=attribute-defined-outside-init
# Parameters for the model:
CONFIG.input_layers = 2
CONFIG.output_layers = 2
CONFIG.amount_of_dropout = 0.2
CONFIG.hidden_size = 128
CONFIG.initialization = "he_normal"  # : Gaussian initialization scaled by fan-in (He et al., 2014)
CONFIG.number_of_chars = 29
CONFIG.max_input_len = 30
CONFIG.inverted = True

# parameters for the training:
CONFIG.number_of_iterations = 20000
CONFIG.epochs_per_iteration = 500
CONFIG.batch_size = 256  # As the model changes in size, play with the batch size to best fit the process in memory
CONFIG.samples_per_epoch = 1000000
CONFIG.number_of_validation_samples = 10000
# pylint:enable=attribute-defined-outside-init

DIGEST = sha256(json.dumps(CONFIG.__dict__, sort_keys=True).encode("utf-8")).hexdigest()

# Parameters for the dataset
MIN_INPUT_LEN = 5
AMOUNT_OF_NOISE = 0.3 / CONFIG.max_input_len
PADDING = "☕"
GO = "→"
CHARS = list("abcdefghijklmnopqrstuvwxyz ") + [PADDING] + [GO]

DATA_FILES_PATH = "data/wikivoyage/"
DATA_FILES_FULL_PATH = os.path.expanduser(DATA_FILES_PATH)
DATA_FILES_URL = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
NEWS_FILE_NAME_COMPRESSED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.shuffled.gz")  # 1.1 GB
NEWS_FILE_NAME_ENGLISH = "input.txt"  # "news.2013.en.shuffled"
NEWS_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, NEWS_FILE_NAME_ENGLISH)
NEWS_FILE_NAME_CLEAN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.clean")
NEWS_FILE_NAME_FILTERED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.filtered")
NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split")
NEWS_FILE_NAME_TRAIN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.train")
NEWS_FILE_NAME_VALIDATE = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.validate")
CHAR_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "char_frequency.json")
SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_e{}.h5")  # an HDF5 file

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)  # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(
    r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769), chr(832),
                                                        chr(833), chr(2387), chr(5151),
                                                        chr(5152), chr(65344), chr(8242)),
    re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(ALLOWED_CURRENCIES)),
                              re.UNICODE)
RE_PUNCT_CLEANER = re.compile(r'[{}]'.format(re.escape(ALLOWED_PUNCTUATION)),
                              re.UNICODE)


# pylint:disable=invalid-name

def download_the_news_data():
    """Download the news data"""
    LOGGER.info("Downloading")
    try:
        os.makedirs(os.path.dirname(NEWS_FILE_NAME_COMPRESSED))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(NEWS_FILE_NAME_COMPRESSED, "wb") as output_file:
        response = requests.get(DATA_FILES_URL, stream=True)
        total_length = response.headers.get('content-length')
        downloaded = percentage = 0
        print("»" * 100)
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=4096):
            downloaded += len(data)
            output_file.write(data)
            new_percentage = 100 * downloaded // total_length
            if new_percentage > percentage:
                print("☑", end="")
                percentage = new_percentage
    print()


def uncompress_data():
    """Uncompress the data files"""
    import gzip
    with gzip.open(NEWS_FILE_NAME_COMPRESSED, 'rb') as compressed_file:
        with open(NEWS_FILE_NAME_COMPRESSED[:-3], 'wb') as outfile:
            outfile.write(compressed_file.read())


def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    if rand() < amount_of_noise * len(a_string):
        # Replace a character with a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if rand() < amount_of_noise * len(a_string):
        # Delete a character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < CONFIG.max_input_len and rand() < amount_of_noise * len(a_string):
        # Add a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
    if rand() < amount_of_noise * len(a_string):
        # Transpose 2 characters
        random_char_position = random_randint(len(a_string) - 1)
        a_string = (
            a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
            a_string[random_char_position + 2:])
    return a_string


def _vectorize(questions, answers, ctable):
    """Vectorize the data as numpy arrays"""
    len_of_questions = len(questions)
    X = np_zeros((len_of_questions, CONFIG.max_input_len), dtype=np.int32)
    for i in range(len(questions)):
        sentence = questions.pop()
        for j, c in enumerate(sentence):
            try:
                X[i, j] = ctable.char_indices[c]
            except KeyError:
                pass  # Padding
    y = np_zeros((len_of_questions, CONFIG.max_input_len + 1), dtype=np.int32)
    y_weig = np_zeros((len_of_questions, CONFIG.max_input_len + 1), dtype=np.int32)
    for i in range(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            try:
                y[i, j] = ctable.char_indices[c]
                y_weig[i, j] = 1.
            except KeyError:
                pass  # Padding
    return X.T, y.T, y_weig.T


def vectorize(questions, answers, chars=None):
    """Vectorize the questions and expected answers"""
    print('Vectorization...')
    chars = chars or CHARS
    ctable = CharacterTable(chars)
    X, y = _vectorize(questions, answers, ctable)
    # Explicitly set apart 10% for validation data that we never train over
    split_at = int(len(X) - len(X) / 10)
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_val, y_train, y_val, CONFIG.max_input_len, ctable


def generate_model(batch_size, chars=None):
    return BabySeq2Seq(len(chars), len(chars), [(CONFIG.max_input_len, CONFIG.max_input_len + 1)],
                       size=CONFIG.hidden_size, num_layers=2, batch_size=batch_size)


class Colors(object):
    """For nicer printouts"""
    green = '\033[92m'
    red = '\033[91m'
    close = '\033[0m'


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    @property
    def size(self):
        """The number of chars"""
        return len(self.chars)

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool)  # pylint:disable=no-member
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=False):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X if x is not None)


def generator(file_name):
    """Returns a tuple (inputs, targets)
    All arrays should contain the same number of samples.
    The generator is expected to loop over its data indefinitely.
    An epoch finishes when  samples_per_epoch samples have been seen by the model.
    """
    ctable = CharacterTable(read_top_chars())
    batch_of_answers = []
    while True:
        with open(file_name) as answers:
            for answer in answers:
                batch_of_answers.append(answer.strip())
                if len(batch_of_answers) == CONFIG.batch_size:
                    random_shuffle(batch_of_answers)
                    batch_of_questions = []
                    for answer_index, answer in enumerate(batch_of_answers):
                        question, answer = generate_question(answer)
                        batch_of_answers[answer_index] = answer
                        assert len(answer) == CONFIG.max_input_len + 1
                        question = question[::-1] if CONFIG.inverted else question
                        batch_of_questions.append(question)
                    X, y, weights = _vectorize(batch_of_questions, batch_of_answers, ctable)  # X = y = (128, 30, 28)
                    yield X, y, weights
                    batch_of_answers = []


def print_random_predictions(model, session):
    """Select 10 samples from the validation set at random so we can visualize errors"""
    print()
    ctable = CharacterTable(read_top_chars())
    for batch_encoder_inputs, batch_decoder_inputs, decoder_weights in generator(NEWS_FILE_NAME_VALIDATE):
        # print_random_predictions(model, batch_encoder_inputs, batch_decoder_inputs, decoder_weights)
        _, results = test(model, session, batch_encoder_inputs, batch_decoder_inputs, decoder_weights)
        batch_encoder_inputs = batch_encoder_inputs.T
        batch_decoder_inputs = batch_decoder_inputs.T
        break

    for _ in range(10):
        ind = random_randint(0, len(batch_encoder_inputs))
        rowX, rowy, preds = batch_encoder_inputs[np.array([ind])], batch_decoder_inputs[np.array([ind])], results[
            np.array([ind])]  # pylint:disable=no-member
        # preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        if CONFIG.inverted:
            print('Q', q[::-1])  # inverted back!
        else:
            print('Q', q)
        print('A', correct)
        print(Colors.green + '☑' + Colors.close if correct == guess else Colors.red + '☒' + Colors.close, guess)
        print('---')
    print()


def eval_model(model, session):
    eval_losses = []
    for batch_encoder_inputs, batch_decoder_inputs, decoder_weights in generator(NEWS_FILE_NAME_VALIDATE):
        eval_loss, _ = test(model, session, batch_encoder_inputs, batch_decoder_inputs, decoder_weights)
        eval_losses.append(eval_loss)
    print(np.mean(eval_losses))


def itarative_train(model, from_file):
    """
    Iterative training of the model
     - To allow for finite RAM...
     - To allow infinite training data as the training noise is injected in runtime
    """
    # model.fit_generator(generator(NEWS_FILE_NAME_TRAIN), samples_per_epoch=CONFIG.samples_per_epoch,
    #                     nb_epoch=CONFIG.epochs_per_iteration,
    #                     verbose=1, callbacks=[ON_EPOCH_END_CALLBACK, ],
    #                     validation_data=generator(NEWS_FILE_NAME_VALIDATE),
    #                     nb_val_samples=CONFIG.number_of_validation_samples,
    #                     class_weight=None, max_q_size=10, nb_worker=1,
    #                     pickle_safe=False)
    step = 0
    with tf.Session() as session:
        if from_file:
            checkpoint_path = os.path.join(DATA_FILES_PATH, "spell_corrector.ckpt") + "-1600"
            print("loading model from:", checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(session, checkpoint_path)
        else:
            session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(DATA_FILES_PATH)

        for batch_encoder_inputs, batch_decoder_inputs, decoder_weights in generator(NEWS_FILE_NAME_TRAIN):
            _, loss, summary_ops = model.step(session, batch_encoder_inputs, batch_decoder_inputs, decoder_weights,
                                              test=False)  # no outputs in training
            train_writer.add_summary(summary_ops, global_step=step)
            if step % 10 == 0:
                print("Iteration:", step, "; losses:", loss)
            if step % 100 == 0:
                print("eval loss:", eval_model(model, session))
                print_random_predictions(model, session)
                checkpoint_path = os.path.join(DATA_FILES_PATH, "spell_corrector.ckpt")
                model.saver.save(session, checkpoint_path, global_step=step)
            step = step + 1


def clean_text(text):
    """Clean the text - remove unwanted chars, fold punctuation etc."""
    result = text.lower()
    result = RE_DASH_FILTER.sub('-', result)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    result = RE_PUNCT_CLEANER.sub(" ", result)
    result = RE_BASIC_CLEANER.sub('', result)
    result = re.sub(r"[0-9]+", "", result)
    result = re.sub(r"\n+", r"\n", result)
    result = re.sub(r"[ ]+", r" ", result)
    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', result.strip())

    return result


def preprocesses_data_clean():
    """Pre-process the data - step 1 - cleanup"""
    with open(NEWS_FILE_NAME_CLEAN, "wb") as clean_data:
        for line in open(NEWS_FILE_NAME):
            decoded_line = line
            cleaned_line = clean_text(decoded_line)
            encoded_line = cleaned_line.encode("utf-8")
            clean_data.write(encoded_line + b"\n")


def preprocesses_data_analyze_chars():
    """Pre-process the data - step 2 - analyze the characters"""
    counter = Counter()
    LOGGER.info("Reading data:")
    for line in open(NEWS_FILE_NAME_CLEAN):
        decoded_line = line
        counter.update(decoded_line)
    # data = open(NEWS_FILE_NAME_CLEAN).read().decode('utf-8')
    #     LOGGER.info("Read.\nCounting characters:")
    #     counter = Counter(data.replace("\n", ""))
    LOGGER.info("Done.\nWriting to file:")
    with open(CHAR_FREQUENCY_FILE_NAME, 'w') as output_file:
        output_file.write(json.dumps(counter))
    most_popular_chars = {key for key, _value in counter.most_common(CONFIG.number_of_chars)}
    LOGGER.info("The top %s chars are:", CONFIG.number_of_chars)
    LOGGER.info("".join(sorted(most_popular_chars)))


def read_top_chars():
    """Read the top chars we saved to file"""
    # chars = json.loads(open(CHAR_FREQUENCY_FILE_NAME).read())
    # counter = Counter(chars)
    # most_popular_chars = {key for key, _value in counter.most_common(CONFIG.number_of_chars)}
    # return most_popular_chars
    return set(CHARS)


def preprocesses_data_filter():
    """Pre-process the data - step 3 - filter only sentences with the right chars"""
    most_popular_chars = read_top_chars()
    LOGGER.info("Reading and filtering data:")
    with open(NEWS_FILE_NAME_FILTERED, "w") as output_file:
        for line in open(NEWS_FILE_NAME_CLEAN):
            decoded_line = line
            if decoded_line.strip() and not bool(set(decoded_line) - most_popular_chars - set("\n")):
                output_file.write(line)
    LOGGER.info("Done.")


def read_filtered_data():
    """Read the filtered data corpus"""
    LOGGER.info("Reading filtered data:")
    lines = open(NEWS_FILE_NAME_FILTERED).read().split("\n")
    LOGGER.info("Read filtered data - %s lines", len(lines))
    return lines


def preprocesses_split_lines():
    """Preprocess the text by splitting the lines between min-length and max_length
    I don't like this step:
      I think the start-of-sentence is important.
      I think the end-of-sentence is important.
      Sometimes the stripped down sub-sentence is missing crucial context.
      Important NGRAMs are cut (though given enough data, that might be moot).
    I do this to enable batch-learning by padding to a fixed length.
    """
    LOGGER.info("Reading filtered data:")
    answers = set()
    with open(NEWS_FILE_NAME_SPLIT, "wb") as output_file:
        for _line in open(NEWS_FILE_NAME_FILTERED):
            line = _line
            while len(line) > MIN_INPUT_LEN:
                if len(line) <= CONFIG.max_input_len:
                    answer = line
                    line = ""
                else:
                    space_location = line.rfind(" ", MIN_INPUT_LEN, CONFIG.max_input_len - 1)
                    if space_location > -1:
                        answer = line[:space_location]
                        line = line[len(answer) + 1:]
                    else:
                        space_location = line.rfind(" ")  # no limits this time
                        if space_location == -1:
                            break  # we are done with this line
                        else:
                            line = line[space_location + 1:]
                            continue
                answers.add(answer)
                output_file.write(answer.encode('utf-8') + b"\n")


def preprocesses_split_lines2():
    """Preprocess the text by splitting the lines between min-length and max_length
    Alternative split.
    """
    LOGGER.info("Reading filtered data:")
    answers = set()
    with open(NEWS_FILE_NAME_SPLIT, "w") as output_file:
        for encoded_line in open(NEWS_FILE_NAME_FILTERED):
            line = encoded_line
            if CONFIG.max_input_len >= len(line) > MIN_INPUT_LEN:
                answers.add(line)
                output_file.write(encoded_line)
    LOGGER.info("There are %s 'answers' (sub-sentences)", len(answers))
    LOGGER.info("Here are some examples:")
    for answer in itertools.islice(answers, 10):
        LOGGER.info(answer)
    with open(NEWS_FILE_NAME_SPLIT, "w") as output_file:
        output_file.write("".join(answers).encode('utf-8'))


def preprocess_partition_data():
    """Set asside data for validation"""
    answers = [x for x in open(NEWS_FILE_NAME_SPLIT).read().split("\n") if x]
    print('shuffle', end=" ")
    random_shuffle(answers)
    print("Done")
    # Explicitly set apart 10% for validation data that we never train over
    split_at = len(answers) - len(answers) // 10
    with open(NEWS_FILE_NAME_TRAIN, "w") as output_file:
        output_file.write("\n".join(answers[:split_at]))
    with open(NEWS_FILE_NAME_VALIDATE, "w") as output_file:
        output_file.write("\n".join(answers[split_at:]))


def generate_question(answer):
    """Generate a question by adding noise"""
    question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
    # Add padding:
    question += PADDING * (CONFIG.max_input_len - len(question))
    answer += PADDING * (CONFIG.max_input_len - len(answer))
    answer = GO + answer
    return question, answer


def generate_news_data():
    """Generate some news data"""
    print("Generating Data")
    answers = open(NEWS_FILE_NAME_SPLIT).read().split("\n")
    questions = []
    print('shuffle', end=" ")
    random_shuffle(answers)
    print("Done")
    for answer_index, answer in enumerate(answers):
        question, answer = generate_question(answer)
        answers[answer_index] = answer
        assert len(answer) == CONFIG.max_input_len
        if random_randint(100000) == 8:  # Show some progress
            print(len(answers))
            print("answer:   '{}'".format(answer))
            print("question: '{}'".format(question))
            print()
        question = question[::-1] if CONFIG.inverted else question
        questions.append(question)

    return questions, answers


def train_speller(from_file=None):
    """Train the speller"""
    model = generate_model(CONFIG.batch_size, chars=CHARS)
    itarative_train(model, from_file)


if __name__ == '__main__':
    # download_the_news_data()
    # uncompress_data()
    # preprocesses_data_clean()
    # preprocesses_data_analyze_chars()
    # preprocesses_data_filter()
    # preprocesses_split_lines()  # --- Choose this step or:
    # # preprocesses_split_lines2()
    # preprocess_partition_data()
    # # train_speller(os.path.join(DATA_FILES_FULL_PATH, "keras_spell_e15.h5"))

    train_speller(True)
